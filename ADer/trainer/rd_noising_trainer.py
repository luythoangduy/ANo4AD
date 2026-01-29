"""
Trainer for RD with Adaptive Noising - Two-Phase Training Framework.

Phase 1: Build memory bank from teacher features (no training)
Phase 2: Train decoder with adaptive noise injection based on memory bank influence

This trainer extends the RD trainer with:
- Memory bank construction phase
- Adaptive noise injection during training
- Influence-based analysis for noise generation
"""
import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
import torch.nn.functional as F
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.util import log_wandb, save_wandb_model, finish_wandb
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp


@TRAINER.register_module
class RDNoisingTrainer(BaseTrainer):
    """
    Trainer for RD with Adaptive Noising.
    
    Two-phase training:
    1. Memory Bank Construction: Extract teacher features and build memory banks
    2. Training: Train decoder with adaptive noise based on memory bank influence
    """

    def __init__(self, cfg):
        super(RDNoisingTrainer, self).__init__(cfg)
        
        # Memory bank construction flag
        self.memory_bank_built = False
        
        # Get noise configuration from config
        self.noise_enabled = getattr(cfg.trainer, 'noise_enabled', True)
        self.noise_warmup_epochs = getattr(cfg.trainer, 'noise_warmup_epochs', 0)
        
        # Memory bank sampling configuration
        self.sampling_method = getattr(cfg.trainer, 'sampling_method', 'auto')
        self.max_features_for_greedy = getattr(cfg.trainer, 'max_features_for_greedy', 100000)
        self.coreset_device = getattr(cfg.trainer, 'coreset_device', 'auto')
        
        log_msg(self.logger, '='*50)
        log_msg(self.logger, 'RD Noising Trainer Initialized')
        log_msg(self.logger, f'Noise Enabled: {self.noise_enabled}')
        log_msg(self.logger, f'Noise Warmup Epochs: {self.noise_warmup_epochs}')
        log_msg(self.logger, f'Sampling Method: {self.sampling_method}')
        log_msg(self.logger, f'Max Features for Greedy: {self.max_features_for_greedy}')
        log_msg(self.logger, f'Coreset Device: {self.coreset_device}')
        log_msg(self.logger, '='*50)

    def set_input(self, inputs):
        """Set input data."""
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]

    def build_memory_bank(self):
        """
        Phase 1: Build memory bank from teacher features.
        
        This should be called before training starts.
        """
        if self.memory_bank_built:
            log_msg(self.logger, 'Memory bank already built, skipping...')
            return
        
        log_msg(self.logger, '='*50)
        log_msg(self.logger, 'Phase 1: Building Memory Bank')
        log_msg(self.logger, f'Coreset computation device: {self.coreset_device}')
        log_msg(self.logger, '='*50)
        
        # Get the actual model (handle DDP wrapper)
        if self.cfg.dist:
            model = self.net.module
        else:
            model = self.net
        
        # Build memory bank with configurable sampling
        model.build_memory_bank(
            self.train_loader, 
            device=f'cuda:{self.cfg.local_rank}',
            sampling_method=self.sampling_method,
            max_features_for_greedy=self.max_features_for_greedy,
            coreset_device=self.coreset_device
        )
        
        self.memory_bank_built = True
        
        log_msg(self.logger, 'Memory bank construction complete!')
        log_msg(self.logger, '='*50)

    def forward(self):
        """Forward pass with adaptive noise (during training)."""
        # Determine if noise should be applied based on warmup
        apply_noise = self.noise_enabled and self.epoch >= self.noise_warmup_epochs
        
        # Forward pass
        self.feats_t, self.feats_s, self.noise_info = self.net(self.imgs, apply_noise=apply_noise)

    def optimize_parameters(self):
        """Optimization step with cosine loss."""
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        
        with self.amp_autocast():
            self.forward()
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
        
        self.backward_term(loss_cos, self.optim)
        update_log_term(
            self.log_terms.get('cos'),
            reduce_tensor(loss_cos, self.world_size).clone().detach().item(),
            1,
            self.master
        )
        
        # Log noise info if available (now single map instead of list)
        if self.noise_info.get('influence_map') is not None:
            avg_influence = self.noise_info['influence_map'].mean()
            avg_noise_std = self.noise_info['noise_std_map'].mean()
            update_log_term(self.log_terms.get('influence'), avg_influence.item(), 1, self.master)
            update_log_term(self.log_terms.get('noise_std'), avg_noise_std.item(), 1, self.master)

    def train(self):
        """Training loop with two phases."""
        # Phase 1: Build memory bank before training
        self.build_memory_bank()
        
        # Phase 2: Training with adaptive noise
        log_msg(self.logger, '='*50)
        log_msg(self.logger, 'Phase 2: Training with Adaptive Noise')
        log_msg(self.logger, '='*50)
        
        self.reset(isTrain=True)
        while self.iter < self.iter_full:
            self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
            train_length = self.cfg.data.train_size
            train_loader = iter(self.train_loader)
            
            for batch_idx in range(train_length):
                if self.iter >= self.iter_full:
                    break
                
                # ---------- data ----------
                t1 = get_timepc()
                train_data = next(train_loader)
                self.set_input(train_data)
                t2 = get_timepc()
                update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
                
                # ---------- optimization ----------
                self.optimize_parameters()
                self.scheduler_step(self.iter)
                t3 = get_timepc()
                update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
                update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
                
                # ---------- log ----------
                self.iter += 1
                if self.master:
                    if self.iter % self.cfg.logging.train_log_per == 0:
                        noise_status = 'ON' if (self.noise_enabled and self.epoch >= self.noise_warmup_epochs) else 'OFF'
                        msg = able(self.progress.get_msg(
                            self.iter, self.iter_full,
                            self.iter / train_length, self.iter_full / train_length
                        ), self.master, None)
                        msg = f'{msg} [Noise: {noise_status}]'
                        log_msg(self.logger, msg)
                        
                        if self.writer:
                            for k, v in self.log_terms.items():
                                self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
                            self.writer.flush()
                        
                        # Log to WandB
                        if hasattr(self.cfg, 'wandb') and self.cfg.wandb.enable:
                            wandb_metrics = {k: v.val for k, v in self.log_terms.items()}
                            wandb_metrics['epoch'] = self.epoch
                            wandb_metrics['noise_enabled'] = float(self.noise_enabled and self.epoch >= self.noise_warmup_epochs)
                            log_wandb(self.cfg, wandb_metrics, step=self.iter, prefix='train')
                
                if self.iter % self.cfg.logging.train_reset_log_per == 0:
                    self.reset(isTrain=True)
            
            # ---------- epoch end ----------
            self.epoch += 1
            if self.cfg.dist and self.dist_BN != '':
                distribute_bn(self.net, self.world_size, self.dist_BN)
            self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
            
            if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
                self.test()
            else:
                self.test_ghost()
            
            self.cfg.total_time = get_timepc() - self.cfg.task_start_time
            total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
            eta_time_str = str(datetime.timedelta(
                seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))
            ))
            log_msg(self.logger, f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
            
            self.save_checkpoint()
            self.reset(isTrain=True)
        
        self._finish()

    @torch.no_grad()
    def test(self):
        """Test with anomaly detection evaluation."""
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        
        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            
            # Forward pass without noise for testing
            self.feats_t, self.feats_s, _ = self.net(self.imgs, apply_noise=False)
            
            # Compute loss for logging
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            update_log_term(
                self.log_terms.get('cos'),
                reduce_tensor(loss_cos, self.world_size).clone().detach().item(),
                1,
                self.master
            )
            
            # Get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(
                self.feats_t, self.feats_s,
                [self.imgs.shape[2], self.imgs.shape[3]],
                uni_am=False, amap_mode='add', gaussian_sigma=4
            )
            
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            
            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(
                    self.img_path, self.imgs,
                    self.imgs_mask.cpu().numpy().astype(int),
                    anomaly_map, self.cfg.model.name, root_out,
                    self.cfg.data.root.split('/')[1]
                )
            
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))
            
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            
            # ---------- log ----------
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                    log_msg(self.logger, msg)
        
        # Merge results
        if self.cfg.dist:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f'{self.tmp_dir}/*.pth')
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except:
                                time.sleep(1)
                        valid_results = True
        else:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        
        if self.master:
            results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center")
            log_msg(self.logger, f'\n{msg}')
            
            # Log test metrics to WandB
            if hasattr(self.cfg, 'wandb') and self.cfg.wandb.enable:
                wandb_test_metrics = {}
                for idx, cls_name in enumerate(self.cls_names):
                    for metric in self.metrics:
                        metric_result = self.metric_recorder[f'{metric}_{cls_name}'][-1]
                        wandb_test_metrics[f'test/{metric}_{cls_name}'] = metric_result
                    if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1:
                        for metric in self.metrics:
                            metric_result_avg = self.metric_recorder[f'{metric}_Avg'][-1]
                            wandb_test_metrics[f'test/{metric}_Avg'] = metric_result_avg
                wandb_test_metrics['test/epoch'] = self.epoch
                log_wandb(self.cfg, wandb_test_metrics, step=self.iter, prefix='test')

    def save_checkpoint(self):
        """Save checkpoint including memory bank status."""
        if self.master:
            # Get model state
            if self.cfg.dist:
                net_state = trans_state_dict(self.net.module.state_dict(), dist=False)
                memory_banks = self.net.module.memory_banks
            else:
                net_state = trans_state_dict(self.net.state_dict(), dist=False)
                memory_banks = self.net.memory_banks
            
            checkpoint_info = {
                'net': net_state,
                'optimizer': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.loss_scaler.state_dict() if self.loss_scaler else None,
                'iter': self.iter,
                'epoch': self.epoch,
                'metric_recorder': self.metric_recorder,
                'total_time': self.cfg.total_time,
                'memory_bank_built': self.memory_bank_built,
                'memory_banks': {k: v.cpu() for k, v in memory_banks.items()},
            }
            
            save_path = f'{self.cfg.logdir}/ckpt.pth'
            torch.save(checkpoint_info, save_path)
            torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/net.pth')
            
            # Save best model to WandB
            if hasattr(self.cfg, 'wandb') and self.cfg.wandb.enable and self.cfg.wandb.log_model:
                if len(self.cls_names) > 1:
                    auroc_avg = self.metric_recorder.get('mAUROC_sp_max_Avg', [0])[-1]
                else:
                    auroc_avg = self.metric_recorder.get(f'mAUROC_sp_max_{self.cls_names[0]}', [0])[-1]
                
                metadata = {
                    'epoch': self.epoch,
                    'mAUROC_sp_max': auroc_avg,
                    'memory_bank_built': self.memory_bank_built,
                }
                save_wandb_model(self.cfg, save_path, metadata=metadata)
            
            if self.epoch % self.cfg.trainer.test_per_epoch == 0:
                torch.save(checkpoint_info['net'], f'{self.cfg.logdir}/net_{self.epoch}.pth')

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint including memory bank."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if self.cfg.dist:
            self.net.module.load_state_dict(checkpoint['net'])
        else:
            self.net.load_state_dict(checkpoint['net'])
        
        # Load memory banks
        if 'memory_banks' in checkpoint:
            device = f'cuda:{self.cfg.local_rank}'
            if self.cfg.dist:
                self.net.module.memory_banks = {k: v.to(device) for k, v in checkpoint['memory_banks'].items()}
                self.net.module.memory_bank_built = checkpoint.get('memory_bank_built', True)
            else:
                self.net.memory_banks = {k: v.to(device) for k, v in checkpoint['memory_banks'].items()}
                self.net.memory_bank_built = checkpoint.get('memory_bank_built', True)
            self.memory_bank_built = checkpoint.get('memory_bank_built', True)
        
        # Load optimizer and scheduler
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.loss_scaler and checkpoint['scaler']:
            self.loss_scaler.load_state_dict(checkpoint['scaler'])
        
        self.iter = checkpoint['iter']
        self.epoch = checkpoint['epoch']
        self.metric_recorder = checkpoint['metric_recorder']
        
        log_msg(self.logger, f'Loaded checkpoint from {checkpoint_path}')
        log_msg(self.logger, f'Resuming from epoch {self.epoch}, iter {self.iter}')
        log_msg(self.logger, f'Memory bank built: {self.memory_bank_built}')
