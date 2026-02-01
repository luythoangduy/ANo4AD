"""
Trainer for RD++ (Revisit RD) with Adaptive Noising - Two-Phase Training Framework.

Combines:
- RD++ training: Multi-projection layer, SSOT loss, reconstruct loss, contrast loss
- Adaptive Noising: Memory bank based influence analysis for noise injection

Phase 1: Build memory bank from teacher features (no training)
Phase 2: Train decoder with adaptive noise + RD++ losses
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
class RDPPNoisingTrainer(BaseTrainer):
    """
    Trainer for RD++ with Adaptive Noising.
    
    Two-phase training:
    1. Memory Bank Construction: Extract teacher features and build memory banks
    2. Training: Train decoder with adaptive noise + RD++ projection losses
    """

    def __init__(self, cfg):
        super(RDPPNoisingTrainer, self).__init__(cfg)
        
        # Memory bank construction flag
        self.memory_bank_built = False
        
        # Get noise configuration from config
        self.noise_enabled = getattr(cfg.trainer, 'noise_enabled', True)
        self.noise_warmup_epochs = getattr(cfg.trainer, 'noise_warmup_epochs', 0)
        
        # Memory bank sampling configuration
        self.sampling_method = getattr(cfg.trainer, 'sampling_method', 'auto')
        self.max_features_for_greedy = getattr(cfg.trainer, 'max_features_for_greedy', 100000)
        self.coreset_device = getattr(cfg.trainer, 'coreset_device', 'auto')
        
        # RD++ specific optimizers
        self.optim.proj_opt = get_optim(
            cfg.optim.proj_opt.kwargs if hasattr(cfg.optim, 'proj_opt') else cfg.optim.kwargs,
            self.net.proj_layer,
            lr=cfg.optim.lr
        )
        
        # Temporarily remove proj_layer to create distill optimizer for other components
        proj_layer = self.net.proj_layer
        self.net.proj_layer = None
        self.optim.distill_opt = get_optim(
            cfg.optim.distill_opt.kwargs if hasattr(cfg.optim, 'distill_opt') else cfg.optim.kwargs,
            self.net,
            lr=cfg.optim.lr * 5
        )
        self.net.proj_layer = proj_layer
        
        log_msg(self.logger, '='*50)
        log_msg(self.logger, 'RD++ Noising Trainer Initialized')
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
        
        # Optional: external noise image (for compatibility with original RD++)
        if 'img_noise' in inputs:
            self.img_noise = inputs['img_noise'].cuda()
        else:
            self.img_noise = None

    def build_memory_bank(self):
        """
        Phase 1: Build memory bank from teacher features.
        """
        if self.memory_bank_built:
            log_msg(self.logger, 'Memory bank already built, skipping...')
            return
        
        if hasattr(self.net, 'module'):
            net = self.net.module
        else:
            net = self.net
        
        log_msg(self.logger, '='*50)
        log_msg(self.logger, 'Phase 1: Building Memory Bank')
        log_msg(self.logger, '='*50)
        
        # Build memory bank using model's method
        net.build_memory_bank(
            train_loader=self.train_loader,
            device='cuda',
            sampling_method=self.sampling_method,
            max_features_for_greedy=self.max_features_for_greedy,
            coreset_device=self.coreset_device
        )
        
        self.memory_bank_built = True
        log_msg(self.logger, 'Memory bank construction complete!')
        log_msg(self.logger, '='*50)

    def forward(self):
        """Forward pass."""
        # Use adaptive noise if enabled and memory bank is built
        apply_noise = self.noise_enabled and self.memory_bank_built
        
        # Check warmup
        if self.noise_warmup_epochs > 0 and self.epoch < self.noise_warmup_epochs:
            apply_noise = False
        
        self.feats_t, self.feats_s, self.L_proj = self.net(
            self.imgs, 
            img_noise=self.img_noise,
            apply_noise=apply_noise
        )

    def backward_term(self, loss_term, optim):
        """Backward pass with RD++ dual optimizer."""
        optim.proj_opt.zero_grad()
        optim.distill_opt.zero_grad()
        
        if self.loss_scaler:
            self.loss_scaler(
                loss_term, optim, 
                clip_grad=self.cfg.loss.clip_grad, 
                parameters=self.net.parameters(),
                create_graph=self.cfg.loss.create_graph
            )
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            if (self.iter + 1) % 2 == 0:
                optim.proj_opt.step()
                optim.distill_opt.step()

    def optimize_parameters(self):
        """Optimization step."""
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        
        with self.amp_autocast():
            self.forward()
            
            # Compute cosine loss + projection loss
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            
            # Add projection loss (RD++ specific)
            if self.L_proj is not None and isinstance(self.L_proj, torch.Tensor):
                loss_total = loss_cos + 0.2 * self.L_proj
            else:
                loss_total = loss_cos
        
        self.backward_term(loss_total, self.optim)
        
        # Update logs
        update_log_term(
            self.log_terms.get('cos'), 
            reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 
            1, self.master
        )
        
        if self.L_proj is not None and isinstance(self.L_proj, torch.Tensor):
            update_log_term(
                self.log_terms.get('proj'), 
                reduce_tensor(self.L_proj, self.world_size).clone().detach().item(), 
                1, self.master
            )

    def train_epoch(self):
        """Train one epoch with memory bank building if needed."""
        # Build memory bank before first epoch if not already built
        if not self.memory_bank_built and self.noise_enabled:
            self.build_memory_bank()
        
        # Call parent's train_epoch
        super().train_epoch()

    @torch.no_grad()
    def test(self):
        """Test the model."""
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
            
            # Forward without noise during testing
            self.feats_t, self.feats_s, _ = self.net(self.imgs, apply_noise=False)
            
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            update_log_term(
                self.log_terms.get('cos'), 
                reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 
                1, self.master
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
            
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(
                        self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), 
                        self.master, None
                    )
                    log_msg(self.logger, msg)
        
        # Merge results
        if self.cfg.dist:
            results = dict(
                imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, 
                cls_names=cls_names, anomalys=anomalys
            )
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
            results = dict(
                imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, 
                cls_names=cls_names, anomalys=anomalys
            )
        
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
            
            msg = tabulate.tabulate(
                msg, headers='keys', tablefmt="pipe", 
                floatfmt='.3f', numalign="center", stralign="center"
            )
            log_msg(self.logger, f'\n{msg}')
