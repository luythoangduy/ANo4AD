from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from ..__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_patchcore_noising):

	def __init__(self):
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_patchcore_noising.__init__(self)

		self.seed = 42
		self.size = 256
		self.image_size = 256
		self.input_size = (3, self.image_size, self.image_size)
		self.epoch_full = 100
		self.warmup_epochs = 5
		self.test_start_epoch = 10
		self.test_per_epoch = 10
		self.batch_train = 8
		self.batch_test_per = 8
		self.lr = 0.005 * self.batch_train / 8
		self.weight_decay = 0.0001
		self.metrics = [
			'mAUROC_sp_max','AUROC_sp', 'mAUROC_px', 'mAUPRO_px',
			'mAP_sp_max', 'mAP_px',
			'mF1_max_sp_max',
			'mF1_px_0.2_0.8_0.1', 'mAcc_px_0.2_0.8_0.1', 'mIoU_px_0.2_0.8_0.1',
			'mF1_max_px', 'mIoU_max_px',
		]
		self.fvcore_is = False
		self.fvcore_b = 2

		# ==> data
		self.data.type = 'DefaultAD'
		self.data.root = 'data/mvtec'
		self.data.meta = 'meta.json'
		self.data.cls_names = []

		self.data.anomaly_source_path = 'data/dtd/images/'
		self.data.resize_shape = [self.size, self.size]

		self.data.use_sample = False
		self.data.views = []

		self.data.train_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.image_size, self.image_size)),
			dict(type='ToTensor'),
			dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
		]
		self.data.test_transforms = self.data.train_transforms
		self.data.target_transforms = [
			dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
			dict(type='CenterCrop', size=(self.image_size, self.image_size)),
			dict(type='ToTensor'),
		]

		# ==> model (with Adaptive Propose Noising)
		self.layers_to_extract_from = ('layer2', 'layer3')
		self.model_backbone = Namespace()
		self.model_backbone.name = 'timm_wide_resnet50_2'
		self.model_backbone.kwargs = dict(
			pretrained=True,
			checkpoint_path='',
			strict=False,
			features_only=True,
			out_indices=[1, 2]  # layer2, layer3
		)

		self.model = Namespace()
		self.model.name = 'patchcore_noising'
		self.model.kwargs = dict(
			pretrained=False,
			checkpoint_path='',
			strict=True,
			model_backbone=self.model_backbone,
			layers_to_extract_from=self.layers_to_extract_from,
			input_size=self.input_size,
			pretrain_embed_dimension=1024,
			target_embed_dimension=1024,
			n_neighbors=9,
			influence_ratio=0.1,
			noise_std_range=(0.01, 0.5),
			coreset_sampling_ratio=0.01,
		)

		# ==> evaluator
		self.evaluator.kwargs = dict(
			metrics=self.metrics,
			pooling_ks=None,
			max_step_aupro=100,
			use_adeval=self.use_adeval
		)

		# ==> optimizer
		self.optim = Namespace()
		self.optim.lr = self.lr
		self.optim.kwargs = dict(name='adamw', betas=(0.9, 0.999), weight_decay=self.weight_decay)

		# ==> trainer (PatchCore-style: no training, only memory bank building)
		self.trainer.name = 'PatchCoreNoisingTrainer'
		self.trainer.logdir_sub = ''
		self.trainer.resume_dir = ''
		self.trainer.epoch_full = 1  # Only need 1 epoch to build memory bank
		self.trainer.test_start_epoch = 1
		self.trainer.test_per_epoch = 1

		self.trainer.data.batch_size = self.batch_train
		self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

		# No loss needed (unsupervised)

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]

		# ==> wandb
		self.wandb.enable = True  # Enable WandB logging
		self.wandb.project = 'patchcore-rd-anomaly'
		self.wandb.name = f'patchcore_rd_{self.size}_{self.epoch_full}e'
		self.wandb.tags = ['patchcore', 'reverse-distillation', 'anomaly-detection']
		self.wandb.notes = 'PatchCore with Reverse Distillation for Anomaly Detection'
		self.wandb.log_model = True  # Save best model to wandb
		self.wandb.log_freq = 50
