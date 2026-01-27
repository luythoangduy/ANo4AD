from .patchcore_noising_256_100e import cfg as cfg_base


class cfg(cfg_base):

	def __init__(self):
		cfg_base.__init__(self)

		# Extended training
		self.epoch_full = 300
		self.test_start_epoch = 30
		self.test_per_epoch = 30
		self.warmup_epochs = 10

		# Update scheduler
		self.trainer.epoch_full = self.epoch_full
		self.trainer.test_start_epoch = self.test_start_epoch
		self.trainer.test_per_epoch = self.test_per_epoch
		self.trainer.scheduler_kwargs['warmup_epochs'] = self.warmup_epochs
		self.trainer.scheduler_kwargs['cooldown_epochs'] = 30
		self.trainer.scheduler_kwargs['decay_epochs'] = int(self.epoch_full * 0.8)

		# Update wandb name for 300 epochs
		self.wandb.name = f'patchcore_rd_{self.size}_{self.epoch_full}e'
