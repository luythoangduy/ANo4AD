from argparse import Namespace


class cfg_model_patchcore_noising(Namespace):

	def __init__(self):
		Namespace.__init__(self)
		# Backbone for feature extraction
		self.model_backbone = Namespace()
		self.model_backbone.name = 'timm_wide_resnet50_2'
		self.model_backbone.kwargs = dict(
			pretrained=True,
			checkpoint_path='',
			strict=False,
			features_only=True,
			out_indices=[1, 2]  # layer2, layer3
		)

		# Main model configuration (Adaptive Propose Noising)
		self.model = Namespace()
		self.model.name = 'patchcore_noising'
		self.model.kwargs = dict(
			pretrained=False,
			checkpoint_path='',
			strict=True,
			model_backbone=self.model_backbone,
			layers_to_extract_from=('layer2', 'layer3'),
			input_size=(3, 256, 256),
			pretrain_embed_dimension=1024,
			target_embed_dimension=1024,
			n_neighbors=9,
			influence_ratio=0.1,
			noise_std_range=(0.01, 0.5),
			coreset_sampling_ratio=0.01,
		)
