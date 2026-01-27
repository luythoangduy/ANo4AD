from argparse import Namespace


class cfg_model_rd_noising(Namespace):
    """
    Configuration for RD with Adaptive Noising model.
    
    Two-phase approach:
    Phase 1: Build memory bank from teacher features
    Phase 2: Train decoder with adaptive noise based on memory bank influence
    """

    def __init__(self):
        Namespace.__init__(self)
        
        # Teacher network configuration
        self.model_t = Namespace()
        self.model_t.name = 'timm_wide_resnet50_2'
        self.model_t.kwargs = dict(
            pretrained=False,
            checkpoint_path='model/pretrain/wide_resnet50_racm-8234f177.pth',
            strict=False,
            features_only=True,
            out_indices=[1, 2, 3]  # layer1, layer2, layer3
        )
        
        # Student (decoder) network configuration
        self.model_s = Namespace()
        self.model_s.name = 'de_wide_resnet50_2'
        self.model_s.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=False
        )
        
        # Main model configuration
        self.model = Namespace()
        self.model.name = 'rd_noising'
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=True,
            model_t=self.model_t,
            model_s=self.model_s,
            # Memory bank configuration
            coreset_sampling_ratio=0.01,  # 1% of features for memory bank
            # Adaptive noise configuration
            n_neighbors=9,  # K nearest neighbors for influence computation
            noise_std_range=(0.01, 0.3),  # Range of noise standard deviation
            noise_layers=[0, 1, 2],  # Apply noise to all layers
            enable_noise=True,  # Enable adaptive noise during training
        )
