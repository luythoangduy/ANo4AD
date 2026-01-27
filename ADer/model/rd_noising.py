"""
RD with Adaptive Noising - Two-Phase Anomaly Detection Framework.

Phase 1: Build memory bank from teacher features (like PatchCore)
Phase 2: Train decoder with adaptive noise based on influence from memory bank

This combines the strengths of:
- RD (Reverse Distillation): Teacher-Student knowledge distillation
- PatchCore: Memory bank for normal feature representation
- Adaptive Noising: Gradient-based influence analysis for noise injection
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL

LOGGER = logging.getLogger(__name__)


# ========== Decoder Blocks from RD ==========
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class DeBottleneck(nn.Module):
    """Decoder bottleneck block for reverse distillation."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(DeBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out


class DecoderResNet(nn.Module):
    """Decoder network that mirrors the encoder structure."""

    def __init__(self, block, layers, width_per_group=64, norm_layer=None):
        super(DecoderResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, 1,
                           self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_a = self.layer1(x)   # 512*8*8 -> 256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16 -> 128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32 -> 64*64*64
        return [feature_c, feature_b, feature_a]


# ========== MFF & OCE from RD ==========
class MFF_OCE(nn.Module):
    """Multi-scale Feature Fusion and One-Class Embedding."""

    def __init__(self, block, layers, width_per_group=64, norm_layer=None):
        super(MFF_OCE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes * 3, planes, stride, downsample,
                           base_width=self.base_width, dilation=previous_dilation,
                           norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)
        output = self.bn_layer(feature)
        return output.contiguous()


# ========== Memory-Efficient Coreset Sampling ==========
def random_sampling(features, n_samples):
    """Fast random sampling."""
    N = features.shape[0]
    indices = torch.randperm(N)[:n_samples]
    return features[indices]


def greedy_coreset_sampling_batched(features, n_coreset, batch_size=5000):
    """
    Memory-efficient greedy k-Center coreset sampling using batched computation.
    
    Args:
        features: [N, D] tensor on GPU
        n_coreset: number of samples to select
        batch_size: batch size for distance computation
    
    Returns:
        selected_features: [n_coreset, D]
    """
    N, D = features.shape
    device = features.device
    
    # Start with random center
    selected_indices = [torch.randint(0, N, (1,)).item()]
    min_distances = torch.full((N,), float('inf'), device=device)
    
    for i in range(n_coreset - 1):
        # Get latest selected feature
        last_selected = features[selected_indices[-1]].unsqueeze(0)  # [1, D]
        
        # Compute distances in batches to save memory
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_features = features[start:end]  # [batch, D]
            
            # Compute L2 distance
            distances = torch.norm(batch_features - last_selected, dim=1)  # [batch]
            
            # Update minimum distances
            min_distances[start:end] = torch.minimum(min_distances[start:end], distances)
        
        # Select feature with maximum distance to nearest center
        next_idx = torch.argmax(min_distances).item()
        selected_indices.append(next_idx)
        
        if (i + 1) % 200 == 0:
            LOGGER.info(f'Coreset sampling progress: {i+1}/{n_coreset}')
    
    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
    return features[selected_indices]


def memory_efficient_coreset_sampling(
    features, 
    coreset_sampling_ratio=0.01, 
    max_features_for_greedy=100000,
    sampling_method='auto',
    device='cuda'
):
    """
    Memory-efficient coreset sampling with automatic method selection.
    
    Args:
        features: [N, D] all features (can be on CPU or GPU)
        coreset_sampling_ratio: ratio of features to keep
        max_features_for_greedy: max features before switching to random pre-sampling
        sampling_method: 'greedy', 'random', or 'auto'
        device: device for computation
    
    Returns:
        coreset_features: [M, D] selected features
    """
    N, D = features.shape
    n_coreset = max(1, int(N * coreset_sampling_ratio))
    
    LOGGER.info(f'Coreset sampling: selecting {n_coreset}/{N} features')
    LOGGER.info(f'Method: {sampling_method}, max_features_for_greedy: {max_features_for_greedy}')
    
    # Move to device if needed
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features)
    features = features.to(device)
    
    # Auto select method based on size
    if sampling_method == 'auto':
        if N > max_features_for_greedy:
            sampling_method = 'hybrid'  # Random pre-sample then greedy
        else:
            sampling_method = 'greedy'
    
    if sampling_method == 'random':
        LOGGER.info('Using fast random sampling')
        result = random_sampling(features, n_coreset)
        
    elif sampling_method == 'greedy':
        LOGGER.info('Using greedy coreset sampling')
        result = greedy_coreset_sampling_batched(features, n_coreset)
        
    elif sampling_method == 'hybrid':
        # First random sample to reduce size, then greedy
        pre_sample_size = min(max_features_for_greedy, N)
        LOGGER.info(f'Using hybrid: random pre-sample to {pre_sample_size}, then greedy')
        
        # Random pre-sampling
        features_reduced = random_sampling(features, pre_sample_size)
        
        # Greedy on reduced set
        n_coreset_adjusted = min(n_coreset, pre_sample_size)
        result = greedy_coreset_sampling_batched(features_reduced, n_coreset_adjusted)
    else:
        raise ValueError(f'Unknown sampling method: {sampling_method}')
    
    LOGGER.info(f'Coreset sampling complete: {result.shape[0]} features selected')
    return result


# ========== Adaptive Noising Module ==========
class AdaptiveNoisingModule(nn.Module):
    """
    Adaptive Noising based on Gradient-based Feature Influence Analysis.
    
    Uses memory bank to compute influence scores and generate adaptive noise
    for knowledge distillation training.
    """

    def __init__(self, feature_dim=2048, n_neighbors=9, noise_std_range=(0.01, 0.5)):
        super(AdaptiveNoisingModule, self).__init__()
        self.feature_dim = feature_dim
        self.n_neighbors = n_neighbors
        self.noise_std_range = noise_std_range

        # Learnable weights for influence scoring
        self.influence_weight = nn.Parameter(torch.ones(feature_dim))
        self.distance_weight = nn.Parameter(torch.ones(1))

    def compute_spatial_distance(self, features, memory_bank):
        """
        Compute spatial distance between features and memory bank.

        Args:
            features: [B*H*W, D] flattened features
            memory_bank: [M, D] normal features from training

        Returns:
            distances: [B*H*W, K] distances to K nearest neighbors
            indices: [B*H*W, K] indices of K nearest neighbors
        """
        # Compute pairwise distances efficiently
        distances = torch.cdist(features, memory_bank)  # [B*H*W, M]

        # Get K nearest neighbors
        topk_distances, topk_indices = torch.topk(
            distances, k=min(self.n_neighbors, memory_bank.shape[0]),
            dim=1, largest=False
        )

        return topk_distances, topk_indices

    def compute_feature_influence(self, features, memory_bank):
        """
        Compute feature influence using distance-based analysis.
        
        Measures how each feature dimension contributes to the distance
        from normal patterns in memory bank.

        Args:
            features: [B*H*W, D] features
            memory_bank: [M, D] memory bank

        Returns:
            influence_scores: [B*H*W, D] influence score per dimension
        """
        # Get nearest neighbors
        distances, indices = self.compute_spatial_distance(features, memory_bank)
        
        # Get nearest neighbor features
        k = min(self.n_neighbors, memory_bank.shape[0])
        nn_features = memory_bank[indices[:, :k]]  # [B*H*W, K, D]
        
        # Compute per-dimension difference to nearest neighbors
        features_expanded = features.unsqueeze(1).expand(-1, k, -1)  # [B*H*W, K, D]
        dim_diff = (features_expanded - nn_features).abs()  # [B*H*W, K, D]
        
        # Average across neighbors
        influence_scores = dim_diff.mean(dim=1)  # [B*H*W, D]
        
        # Apply learnable weights
        influence_scores = influence_scores * self.influence_weight.unsqueeze(0)
        
        return influence_scores, distances

    def propose_adaptive_noise_std(self, influence_scores, distances):
        """
        Propose adaptive noise std based on influence scores and distances.

        Strategy:
        - High influence features → propose larger noise
        - Features far from normal clusters → propose larger noise

        Args:
            influence_scores: [B*H*W, D] influence per dimension
            distances: [B*H*W, K] distances to neighbors

        Returns:
            proposed_noise_std: [B*H*W, D] proposed noise std per dimension
        """
        # Normalize influence scores
        influence_norm = (influence_scores - influence_scores.mean(dim=-1, keepdim=True))
        influence_norm = influence_norm / (influence_scores.std(dim=-1, keepdim=True) + 1e-8)

        # Distance signal: avg distance to neighbors
        distance_signal = distances.mean(dim=-1, keepdim=True)  # [B*H*W, 1]
        distance_signal = distance_signal.expand_as(influence_norm)  # [B*H*W, D]

        # Normalize distance signal
        distance_norm = (distance_signal - distance_signal.mean()) / (distance_signal.std() + 1e-8)

        # Combine signals
        combined_score = influence_norm + self.distance_weight * distance_norm

        # Map to noise std range using sigmoid
        min_std, max_std = self.noise_std_range
        proposed_noise_std = min_std + (max_std - min_std) * torch.sigmoid(combined_score)

        return proposed_noise_std

    def apply_adaptive_noise(self, features, memory_bank):
        """
        Compute influence and apply adaptive noise to features.

        Args:
            features: [B, C, H, W] feature maps
            memory_bank: [M, D] memory bank

        Returns:
            noised_features: [B, C, H, W] features with adaptive noise
            influence_map: [B, H, W] spatial influence map
            noise_std_map: [B, H, W] spatial noise std map
        """
        B, C, H, W = features.shape
        
        # Reshape for computation
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
        
        # Compute influence
        influence_scores, distances = self.compute_feature_influence(
            features_flat, memory_bank
        )
        
        # Get adaptive noise std
        noise_std = self.propose_adaptive_noise_std(influence_scores, distances)
        
        # Generate and apply noise
        noise = torch.randn_like(features_flat) * noise_std
        noised_features_flat = features_flat + noise
        
        # Reshape back
        noised_features = noised_features_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Create spatial maps for visualization
        influence_map = influence_scores.mean(dim=-1).reshape(B, H, W)
        noise_std_map = noise_std.mean(dim=-1).reshape(B, H, W)
        
        return noised_features, influence_map, noise_std_map


# ========== Main RD Noising Model ==========
class RD_NOISING(nn.Module):
    """
    RD with Adaptive Noising - Two-Phase Framework.
    
    Phase 1: Build memory bank from teacher features
    Phase 2: Train decoder with adaptive noise injection
    
    The model uses teacher features to build a memory bank of normal patterns,
    then uses this memory bank to compute influence scores during training.
    Adaptive noise is injected based on these influence scores.
    """

    def __init__(
        self,
        model_t,
        model_s,
        n_neighbors=9,
        noise_std_range=(0.01, 0.5),
        coreset_sampling_ratio=0.01,
        noise_layers=[0, 1, 2],  # Which layers to apply noise: 0=layer1, 1=layer2, 2=layer3
        enable_noise=True,
    ):
        super(RD_NOISING, self).__init__()
        
        # Teacher and student networks
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = get_model(model_s)
        
        # Memory bank configuration
        self.memory_banks = {}  # Dictionary for multi-scale memory banks
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.memory_bank_built = False
        
        # Noise configuration
        self.noise_layers = noise_layers
        self.enable_noise = enable_noise
        self.n_neighbors = n_neighbors
        self.noise_std_range = noise_std_range
        
        # Adaptive noising modules for different feature scales
        # For wide_resnet50_2 with out_indices=[1, 2, 3]:
        # - layer1 (index=1): 256 channels  (64x64 spatial)
        # - layer2 (index=2): 512 channels  (32x32 spatial)
        # - layer3 (index=3): 1024 channels (16x16 spatial)
        self.noising_modules = nn.ModuleDict({
            'layer1': AdaptiveNoisingModule(256, n_neighbors, noise_std_range),
            'layer2': AdaptiveNoisingModule(512, n_neighbors, noise_std_range),
            'layer3': AdaptiveNoisingModule(1024, n_neighbors, noise_std_range),
        })
        
        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        """Freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Set training mode, keeping teacher frozen."""
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def build_memory_bank(self, train_loader, device='cuda', 
                          sampling_method='auto', max_features_for_greedy=100000):
        """
        Phase 1: Build memory banks from teacher features.
        
        Extracts features from all training images and builds multi-scale
        memory banks using efficient coreset sampling.
        
        Args:
            train_loader: DataLoader for training data
            device: Device for computation
            sampling_method: 'greedy', 'random', 'hybrid', or 'auto'
            max_features_for_greedy: max features before using hybrid sampling
        """
        LOGGER.info('='*50)
        LOGGER.info('Phase 1: Building Memory Banks from Teacher Features')
        LOGGER.info('='*50)
        
        self.net_t.eval()
        
        # Collect features at each scale
        all_features = {'layer1': [], 'layer2': [], 'layer3': []}
        
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader):
                images = data['img'].to(device)
                
                # Extract teacher features
                feats_t = self.net_t(images)
                
                # feats_t is list of [B, C, H, W] for each layer
                # Layer mapping: feats_t[0]=layer1 (64x64), feats_t[1]=layer2 (32x32), feats_t[2]=layer3 (16x16)
                layer_names = ['layer1', 'layer2', 'layer3']
                
                for i, (feat, name) in enumerate(zip(feats_t, layer_names)):
                    B, C, H, W = feat.shape
                    # Flatten spatial dimensions and keep on GPU
                    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
                    all_features[name].append(feat_flat)  # Keep on GPU
                
                if (batch_idx + 1) % 10 == 0:
                    LOGGER.info(f'Processed {batch_idx + 1}/{len(train_loader)} batches')
        
        # Build memory bank for each scale
        for name in all_features:
            features = torch.cat(all_features[name], dim=0)
            LOGGER.info(f'{name}: Total features collected: {features.shape[0]}')
            
            # Apply memory-efficient coreset sampling
            if self.coreset_sampling_ratio < 1.0:
                features = memory_efficient_coreset_sampling(
                    features, 
                    coreset_sampling_ratio=self.coreset_sampling_ratio,
                    max_features_for_greedy=max_features_for_greedy,
                    sampling_method=sampling_method,
                    device=device
                )
                LOGGER.info(f'{name}: After coreset sampling: {features.shape[0]}')
            
            self.memory_banks[name] = features.to(device)
            LOGGER.info(f'{name}: Memory bank shape: {self.memory_banks[name].shape}')
            
            # Clear intermediate tensors
            del all_features[name]
            torch.cuda.empty_cache() if 'cuda' in str(device) else None
        
        self.memory_bank_built = True
        LOGGER.info('='*50)
        LOGGER.info('Memory Bank Construction Complete!')
        LOGGER.info('='*50)

    def forward(self, imgs, apply_noise=None):
        """
        Forward pass with optional adaptive noise injection.
        
        Args:
            imgs: [B, C, H, W] input images
            apply_noise: Override for noise application (None uses self.enable_noise)
        
        Returns:
            feats_t: List of teacher features
            feats_s: List of student (decoder) features
            noise_info: Dict with influence and noise maps (if noise applied)
        """
        # Extract teacher features
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        
        # Determine if we should apply noise
        should_apply_noise = apply_noise if apply_noise is not None else self.enable_noise
        should_apply_noise = should_apply_noise and self.training and self.memory_bank_built
        
        noise_info = {'influence_maps': [], 'noise_std_maps': []}
        
        if should_apply_noise:
            # Apply adaptive noise to selected layers
            noised_feats = []
            layer_names = ['layer1', 'layer2', 'layer3']
            
            for i, (feat, name) in enumerate(zip(feats_t, layer_names)):
                if i in self.noise_layers and name in self.memory_banks:
                    noised_feat, influence_map, noise_std_map = self.noising_modules[name].apply_adaptive_noise(
                        feat, self.memory_banks[name]
                    )
                    noised_feats.append(noised_feat)
                    noise_info['influence_maps'].append(influence_map)
                    noise_info['noise_std_maps'].append(noise_std_map)
                else:
                    noised_feats.append(feat)
            
            # Pass through MFF-OCE and decoder
            feats_s = self.net_s(self.mff_oce(noised_feats))
        else:
            # Standard forward without noise
            feats_s = self.net_s(self.mff_oce(feats_t))
        
        return feats_t, feats_s, noise_info

    def compute_anomaly_map(self, feats_t, feats_s, img_size, gaussian_sigma=4):
        """
        Compute anomaly map from teacher-student feature differences.
        
        Args:
            feats_t: List of teacher features
            feats_s: List of student features  
            img_size: (H, W) target size for anomaly map
            gaussian_sigma: Sigma for Gaussian smoothing
        
        Returns:
            anomaly_map: [B, H, W] pixel-level anomaly scores
            anomaly_score: [B] image-level anomaly scores
        """
        anomaly_maps = []
        
        for feat_t, feat_s in zip(feats_t, feats_s):
            # Compute cosine distance
            feat_t_norm = F.normalize(feat_t, p=2, dim=1)
            feat_s_norm = F.normalize(feat_s, p=2, dim=1)
            
            # Cosine similarity -> distance
            cos_sim = (feat_t_norm * feat_s_norm).sum(dim=1)  # [B, H, W]
            cos_dist = 1 - cos_sim
            
            # Upsample to target size
            cos_dist_up = F.interpolate(
                cos_dist.unsqueeze(1),
                size=img_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            
            anomaly_maps.append(cos_dist_up)
        
        # Combine multi-scale anomaly maps
        anomaly_map = torch.stack(anomaly_maps, dim=0).sum(dim=0)  # [B, H, W]
        
        # Apply Gaussian smoothing
        B = anomaly_map.shape[0]
        anomaly_map_np = anomaly_map.cpu().numpy()
        for i in range(B):
            anomaly_map_np[i] = gaussian_filter(anomaly_map_np[i], sigma=gaussian_sigma)
        
        # Image-level score
        anomaly_score = anomaly_map_np.max(axis=(1, 2))
        
        return anomaly_map_np, anomaly_score


@MODEL.register_module
def rd_noising(pretrained=False, **kwargs):
    """Factory function to create RD Noising model."""
    model = RD_NOISING(**kwargs)
    return model


# ========== Register Decoder ==========
@MODEL.register_module
def de_wide_resnet50_2_noising(pretrained=False, progress=True, **kwargs):
    """Decoder for wide_resnet50_2."""
    kwargs['width_per_group'] = 64 * 2
    model = DecoderResNet(DeBottleneck, [3, 4, 6], **kwargs)
    return model


if __name__ == '__main__':
    """Test the model."""
    from argparse import Namespace as _Namespace
    
    # Model configuration
    model_t = _Namespace()
    model_t.name = 'timm_wide_resnet50_2'
    model_t.kwargs = dict(
        pretrained=False,
        checkpoint_path='model/pretrain/wide_resnet50_racm-8234f177.pth',
        strict=False,
        features_only=True,
        out_indices=[1, 2, 3]
    )
    
    model_s = _Namespace()
    model_s.name = 'de_wide_resnet50_2'
    model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
    
    # Create model
    net = RD_NOISING(
        model_t=model_t,
        model_s=model_s,
        n_neighbors=9,
        noise_std_range=(0.01, 0.5),
        coreset_sampling_ratio=0.1,
    ).cuda()
    
    # Test forward
    x = torch.randn(2, 3, 256, 256).cuda()
    net.eval()
    
    feats_t, feats_s, noise_info = net(x)
    print(f'Teacher features: {[f.shape for f in feats_t]}')
    print(f'Student features: {[f.shape for f in feats_s]}')
