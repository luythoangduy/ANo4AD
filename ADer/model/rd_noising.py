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
    indices = torch.randperm(N, device=features.device)[:n_samples]
    return features[indices]


def greedy_coreset_sampling_gpu(features, n_coreset, batch_size=5000):
    """
    Fast greedy k-Center coreset sampling on GPU.
    
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
    selected_indices = [torch.randint(0, N, (1,), device=device).item()]
    min_distances = torch.full((N,), float('inf'), device=device, dtype=features.dtype)
    
    for i in range(n_coreset - 1):
        # Get latest selected feature
        last_selected = features[selected_indices[-1]].unsqueeze(0)  # [1, D]
        
        # Compute distances in batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_features = features[start:end]
            distances = torch.norm(batch_features - last_selected, dim=1)
            min_distances[start:end] = torch.minimum(min_distances[start:end], distances)
        
        next_idx = torch.argmax(min_distances).item()
        selected_indices.append(next_idx)
        
        if (i + 1) % 500 == 0:
            LOGGER.info(f'Coreset sampling progress: {i+1}/{n_coreset}')
    
    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
    return features[selected_indices]


def greedy_coreset_sampling_cpu(features, n_coreset, batch_size=10000):
    """
    Memory-efficient greedy k-Center coreset sampling on CPU.
    Slower but handles large feature sets without OOM.
    
    Args:
        features: [N, D] tensor (will be moved to CPU)
        n_coreset: number of samples to select
        batch_size: batch size for distance computation
    
    Returns:
        selected_features: [n_coreset, D] on CPU
    """
    # Ensure on CPU
    if features.is_cuda:
        features = features.cpu()
    
    N, D = features.shape
    
    # Start with random center
    selected_indices = [torch.randint(0, N, (1,)).item()]
    min_distances = torch.full((N,), float('inf'), dtype=features.dtype)
    
    for i in range(n_coreset - 1):
        last_selected = features[selected_indices[-1]].unsqueeze(0)
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_features = features[start:end]
            distances = torch.norm(batch_features - last_selected, dim=1)
            min_distances[start:end] = torch.minimum(min_distances[start:end], distances)
        
        next_idx = torch.argmax(min_distances).item()
        selected_indices.append(next_idx)
        
        if (i + 1) % 500 == 0:
            LOGGER.info(f'Coreset sampling progress: {i+1}/{n_coreset}')
    
    selected_indices = torch.tensor(selected_indices, dtype=torch.long)
    return features[selected_indices]


def memory_efficient_coreset_sampling(
    features, 
    coreset_sampling_ratio=0.01, 
    max_features_for_greedy=100000,
    sampling_method='auto',
    device='cuda',
    coreset_device='auto'  # 'cpu', 'cuda', or 'auto'
):
    """
    Memory-efficient coreset sampling with automatic method and device selection.
    
    Args:
        features: [N, D] all features
        coreset_sampling_ratio: ratio of features to keep
        max_features_for_greedy: max features before switching to hybrid
        sampling_method: 'greedy', 'random', 'hybrid', or 'auto'
        device: target device for final output
        coreset_device: device for coreset computation ('cpu', 'cuda', 'auto')
            - 'auto': use GPU if features < 500k, else CPU
            - 'cpu': always use CPU (slower but safe)
            - 'cuda': always use GPU (faster but may OOM)
    
    Returns:
        coreset_features: [M, D] selected features on target device
    """
    N, D = features.shape
    n_coreset = max(1, int(N * coreset_sampling_ratio))
    
    LOGGER.info(f'Coreset sampling: selecting {n_coreset}/{N} features')
    LOGGER.info(f'Sampling method: {sampling_method}, coreset_device: {coreset_device}')
    
    # Convert to tensor if needed
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features)
    
    # Auto-select device for coreset computation
    if coreset_device == 'auto':
        # Use GPU if features are small enough (< 500k)
        # Rough estimate: 500k features * 1024 dim * 4 bytes = 2GB
        if N < 500000 and torch.cuda.is_available():
            coreset_device = 'cuda'
            LOGGER.info(f'Auto-selected GPU for coreset (N={N} < 500k)')
        else:
            coreset_device = 'cpu'
            LOGGER.info(f'Auto-selected CPU for coreset (N={N} >= 500k or no GPU)')
    
    # Move features to coreset computation device
    features = features.to(coreset_device)
    
    # Auto select sampling method
    if sampling_method == 'auto':
        if N > max_features_for_greedy:
            sampling_method = 'hybrid'
        else:
            sampling_method = 'greedy'
    
    # Select sampling function based on device
    if coreset_device == 'cuda':
        greedy_fn = greedy_coreset_sampling_gpu
    else:
        greedy_fn = greedy_coreset_sampling_cpu
    
    if sampling_method == 'random':
        LOGGER.info(f'Using fast random sampling on {coreset_device}')
        result = random_sampling(features, n_coreset)
        
    elif sampling_method == 'greedy':
        LOGGER.info(f'Using greedy coreset sampling on {coreset_device}')
        result = greedy_fn(features, n_coreset)
        
    elif sampling_method == 'hybrid':
        pre_sample_size = min(max_features_for_greedy, N)
        LOGGER.info(f'Using hybrid: random to {pre_sample_size}, then greedy on {coreset_device}')
        
        features_reduced = random_sampling(features, pre_sample_size)
        n_coreset_adjusted = min(n_coreset, pre_sample_size)
        result = greedy_fn(features_reduced, n_coreset_adjusted)
    else:
        raise ValueError(f'Unknown sampling method: {sampling_method}')
    
    LOGGER.info(f'Coreset sampling complete: {result.shape[0]} features selected')
    
    # Move result to target device
    return result.to(device)


# ========== Adaptive Noising Module (Analytical Gradient) ==========
class AdaptiveNoisingModule(nn.Module):
    """
    Adaptive Noising using Analytical Gradient Feature Influence.
    
    Mathematical formula (no backward needed!):
        m* = argmin_{m ∈ M} ||x - m||  (Nearest Neighbor)
        L(x) = ||x - m*||              (Distance to NN)
        ∇_x L = (x - m*) / ||x - m*||  (Unit vector from m* to x)
        
    Influence_i = |∂L/∂x_i| = |x_i - m*_i| / ||x - m*||
    
    This is 10x faster than backward() because no computational graph needed!
    Memory efficient: uses batched distance computation to avoid OOM.
    """

    def __init__(self, feature_dim=2048, n_neighbors=None, noise_std_range=(0.01, 0.5),
                 distance_batch_size=1000):
        super(AdaptiveNoisingModule, self).__init__()
        self.feature_dim = feature_dim
        self.n_neighbors = n_neighbors  # None means use all neighbors (max)
        self.noise_std_range = noise_std_range
        self.distance_batch_size = distance_batch_size  # Batch size for distance computation

        # Learnable weights for combining influence and distance signals
        self.influence_scale = nn.Parameter(torch.ones(1))
        self.distance_scale = nn.Parameter(torch.ones(1))

    def compute_knn_distance_batched(self, features, memory_bank):
        """
        Compute K-nearest neighbor distances using BATCHED computation.
        Avoids OOM by not computing full distance matrix at once.
        
        Args:
            features: [N, D] query features
            memory_bank: [M, D] memory bank
            
        Returns:
            knn_distances: [N, K] distances to K nearest neighbors
            knn_indices: [N, K] indices of K nearest neighbors
            nearest_neighbors: [N, D] features of nearest neighbor (m*)
        """
        N, D = features.shape
        M = memory_bank.shape[0]
        K = M if self.n_neighbors is None else min(self.n_neighbors, M)
        device = features.device
        
        # Initialize output tensors
        knn_distances = torch.zeros(N, K, device=device, dtype=features.dtype)
        knn_indices = torch.zeros(N, K, device=device, dtype=torch.long)
        
        # Process features in batches
        batch_size = self.distance_batch_size
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_features = features[start:end]  # [batch, D]
            
            # Compute distances for this batch to all memory bank
            # Using squared L2 for efficiency, then sqrt at the end
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            batch_norm_sq = (batch_features ** 2).sum(dim=1, keepdim=True)  # [batch, 1]
            mem_norm_sq = (memory_bank ** 2).sum(dim=1, keepdim=True).T  # [1, M]
            cross_term = batch_features @ memory_bank.T  # [batch, M]
            
            dist_sq = batch_norm_sq + mem_norm_sq - 2 * cross_term  # [batch, M]
            dist_sq = torch.clamp(dist_sq, min=0)  # Numerical stability
            distances = torch.sqrt(dist_sq + 1e-8)  # [batch, M]
            
            # Get K nearest neighbors
            topk_dist, topk_idx = torch.topk(distances, k=K, dim=1, largest=False)
            
            knn_distances[start:end] = topk_dist
            knn_indices[start:end] = topk_idx
        
        # Get nearest neighbor features (m*) - index 0 is the closest
        nearest_neighbors = memory_bank[knn_indices[:, 0]]  # [N, D]
        
        return knn_distances, knn_indices, nearest_neighbors

    def compute_influence_analytical(self, features, memory_bank):
        """
        Compute feature influence using ANALYTICAL GRADIENT formula.
        
        Mathematical derivation:
            L(x) = ||x - m*||_2 = sqrt(Σ(x_i - m*_i)²)
            ∂L/∂x_i = (x_i - m*_i) / ||x - m*||_2
            
        Influence = |∂L/∂x_i| = |x_i - m*_i| / ||x - m*||_2
        
        This is the Normalized Residual - shows which dimensions
        contribute most to the anomaly.
        
        10x FASTER than backward() - no computational graph needed!
        
        Args:
            features: [N, D] query features (x)
            memory_bank: [M, D] memory bank (M)
            
        Returns:
            influence: [N, D] influence score per dimension
            knn_distances: [N, K] distances to K nearest neighbors
        """
        with torch.no_grad():
            # Get KNN and nearest neighbor
            knn_distances, knn_indices, nearest_neighbors = self.compute_knn_distance_batched(
                features, memory_bank
            )
            
            # Residual: x - m* (numerator of gradient)
            diff = features - nearest_neighbors  # [N, D]
            
            # Distance to nearest neighbor: ||x - m*|| (denominator)
            # This is knn_distances[:, 0] but we compute it fresh for numerical stability
            norm = knn_distances[:, 0:1] + 1e-8  # [N, 1]
            
            # Analytical gradient: (x - m*) / ||x - m*||
            gradient = diff / norm  # [N, D]
            
            # Influence is absolute value of gradient
            influence = gradient.abs()  # [N, D]
        
        return influence, knn_distances

    def compute_influence_fast(self, features, memory_bank):
        """
        Fast influence computation - same as analytical since both are O(N*M/batch).
        Kept for API compatibility.
        
        Args:
            features: [N, D] features
            memory_bank: [M, D] memory bank
            
        Returns:
            influence: [N, D] influence approximation
            knn_distances: [N, K] distances to neighbors
        """
        return self.compute_influence_analytical(features, memory_bank)

    def propose_adaptive_noise_std(self, influence, knn_distances):
        """
        Propose adaptive noise std based on influence and distances.
        
        Strategy:
        - High influence dims → larger noise (more sensitive to change)
        - Far from normal → larger noise (more anomalous)
        
        Args:
            influence: [N, D] influence per dimension
            knn_distances: [N, K] distances to K neighbors
            
        Returns:
            noise_std: [N, D] proposed noise std per dimension
        """
        # Normalize influence to [0, 1] range per sample
        influence_min = influence.min(dim=-1, keepdim=True)[0]
        influence_max = influence.max(dim=-1, keepdim=True)[0]
        influence_norm = (influence - influence_min) / (influence_max - influence_min + 1e-8)
        
        # Distance signal: mean distance to neighbors, expanded to all dims
        distance_signal = knn_distances.mean(dim=-1, keepdim=True)  # [N, 1]
        
        # Normalize distance signal
        dist_min, dist_max = distance_signal.min(), distance_signal.max()
        distance_norm = (distance_signal - dist_min) / (dist_max - dist_min + 1e-8)
        distance_norm = distance_norm.expand_as(influence_norm)  # [N, D]
        
        # Combine signals with learnable scaling
        combined = self.influence_scale * influence_norm + self.distance_scale * distance_norm
        
        # Map to noise std range using sigmoid
        min_std, max_std = self.noise_std_range
        noise_std = min_std + (max_std - min_std) * torch.sigmoid(combined - 0.5)
        
        return noise_std

    def forward(self, features, memory_bank, use_gradient=True):
        """
        Compute influence and propose adaptive noise.
        
        Args:
            features: [N, D] features
            memory_bank: [M, D] memory bank
            use_gradient: ignored (kept for API compatibility)
                         Always uses analytical formula now.
            
        Returns:
            influence: [N, D] influence scores
            noise_std: [N, D] proposed noise std
            knn_distances: [N, K] distances to neighbors
        """
        # Always use analytical formula (fastest!)
        influence, knn_distances = self.compute_influence_analytical(features, memory_bank)
        
        noise_std = self.propose_adaptive_noise_std(influence, knn_distances)
        
        return influence, noise_std, knn_distances

    def apply_adaptive_noise(self, features, memory_bank, use_gradient=True):
        """
        Compute influence and apply adaptive noise to features.

        Args:
            features: [B, C, H, W] feature maps
            memory_bank: [M, D] memory bank
            use_gradient: use gradient-based influence

        Returns:
            noised_features: [B, C, H, W] features with adaptive noise
            influence_map: [B, H, W] spatial influence map
            noise_std_map: [B, H, W] spatial noise std map
        """
        B, C, H, W = features.shape
        
        # Reshape: [B, C, H, W] -> [B*H*W, C]
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        
        # Compute influence and noise std
        influence, noise_std, _ = self.forward(features_flat, memory_bank, use_gradient)
        
        # Apply noise
        noise = torch.randn_like(features_flat) * noise_std
        noised_features_flat = features_flat + noise
        
        # Reshape back: [B*H*W, C] -> [B, C, H, W]
        noised_features = noised_features_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # Create spatial maps for visualization
        influence_map = influence.mean(dim=-1).reshape(B, H, W)
        noise_std_map = noise_std.mean(dim=-1).reshape(B, H, W)
        
        return noised_features, influence_map.detach(), noise_std_map.detach()


# ========== Main RD Noising Model ==========
class RD_NOISING(nn.Module):
    """
    RD with Adaptive Noising - Two-Phase Framework.
    
    Phase 1: Build memory bank from MFF_OCE features (GAP)
             MFF_OCE output: [B, 2048, 8, 8] → GAP → [B, 2048] per image
    Phase 2: Train decoder with adaptive noise injection on MFF_OCE output
    
    Memory bank strategy:
    - Use GAP on MFF_OCE output → 1 vector (2048 dim) per image
    - Memory: 1 × N_images vectors (compact!)
    
    Noise injection:
    - Applied on MFF_OCE output [B, 2048, 8, 8]
    - Influence computed per spatial position [B*64, 2048] vs memory bank [M, 2048]
    - Noise varies spatially based on each position's distance to normal
    """

    def __init__(
        self,
        model_t,
        model_s,
        n_neighbors=None,  # None = use all neighbors (max)
        noise_std_range=(0.01, 0.5),
        coreset_sampling_ratio=0.1,  # Higher ratio since only 1 vector per image
        enable_noise=True,
        distance_batch_size=512,
    ):
        super(RD_NOISING, self).__init__()
        
        # Teacher and student networks
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = get_model(model_s)
        
        # Memory bank configuration - GAP of MFF_OCE output
        # MFF_OCE output: [B, 2048, 8, 8] → GAP → [B, 2048]
        self.memory_bank = None  # [M, 2048] GAP feature memory
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.memory_bank_built = False
        
        # Noise configuration
        self.enable_noise = enable_noise
        self.n_neighbors = n_neighbors
        self.noise_std_range = noise_std_range
        
        # Noising module for MFF_OCE output (2048 channels)
        # Used for spatial influence computation
        self.noising_module = AdaptiveNoisingModule(
            feature_dim=2048,  # MFF_OCE output dimension
            n_neighbors=n_neighbors, 
            noise_std_range=noise_std_range,
            distance_batch_size=distance_batch_size
        )
        
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
                          sampling_method='auto', max_features_for_greedy=100000,
                          coreset_device='auto'):
        """
        Phase 1: Build memory bank from MFF_OCE features using GAP.
        
        Strategy:
        - Extract MFF_OCE output [B, 2048, 8, 8]
        - Apply GAP → [B, 2048] per image
        - Results in 1 vector per image (compact memory bank!)
        
        Memory comparison (for 1000 training images):
        - Old MFF_OCE spatial: 8×8 × 1000 = 64K vectors × 2048 dim
        - New GAP approach: 1 × 1000 = 1K vectors × 2048 dim (64x fewer vectors!)
        
        Args:
            train_loader: DataLoader for training data
            device: Device for final memory bank storage
            sampling_method: 'greedy', 'random', 'hybrid', or 'auto'
            max_features_for_greedy: max features before using hybrid sampling
            coreset_device: Device for coreset computation ('cpu', 'cuda', 'auto')
        """
        LOGGER.info('='*50)
        LOGGER.info('Phase 1: Building Memory Bank from MFF_OCE GAP Features')
        LOGGER.info(f'Feature dim: 2048')
        LOGGER.info(f'Coreset device: {coreset_device}')
        LOGGER.info('='*50)
        
        self.net_t.eval()
        self.mff_oce.eval()
        
        # Collect GAP features from MFF_OCE output
        all_features = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader):
                images = data['img'].to(device)
                
                # Extract teacher features and fuse with MFF_OCE
                feats_t = self.net_t(images)
                fused_feat = self.mff_oce(feats_t)  # [B, 2048, 8, 8]
                
                # Apply GAP: [B, 2048, 8, 8] -> [B, 2048]
                gap_feat = F.adaptive_avg_pool2d(fused_feat, 1).flatten(1).cpu()
                all_features.append(gap_feat)
                
                if (batch_idx + 1) % 10 == 0:
                    LOGGER.info(f'Processed {batch_idx + 1}/{len(train_loader)} batches')
                    
                # Clear GPU cache periodically
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Concatenate all features
        LOGGER.info('Concatenating features...')
        features = torch.cat(all_features, dim=0)  # [N_images, 2048]
        LOGGER.info(f'Total features collected: {features.shape[0]} images × {features.shape[1]} dim')
        
        # Clear the list to free memory
        all_features = []
        
        # Apply memory-efficient coreset sampling
        if self.coreset_sampling_ratio < 1.0:
            features = memory_efficient_coreset_sampling(
                features, 
                coreset_sampling_ratio=self.coreset_sampling_ratio,
                max_features_for_greedy=max_features_for_greedy,
                sampling_method=sampling_method,
                device=device,
                coreset_device=coreset_device
            )
            LOGGER.info(f'After coreset sampling: {features.shape[0]} features')
        else:
            features = features.to(device)
        
        self.memory_bank = features
        LOGGER.info(f'Memory bank shape: {self.memory_bank.shape}')
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        self.memory_bank_built = True
        LOGGER.info('='*50)
        LOGGER.info('Memory Bank Construction Complete!')
        LOGGER.info('='*50)

    def forward(self, imgs, apply_noise=None):
        """
        Forward pass with optional adaptive noise injection on MFF_OCE output.
        
        Flow:
        1. Teacher extracts multi-scale features
        2. MFF_OCE fuses features → [B, 2048, 8, 8]
        3. Compute influence per spatial position vs memory bank (GAP)
        4. Apply spatially-varying noise based on influence
        5. Decoder reconstructs multi-scale features
        
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
        
        # Fuse features with MFF_OCE
        fused_feat = self.mff_oce(feats_t)  # [B, 2048, 8, 8]
        
        # Determine if we should apply noise
        should_apply_noise = apply_noise if apply_noise is not None else self.enable_noise
        should_apply_noise = should_apply_noise and self.training and self.memory_bank_built
        
        noise_info = {'influence_map': None, 'noise_std_map': None}
        
        if should_apply_noise:
            # Apply adaptive noise with spatial variation
            # Uses noising_module.apply_adaptive_noise which computes per-position influence
            noised_fused, influence_map, noise_std_map = self.noising_module.apply_adaptive_noise(
                fused_feat, self.memory_bank
            )
            noise_info['influence_map'] = influence_map  # [B, 8, 8]
            noise_info['noise_std_map'] = noise_std_map  # [B, 8, 8]
            
            # Decode noised features
            feats_s = self.net_s(noised_fused)
        else:
            # Standard forward without noise
            feats_s = self.net_s(fused_feat)
        
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
        n_neighbors=None,  # Use all neighbors (max)
        noise_std_range=(0.01, 0.5),
        coreset_sampling_ratio=0.1,
    ).cuda()
    
    # Test forward
    x = torch.randn(2, 3, 256, 256).cuda()
    net.eval()
    
    feats_t, feats_s, noise_info = net(x)
    print(f'Teacher features: {[f.shape for f in feats_t]}')
    print(f'Student features: {[f.shape for f in feats_s]}')
