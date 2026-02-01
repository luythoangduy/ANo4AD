"""
RD++ (Revisit RD) with Adaptive Noising - Two-Phase Anomaly Detection Framework.

Combines:
- RD++ features: Multi-projection layer, SSOT loss, reconstruct loss, contrast loss
- Adaptive Noising: Memory bank based influence analysis for noise injection

Phase 1: Build memory bank from teacher features (like PatchCore)
Phase 2: Train decoder with adaptive noise + RD++ projection losses

This model extends RD++ with:
- Memory bank for normal feature representation
- Adaptive noise injection based on feature influence
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

try:
    import geomloss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False
    logging.warning("geomloss not installed. RDPP Noising will use simplified loss.")

LOGGER = logging.getLogger(__name__)


# ========== Decoder Blocks (from RD++) ==========
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


# ========== MFF & OCE ==========
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


# ========== RD++ Projection Layer ==========
class ProjLayer(nn.Module):
    """Projection layer for feature transformation."""
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_c // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_c // 4),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_c // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, out_c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(),
        )
        
    def forward(self, x):
        return self.proj(x)


class MultiProjectionLayer(nn.Module):
    """Multi-scale projection layer for RD++."""
    def __init__(self, base=64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)

    def forward(self, features, features_noise=None):
        if features_noise is not None:
            return (
                [self.proj_a(features_noise[0]), self.proj_b(features_noise[1]), self.proj_c(features_noise[2])],
                [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]
            )
        else:
            return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]


# ========== RD++ Loss Functions ==========
class CosineReconstruct(nn.Module):
    """Cosine reconstruction loss."""
    def __init__(self):
        super(CosineReconstruct, self).__init__()
        
    def forward(self, x, y):
        return torch.mean(1 - F.cosine_similarity(x.flatten(2), y.flatten(2), dim=2))


class Revisit_RDLoss(nn.Module):
    """
    RD++ multi-task loss: SSOT loss, Reconstruct Loss, Contrast Loss
    """
    def __init__(self, consistent_shuffle=True):
        super(Revisit_RDLoss, self).__init__()
        if HAS_GEOMLOSS:
            self.sinkhorn = geomloss.SamplesLoss(
                loss='sinkhorn', p=2, blur=0.05,
                reach=None, diameter=10000000, scaling=0.95,
                truncate=10, cost=None, kernel=None, cluster_scale=None,
                debias=True, potentials=False, verbose=False, backend='auto'
            )
        else:
            self.sinkhorn = None
        self.reconstruct = CosineReconstruct()
        self.contrast = nn.CosineEmbeddingLoss(margin=0.5)
        
    def forward(self, noised_feature, projected_noised_feature, projected_normal_feature):
        """
        Compute RD++ loss.
        
        Args:
            noised_feature: output of encoder at each_blocks
            projected_noised_feature: projection layer output on noised features
            projected_normal_feature: projection layer output on normal features
        """
        current_batchsize = projected_normal_feature[0].shape[0]
        target = -torch.ones(current_batchsize).to(projected_normal_feature[0].device)

        normal_proj1 = projected_normal_feature[0]
        normal_proj2 = projected_normal_feature[1]
        normal_proj3 = projected_normal_feature[2]
        
        # Shuffle samples for SSOT loss
        shuffle_index = torch.randperm(current_batchsize)
        shuffle_1 = normal_proj1[shuffle_index]
        shuffle_2 = normal_proj2[shuffle_index]
        shuffle_3 = normal_proj3[shuffle_index]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = projected_noised_feature
        noised_feature1, noised_feature2, noised_feature3 = noised_feature

        # SSOT loss (Sinkhorn)
        if self.sinkhorn is not None:
            loss_ssot = (
                self.sinkhorn(
                    torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1),
                    torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1), -1)
                ) +
                self.sinkhorn(
                    torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1), -1),
                    torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1), -1)
                ) +
                self.sinkhorn(
                    torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1), -1),
                    torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1), -1)
                )
            )
        else:
            # Fallback: use MSE as proxy
            loss_ssot = (
                F.mse_loss(normal_proj1, shuffle_1) +
                F.mse_loss(normal_proj2, shuffle_2) +
                F.mse_loss(normal_proj3, shuffle_3)
            )

        # Reconstruct loss
        loss_reconstruct = (
            self.reconstruct(abnormal_proj1, normal_proj1) +
            self.reconstruct(abnormal_proj2, normal_proj2) +
            self.reconstruct(abnormal_proj3, normal_proj3)
        )

        # Contrast loss
        loss_contrast = (
            self.contrast(
                noised_feature1.view(noised_feature1.shape[0], -1),
                normal_proj1.view(normal_proj1.shape[0], -1),
                target=target
            ) +
            self.contrast(
                noised_feature2.view(noised_feature2.shape[0], -1),
                normal_proj2.view(normal_proj2.shape[0], -1),
                target=target
            ) +
            self.contrast(
                noised_feature3.view(noised_feature3.shape[0], -1),
                normal_proj3.view(normal_proj3.shape[0], -1),
                target=target
            )
        )

        return (loss_ssot + 0.01 * loss_reconstruct + 0.1 * loss_contrast) / 1.11


# ========== Memory-Efficient Coreset Sampling ==========
def random_sampling(features, n_samples):
    """Fast random sampling."""
    N = features.shape[0]
    indices = torch.randperm(N, device=features.device)[:n_samples]
    return features[indices]


def greedy_coreset_sampling_gpu(features, n_coreset, batch_size=5000):
    """Fast greedy k-Center coreset sampling on GPU."""
    N, D = features.shape
    device = features.device
    
    selected_indices = [torch.randint(0, N, (1,), device=device).item()]
    min_distances = torch.full((N,), float('inf'), device=device, dtype=features.dtype)
    
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
    
    selected_indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
    return features[selected_indices]


def greedy_coreset_sampling_cpu(features, n_coreset, batch_size=10000):
    """Memory-efficient greedy k-Center coreset sampling on CPU."""
    if features.is_cuda:
        features = features.cpu()
    
    N, D = features.shape
    
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
    coreset_device='auto'
):
    """Memory-efficient coreset sampling with automatic method and device selection."""
    N, D = features.shape
    n_coreset = max(1, int(N * coreset_sampling_ratio))
    
    LOGGER.info(f'Coreset sampling: selecting {n_coreset}/{N} features')
    LOGGER.info(f'Sampling method: {sampling_method}, coreset_device: {coreset_device}')
    
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features)
    
    if coreset_device == 'auto':
        if N < 500000 and torch.cuda.is_available():
            coreset_device = 'cuda'
            LOGGER.info(f'Auto-selected GPU for coreset (N={N} < 500k)')
        else:
            coreset_device = 'cpu'
            LOGGER.info(f'Auto-selected CPU for coreset (N={N} >= 500k or no GPU)')
    
    features = features.to(coreset_device)
    
    if sampling_method == 'auto':
        if N > max_features_for_greedy:
            sampling_method = 'hybrid'
        else:
            sampling_method = 'greedy'
    
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
    return result.to(device)


# ========== Adaptive Noising Module ==========
class AdaptiveNoisingModule(nn.Module):
    """
    Adaptive Noising using Analytical Gradient Feature Influence.
    """

    def __init__(self, feature_dim=2048, n_neighbors=9, noise_std_range=(0.01, 0.5),
                 distance_batch_size=1000):
        super(AdaptiveNoisingModule, self).__init__()
        self.feature_dim = feature_dim
        self.n_neighbors = n_neighbors
        self.noise_std_range = noise_std_range
        self.distance_batch_size = distance_batch_size

        self.influence_scale = nn.Parameter(torch.ones(1))
        self.distance_scale = nn.Parameter(torch.ones(1))

    def compute_knn_distance_batched(self, features, memory_bank):
        """Compute K-nearest neighbor distances using batched computation."""
        N, D = features.shape
        M = memory_bank.shape[0]
        K = min(self.n_neighbors, M)
        device = features.device
        
        knn_distances = torch.zeros(N, K, device=device, dtype=features.dtype)
        knn_indices = torch.zeros(N, K, device=device, dtype=torch.long)
        
        batch_size = self.distance_batch_size
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_features = features[start:end]
            
            batch_norm_sq = (batch_features ** 2).sum(dim=1, keepdim=True)
            mem_norm_sq = (memory_bank ** 2).sum(dim=1, keepdim=True).T
            cross_term = batch_features @ memory_bank.T
            
            dist_sq = batch_norm_sq + mem_norm_sq - 2 * cross_term
            dist_sq = torch.clamp(dist_sq, min=0)
            distances = torch.sqrt(dist_sq + 1e-8)
            
            topk_dist, topk_idx = torch.topk(distances, k=K, dim=1, largest=False)
            
            knn_distances[start:end] = topk_dist
            knn_indices[start:end] = topk_idx
        
        nearest_neighbors = memory_bank[knn_indices[:, 0]]
        return knn_distances, knn_indices, nearest_neighbors

    def compute_influence_analytical(self, features, memory_bank):
        """Compute feature influence using analytical gradient formula."""
        with torch.no_grad():
            knn_distances, knn_indices, nearest_neighbors = self.compute_knn_distance_batched(
                features, memory_bank
            )
            
            diff = features - nearest_neighbors
            norm = knn_distances[:, 0:1] + 1e-8
            gradient = diff / norm
            influence = gradient.abs()
        
        return influence, knn_distances

    def apply_adaptive_noise(self, features, memory_bank):
        """
        Apply adaptive noise to features based on memory bank influence.
        
        Args:
            features: [B, C, H, W] features from MFF_OCE
            memory_bank: [M, D] memory bank features
            
        Returns:
            noised_features: [B, C, H, W]
            influence_map: [B, H, W]
            noise_std_map: [B, H, W]
        """
        B, C, H, W = features.shape
        device = features.device
        
        # Reshape to [B*H*W, C] for per-position processing
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Compute influence for each spatial position
        influence, knn_distances = self.compute_influence_analytical(features_flat, memory_bank)
        
        # Aggregate influence per position (mean across channels)
        influence_per_pos = influence.mean(dim=1)  # [B*H*W]
        
        # Normalize influence to [0, 1]
        influence_min = influence_per_pos.min()
        influence_max = influence_per_pos.max()
        if influence_max - influence_min > 1e-8:
            influence_norm = (influence_per_pos - influence_min) / (influence_max - influence_min)
        else:
            influence_norm = torch.zeros_like(influence_per_pos)
        
        # Map influence to noise std
        noise_min, noise_max = self.noise_std_range
        noise_std = noise_min + influence_norm * (noise_max - noise_min)
        
        # Generate noise
        noise = torch.randn_like(features_flat) * noise_std.unsqueeze(1)
        
        # Apply noise
        noised_features_flat = features_flat + noise
        
        # Reshape back
        noised_features = noised_features_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        influence_map = influence_norm.reshape(B, H, W)
        noise_std_map = noise_std.reshape(B, H, W)
        
        return noised_features, influence_map, noise_std_map


# ========== Main RDPP Noising Model ==========
class RDPP_NOISING(nn.Module):
    """
    RD++ with Adaptive Noising Model.
    
    Combines RD++ architecture with memory bank based adaptive noise injection.
    """

    def __init__(self, model_t, model_s, n_neighbors=9, noise_std_range=(0.01, 0.3),
                 coreset_sampling_ratio=0.01, enable_noise=True, proj_base=64, **kwargs):
        super(RDPP_NOISING, self).__init__()
        
        # Networks
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = get_model(model_s)
        
        # RD++ components
        self.proj_layer = MultiProjectionLayer(base=proj_base)
        self.proj_loss = Revisit_RDLoss()
        
        # Adaptive noising - uses last teacher layer features (1024 for wide_resnet50)
        # feature_dim should match feats_t[-1] channel count
        self.noising_module = AdaptiveNoisingModule(
            feature_dim=1024,  # Last teacher layer: 1024 channels for wide_resnet50
            n_neighbors=n_neighbors,
            noise_std_range=noise_std_range,
        )
        
        # Configuration
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.enable_noise = enable_noise
        
        # Memory bank (built during training)
        self.memory_bank = None
        self.memory_bank_built = False
        
        # Frozen layers
        self.frozen_layers = ['net_t']
        
        LOGGER.info('='*50)
        LOGGER.info('RDPP Noising Model Initialized')
        LOGGER.info(f'Noise Enabled: {enable_noise}')
        LOGGER.info(f'N Neighbors: {n_neighbors}')
        LOGGER.info(f'Noise STD Range: {noise_std_range}')
        LOGGER.info(f'Coreset Sampling Ratio: {coreset_sampling_ratio}')
        LOGGER.info(f'Projection Base: {proj_base}')
        LOGGER.info('='*50)

    def freeze_layer(self, module):
        """Freeze a module."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Set training mode with frozen layers."""
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def build_memory_bank(self, train_loader, device='cuda', sampling_method='auto',
                          max_features_for_greedy=100000, coreset_device='auto'):
        """
        Build memory bank from training data.
        
        Stores GAP features from the last layer of teacher encoder (1024 channels for wide_resnet50).
        This matches the dimension used in _apply_adaptive_noise_to_feats.
        """
        if self.memory_bank_built:
            LOGGER.info('Memory bank already built, skipping...')
            return
        
        LOGGER.info('='*50)
        LOGGER.info('Building Memory Bank for RDPP Noising...')
        LOGGER.info('='*50)
        
        self.net_t.eval()
        
        all_features = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(train_loader):
                images = data['img'].to(device)
                
                feats_t = self.net_t(images)
                
                # Use last layer features (1024 channels for wide_resnet50)
                # This matches the reference feature in _apply_adaptive_noise_to_feats
                ref_feat = feats_t[-1]  # [B, 1024, 16, 16]
                
                # GAP to get compact representation
                gap_feat = F.adaptive_avg_pool2d(ref_feat, 1).flatten(1).cpu()  # [B, 1024]
                all_features.append(gap_feat)
                
                if (batch_idx + 1) % 10 == 0:
                    LOGGER.info(f'Processed {batch_idx + 1}/{len(train_loader)} batches')
                
                if (batch_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        
        LOGGER.info('Concatenating features...')
        features = torch.cat(all_features, dim=0)
        LOGGER.info(f'Total features: {features.shape[0]} images × {features.shape[1]} dim')
        
        all_features = []
        
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
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        self.memory_bank_built = True
        LOGGER.info('='*50)
        LOGGER.info('Memory Bank Construction Complete!')
        LOGGER.info('='*50)

    def forward(self, imgs, img_noise=None, apply_noise=None):
        """
        Forward pass with RD++ architecture + Adaptive Noising.
        
        RD++ Original Flow:
        1. feats_t = net_t(imgs)
        2. inputs_noise = net_t(img_noise)  <-- External noised image
        3. (proj_noise, proj_clean) = proj_layer(feats_t, features_noise=inputs_noise)
        4. L_proj = proj_loss(inputs_noise, proj_noise, proj_clean)
        5. feats_s = net_s(mff_oce(proj_clean))
        
        RDPP_NOISING Flow (Adaptive Noise):
        1. feats_t = net_t(imgs)
        2. noised_feats_t = apply_adaptive_noise(feats_t)  <-- Memory bank based
        3. (proj_noise, proj_clean) = proj_layer(feats_t, features_noise=noised_feats_t)
        4. L_proj = proj_loss(noised_feats_t, proj_noise, proj_clean)
        5. feats_s = net_s(mff_oce(proj_clean))
        
        Args:
            imgs: [B, C, H, W] input images
            img_noise: [B, C, H, W] noised input (for original RD++ mode, optional)
            apply_noise: Override for adaptive noise application
        
        Returns:
            feats_t: List of teacher features
            feats_s: List of student features  
            L_proj: Projection loss (if training)
        """
        # 1. Extract teacher features
        feats_t = self.net_t(imgs)
        
        if self.training:
            # Determine noise mode
            should_apply_adaptive_noise = (
                apply_noise if apply_noise is not None else self.enable_noise
            ) and self.memory_bank_built
            
            if should_apply_adaptive_noise:
                # === RD++ with Adaptive Noise (Memory Bank based) ===
                
                # 2. Apply adaptive noise to teacher features
                # This replaces the external img_noise path
                noised_feats_t = self._apply_adaptive_noise_to_feats(feats_t)
                
                # 3. Projection layer: project both clean and noised features
                feature_space_noise, feature_space = self.proj_layer(
                    feats_t, features_noise=noised_feats_t
                )
                
                # 4. Compute RD++ projection loss
                L_proj = self.proj_loss(noised_feats_t, feature_space_noise, feature_space)
                
                # 5. Decode from clean projected features through MFF_OCE
                feats_s = self.net_s(self.mff_oce(feature_space))
                
                return feats_t, feats_s, L_proj
            
            elif img_noise is not None:
                # === Original RD++ mode with external noise ===
                inputs_noise = self.net_t(img_noise)
                feature_space_noise, feature_space = self.proj_layer(feats_t, features_noise=inputs_noise)
                L_proj = self.proj_loss(inputs_noise, feature_space_noise, feature_space)
                feats_s = self.net_s(self.mff_oce(feature_space))
                return feats_t, feats_s, L_proj
            
            else:
                # === No noise mode ===
                features = self.proj_layer(feats_t)
                feats_s = self.net_s(self.mff_oce(features))
                return feats_t, feats_s, torch.tensor(0.0, device=imgs.device)
        
        else:
            # === Inference mode ===
            features = self.proj_layer(feats_t)
            feats_s = self.net_s(self.mff_oce(features))
            return feats_t, feats_s, None
    
    def _apply_adaptive_noise_to_feats(self, feats_t):
        """
        Apply adaptive noise to multi-scale teacher features.
        
        Computes influence from memory bank (which stores GAP features of MFF output)
        and applies spatially-varying noise to each scale of teacher features.
        
        Args:
            feats_t: List of teacher features at different scales
                     [feat1: B,C1,H1,W1], [feat2: B,C2,H2,W2], [feat3: B,C3,H3,W3]
        
        Returns:
            noised_feats: List of noised features with same shapes as input
        """
        # First, compute influence using the largest scale feature (last one, 1024 channels)
        # or use a representative feature to get noise std map
        ref_feat = feats_t[-1]  # [B, 1024, 16, 16] for wide_resnet50
        B, C_ref, H_ref, W_ref = ref_feat.shape
        
        # Flatten for influence computation
        ref_flat = ref_feat.permute(0, 2, 3, 1).reshape(-1, C_ref)  # [B*H*W, C]
        
        # Compute influence using analytical gradient
        influence, knn_distances = self.noising_module.compute_influence_analytical(
            ref_flat, self.memory_bank
        )
        
        # Get noise std from influence
        noise_std = self.noising_module.propose_adaptive_noise_std(influence, knn_distances)
        
        # Mean noise std per spatial position
        noise_std_per_pos = noise_std.mean(dim=1)  # [B*H*W]
        noise_std_map = noise_std_per_pos.reshape(B, H_ref, W_ref)  # [B, H_ref, W_ref]
        
        # Apply noise to each scale with upsampled noise_std_map
        noised_feats = []
        for feat in feats_t:
            _, C, H_f, W_f = feat.shape
            
            # Upsample noise_std_map to match this feature's spatial size
            noise_std_upsampled = F.interpolate(
                noise_std_map.unsqueeze(1),  # [B, 1, H_ref, W_ref]
                size=(H_f, W_f),
                mode='bilinear',
                align_corners=False
            )  # [B, 1, H_f, W_f]
            
            # Generate and apply noise
            noise = torch.randn_like(feat) * noise_std_upsampled
            noised_feats.append(feat + noise)
        
        return noised_feats


@MODEL.register_module
def rdpp_noising(pretrained=False, **kwargs):
    """Factory function to create RDPP Noising model."""
    model = RDPP_NOISING(**kwargs)
    return model


if __name__ == '__main__':
    """Test the model."""
    from argparse import Namespace as _Namespace
    
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
    
    net = RDPP_NOISING(
        model_t=model_t,
        model_s=model_s,
        n_neighbors=9,
        noise_std_range=(0.01, 0.5),
        coreset_sampling_ratio=0.1,
    ).cuda()
    
    x = torch.randn(2, 3, 256, 256).cuda()
    net.eval()
    
    feats_t, feats_s, L_proj = net(x)
    print(f'Teacher features: {[f.shape for f in feats_t]}')
    print(f'Student features: {[f.shape for f in feats_s]}')
