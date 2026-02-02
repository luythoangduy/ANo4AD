"""
Visualize Adaptive Noise from RDPP Noising Model.

This script:
1. Loads a trained RDPP noising model with memory bank
2. Extracts adaptive noise std maps from features
3. Upsamples noise map to image resolution
4. Visualizes original image, noise map, and noised image side-by-side
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model
from model.rdpp_noising import RDPP_NOISING


def denormalize_image(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def create_test_model():
    """Create RDPP noising model for testing."""
    from argparse import Namespace
    
    model_t = Namespace()
    model_t.name = 'timm_wide_resnet50_2'
    model_t.kwargs = dict(
        pretrained=False,
        checkpoint_path='model/pretrain/wide_resnet50_racm-8234f177.pth',
        strict=False,
        features_only=True,
        out_indices=[1, 2, 3]
    )
    
    model_s = Namespace()
    model_s.name = 'de_wide_resnet50_2'
    model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
    
    model = RDPP_NOISING(
        model_t=model_t,
        model_s=model_s,
        n_neighbors=9,
        noise_std_range=(0.01, 0.5),
        coreset_sampling_ratio=0.1,
        enable_noise=True,
    )
    return model


def extract_noise_visualization(model, images, device='cuda'):
    """
    Extract and visualize adaptive noise maps.
    
    Returns:
        noise_std_map_image: [B, H_img, W_img] noise std upsampled to image size
        influence_map_image: [B, H_img, W_img] influence map upsampled to image size
        noised_image: [B, C, H, W] image with noise applied in image space
    """
    B, C, H_img, W_img = images.shape
    model.eval()
    model.to(device)
    images = images.to(device)
    
    with torch.no_grad():
        # Extract teacher features
        feats_t = model.net_t(images)
        
        # Get reference feature (last layer, 1024 channels for wide_resnet50)
        ref_feat = feats_t[-1]  # [B, 1024, H_ref, W_ref]
        B, C_ref, H_ref, W_ref = ref_feat.shape
        
        print(f"Image size: {H_img}x{W_img}")
        print(f"Reference feature size: {H_ref}x{W_ref} with {C_ref} channels")
        
        # Flatten for influence computation
        ref_flat = ref_feat.permute(0, 2, 3, 1).reshape(-1, C_ref)  # [B*H*W, C]
        
        # Compute influence using analytical gradient
        influence, knn_distances = model.noising_module.compute_influence_analytical(
            ref_flat, model.memory_bank
        )
        
        # Get noise std from influence
        noise_std = model.noising_module.propose_adaptive_noise_std(influence, knn_distances)
        
        # Mean noise std per spatial position
        noise_std_per_pos = noise_std.mean(dim=1)  # [B*H*W]
        noise_std_map = noise_std_per_pos.reshape(B, H_ref, W_ref)  # [B, H_ref, W_ref]
        
        # Mean influence per spatial position
        influence_per_pos = influence.mean(dim=1)  # [B*H*W]
        influence_map = influence_per_pos.reshape(B, H_ref, W_ref)  # [B, H_ref, W_ref]
        
        # Normalize influence map for visualization
        influence_min = influence_map.min()
        influence_max = influence_map.max()
        if influence_max - influence_min > 1e-8:
            influence_map_norm = (influence_map - influence_min) / (influence_max - influence_min)
        else:
            influence_map_norm = torch.zeros_like(influence_map)
        
        # Normalize noise std map for visualization
        noise_min = noise_std_map.min()
        noise_max = noise_std_map.max()
        if noise_max - noise_min > 1e-8:
            noise_std_map_norm = (noise_std_map - noise_min) / (noise_max - noise_min)
        else:
            noise_std_map_norm = torch.zeros_like(noise_std_map)
        
        # Upsample to image resolution
        noise_std_map_image = F.interpolate(
            noise_std_map.unsqueeze(1),  # [B, 1, H_ref, W_ref]
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H_img, W_img]
        
        noise_std_map_norm_image = F.interpolate(
            noise_std_map_norm.unsqueeze(1),
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        influence_map_image = F.interpolate(
            influence_map_norm.unsqueeze(1),
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H_img, W_img]
        
        # Create noised image in image space for visualization
        # Apply noise to image using upsampled noise_std_map
        noise = torch.randn_like(images) * noise_std_map_image.unsqueeze(1)
        noised_images = images + noise
        
        # Get KNN distance map for additional visualization
        knn_dist_mean = knn_distances.mean(dim=1)  # [B*H*W]
        knn_dist_map = knn_dist_mean.reshape(B, H_ref, W_ref)
        knn_dist_min = knn_dist_map.min()
        knn_dist_max = knn_dist_map.max()
        if knn_dist_max - knn_dist_min > 1e-8:
            knn_dist_map_norm = (knn_dist_map - knn_dist_min) / (knn_dist_max - knn_dist_min)
        else:
            knn_dist_map_norm = torch.zeros_like(knn_dist_map)
        
        knn_dist_map_image = F.interpolate(
            knn_dist_map_norm.unsqueeze(1),
            size=(H_img, W_img),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
    return {
        'noise_std_map': noise_std_map_norm_image.cpu(),
        'influence_map': influence_map_image.cpu(),
        'knn_distance_map': knn_dist_map_image.cpu(),
        'noised_images': noised_images.cpu(),
        'noise_std_raw': noise_std_map_image.cpu(),
    }


def visualize_single_image(original_img, noise_info, save_path=None, title_prefix=''):
    """
    Visualize noise analysis for a single image.
    
    Args:
        original_img: [C, H, W] normalized tensor
        noise_info: dict with noise_std_map, influence_map, noised_images
        save_path: optional path to save figure
        title_prefix: prefix for figure title
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Denormalize images for display
    orig_display = denormalize_image(original_img).permute(1, 2, 0).numpy()
    noised_display = denormalize_image(noise_info['noised_images'][0]).permute(1, 2, 0).numpy()
    noised_display = np.clip(noised_display, 0, 1)
    
    # Original image
    axes[0, 0].imshow(orig_display)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Noised image
    axes[0, 1].imshow(noised_display)
    axes[0, 1].set_title('Noised Image')
    axes[0, 1].axis('off')
    
    # Difference (amplified)
    diff = np.abs(noised_display - orig_display)
    diff_amplified = np.clip(diff * 10, 0, 1)  # Amplify for visibility
    axes[0, 2].imshow(diff_amplified)
    axes[0, 2].set_title('Noise Applied (×10)')
    axes[0, 2].axis('off')
    
    # Influence map (heat map)
    influence = noise_info['influence_map'][0].numpy()
    im1 = axes[1, 0].imshow(influence, cmap='jet')
    axes[1, 0].set_title('Influence Map (Gradient-based)')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # KNN Distance map
    knn_dist = noise_info['knn_distance_map'][0].numpy()
    im2 = axes[1, 1].imshow(knn_dist, cmap='jet')
    axes[1, 1].set_title('KNN Distance Map')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Noise STD map
    noise_std = noise_info['noise_std_map'][0].numpy()
    im3 = axes[1, 2].imshow(noise_std, cmap='jet')
    axes[1, 2].set_title('Adaptive Noise STD Map')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle(f'{title_prefix}Adaptive Noise Visualization', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_overlay(original_img, noise_info, save_path=None):
    """
    Visualize noise map overlaid on original image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Denormalize image
    orig_display = denormalize_image(original_img).permute(1, 2, 0).numpy()
    
    # Original
    axes[0].imshow(orig_display)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Overlay influence map
    axes[1].imshow(orig_display)
    influence = noise_info['influence_map'][0].numpy()
    axes[1].imshow(influence, cmap='jet', alpha=0.5)
    axes[1].set_title('Influence Overlay')
    axes[1].axis('off')
    
    # Overlay noise std map
    axes[2].imshow(orig_display)
    noise_std = noise_info['noise_std_map'][0].numpy()
    axes[2].imshow(noise_std, cmap='jet', alpha=0.5)
    axes[2].set_title('Noise STD Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved overlay visualization to {save_path}")
    
    plt.show()


def build_dummy_memory_bank(model, device='cuda', n_samples=100):
    """Build a dummy memory bank for testing without full dataset."""
    print("Building dummy memory bank for visualization testing...")
    
    model.to(device)
    model.eval()
    
    # Generate random images to build memory bank
    all_features = []
    
    with torch.no_grad():
        for i in range(n_samples // 4):
            dummy_images = torch.randn(4, 3, 256, 256).to(device)
            feats_t = model.net_t(dummy_images)
            ref_feat = feats_t[-1]
            gap_feat = F.adaptive_avg_pool2d(ref_feat, 1).flatten(1).cpu()
            all_features.append(gap_feat)
    
    features = torch.cat(all_features, dim=0)
    print(f"Memory bank shape: {features.shape}")
    
    model.memory_bank = features.to(device)
    model.memory_bank_built = True
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Visualize Adaptive Noise')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (optional, uses random if not provided)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--output', type=str, default='noise_visualization.png',
                        help='Output path for visualization')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    print("Creating RDPP Noising model...")
    model = create_test_model()
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    
    # Build memory bank (dummy for testing, or from checkpoint)
    if not model.memory_bank_built:
        model = build_dummy_memory_bank(model, device=device, n_samples=100)
    
    # Prepare image
    transform = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}")
        img = Image.open(args.image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        title_prefix = os.path.basename(args.image) + ' - '
    else:
        print("Using random test image")
        img_tensor = torch.randn(1, 3, args.size, args.size)
        # Normalize to valid range
        img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        img_tensor = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(img_tensor)
        title_prefix = 'Random - '
    
    print(f"Image tensor shape: {img_tensor.shape}")
    
    # Extract noise visualization
    print("Extracting adaptive noise maps...")
    noise_info = extract_noise_visualization(model, img_tensor, device=device)
    
    print(f"\nNoise statistics:")
    print(f"  Noise STD range: [{noise_info['noise_std_raw'][0].min():.4f}, {noise_info['noise_std_raw'][0].max():.4f}]")
    print(f"  Influence range: [{noise_info['influence_map'][0].min():.4f}, {noise_info['influence_map'][0].max():.4f}]")
    print(f"  KNN Distance range: [{noise_info['knn_distance_map'][0].min():.4f}, {noise_info['knn_distance_map'][0].max():.4f}]")
    
    # Visualize
    print("Generating visualization...")
    visualize_single_image(
        img_tensor[0], 
        noise_info, 
        save_path=args.output,
        title_prefix=title_prefix
    )
    
    # Also save overlay version
    overlay_path = args.output.replace('.png', '_overlay.png')
    visualize_overlay(img_tensor[0], noise_info, save_path=overlay_path)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
