"""
Visualize Adaptive Noise on MVTec Dataset with Ground Truth Comparison.

This script:
1. Loads images from MVTec dataset (both normal and anomaly)
2. Compares noise map with ground truth mask
3. Analyzes if noise is concentrated in anomaly regions
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
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.rdpp_noising import RDPP_NOISING


def denormalize_image(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """Convert normalized tensor back to displayable image."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def create_model(checkpoint_path=None, device='cuda'):
    """Create RDPP noising model."""
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
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        # Try to load memory bank
        if 'memory_bank' in checkpoint:
            model.memory_bank = checkpoint['memory_bank'].to(device)
            model.memory_bank_built = True
            print(f"Loaded memory bank: {model.memory_bank.shape}")
    
    return model.to(device)


def build_memory_bank_from_normal_images(model, normal_dir, device='cuda', max_images=200):
    """Build memory bank from normal training images."""
    print(f"Building memory bank from {normal_dir}...")
    
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    model.eval()
    all_features = []
    
    image_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    with torch.no_grad():
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(normal_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            feats_t = model.net_t(img_tensor)
            ref_feat = feats_t[-1]
            gap_feat = F.adaptive_avg_pool2d(ref_feat, 1).flatten(1).cpu()
            all_features.append(gap_feat)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)} images")
    
    features = torch.cat(all_features, dim=0)
    print(f"Memory bank shape: {features.shape}")
    
    model.memory_bank = features.to(device)
    model.memory_bank_built = True
    
    return model


def extract_noise_maps(model, image_tensor, device='cuda'):
    """Extract noise and influence maps from model."""
    B, C, H_img, W_img = image_tensor.shape
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        feats_t = model.net_t(image_tensor)
        ref_feat = feats_t[-1]
        B, C_ref, H_ref, W_ref = ref_feat.shape
        
        ref_flat = ref_feat.permute(0, 2, 3, 1).reshape(-1, C_ref)
        
        influence, knn_distances = model.noising_module.compute_influence_analytical(
            ref_flat, model.memory_bank
        )
        
        noise_std = model.noising_module.propose_adaptive_noise_std(influence, knn_distances)
        
        # Per position values
        noise_std_per_pos = noise_std.mean(dim=1).reshape(B, H_ref, W_ref)
        influence_per_pos = influence.mean(dim=1).reshape(B, H_ref, W_ref)
        knn_dist_per_pos = knn_distances[:, 0].reshape(B, H_ref, W_ref)  # Nearest neighbor distance
        
        # Normalize
        def normalize_map(m):
            m_min, m_max = m.min(), m.max()
            if m_max - m_min > 1e-8:
                return (m - m_min) / (m_max - m_min)
            return torch.zeros_like(m)
        
        noise_std_norm = normalize_map(noise_std_per_pos)
        influence_norm = normalize_map(influence_per_pos)
        knn_dist_norm = normalize_map(knn_dist_per_pos)
        
        # Upsample to image resolution
        noise_std_image = F.interpolate(
            noise_std_norm.unsqueeze(1), size=(H_img, W_img), mode='bilinear', align_corners=False
        ).squeeze(1)
        
        influence_image = F.interpolate(
            influence_norm.unsqueeze(1), size=(H_img, W_img), mode='bilinear', align_corners=False
        ).squeeze(1)
        
        knn_dist_image = F.interpolate(
            knn_dist_norm.unsqueeze(1), size=(H_img, W_img), mode='bilinear', align_corners=False
        ).squeeze(1)
        
        # Raw noise std for actual noising
        noise_std_raw = F.interpolate(
            noise_std_per_pos.unsqueeze(1), size=(H_img, W_img), mode='bilinear', align_corners=False
        ).squeeze(1)
        
    return {
        'noise_std': noise_std_image.cpu(),
        'influence': influence_image.cpu(),
        'knn_distance': knn_dist_image.cpu(),
        'noise_std_raw': noise_std_raw.cpu(),
    }


def compute_correlation_with_mask(noise_map, gt_mask):
    """Compute correlation between noise map and ground truth mask."""
    noise_flat = noise_map.flatten().numpy()
    mask_flat = gt_mask.flatten().numpy()
    
    # Pearson correlation
    correlation = np.corrcoef(noise_flat, mask_flat)[0, 1]
    
    # IoU-like metric: overlap between high-noise and anomaly regions
    noise_thresh = np.percentile(noise_flat, 75)  # Top 25% noise regions
    high_noise_mask = noise_flat > noise_thresh
    
    if mask_flat.max() > 0:
        intersection = (high_noise_mask & (mask_flat > 0.5)).sum()
        union = (high_noise_mask | (mask_flat > 0.5)).sum()
        iou = intersection / (union + 1e-8)
    else:
        iou = 0.0
    
    # Mean noise in anomaly vs normal regions
    if mask_flat.max() > 0:
        anomaly_mask = mask_flat > 0.5
        normal_mask = ~anomaly_mask
        mean_noise_anomaly = noise_flat[anomaly_mask].mean() if anomaly_mask.sum() > 0 else 0
        mean_noise_normal = noise_flat[normal_mask].mean() if normal_mask.sum() > 0 else 0
    else:
        mean_noise_anomaly = 0
        mean_noise_normal = noise_flat.mean()
    
    return {
        'correlation': correlation,
        'iou': iou,
        'mean_noise_anomaly': mean_noise_anomaly,
        'mean_noise_normal': mean_noise_normal,
        'noise_ratio': mean_noise_anomaly / (mean_noise_normal + 1e-8),
    }


def visualize_with_gt(image, noise_maps, gt_mask, title='', save_path=None):
    """Visualize noise maps alongside ground truth mask."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Denormalize image
    img_display = denormalize_image(image).permute(1, 2, 0).numpy()
    
    # Row 1: Original, GT Mask, Influence, Noise STD
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask.numpy(), cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(noise_maps['influence'][0].numpy(), cmap='jet')
    axes[0, 2].set_title('Influence Map')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 3].imshow(noise_maps['noise_std'][0].numpy(), cmap='jet')
    axes[0, 3].set_title('Adaptive Noise STD')
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Overlays
    axes[1, 0].imshow(img_display)
    axes[1, 0].imshow(gt_mask.numpy(), cmap='Reds', alpha=0.4)
    axes[1, 0].set_title('GT Overlay')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_display)
    axes[1, 1].imshow(noise_maps['influence'][0].numpy(), cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Influence Overlay')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img_display)
    axes[1, 2].imshow(noise_maps['noise_std'][0].numpy(), cmap='jet', alpha=0.5)
    axes[1, 2].set_title('Noise STD Overlay')
    axes[1, 2].axis('off')
    
    im3 = axes[1, 3].imshow(noise_maps['knn_distance'][0].numpy(), cmap='jet')
    axes[1, 3].set_title('KNN Distance (anomaly score)')
    axes[1, 3].axis('off')
    plt.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    

def visualize_comparison(images_info, save_path=None):
    """Compare noise maps across multiple images."""
    n_images = len(images_info)
    fig, axes = plt.subplots(n_images, 5, figsize=(25, 5 * n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, info in enumerate(images_info):
        img_display = denormalize_image(info['image']).permute(1, 2, 0).numpy()
        
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"{info['name']}\nOriginal")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(info['gt_mask'].numpy(), cmap='gray')
        axes[i, 1].set_title('GT Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(info['noise_maps']['noise_std'][0].numpy(), cmap='jet')
        axes[i, 2].set_title('Noise STD Map')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(img_display)
        axes[i, 3].imshow(info['noise_maps']['noise_std'][0].numpy(), cmap='jet', alpha=0.5)
        axes[i, 3].set_title('Noise Overlay')
        axes[i, 3].axis('off')
        
        # Metrics
        metrics = info['metrics']
        metrics_text = f"Correlation: {metrics['correlation']:.3f}\n"
        metrics_text += f"Noise Ratio: {metrics['noise_ratio']:.2f}\n"
        metrics_text += f"IoU: {metrics['iou']:.3f}"
        axes[i, 4].text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=12,
                        transform=axes[i, 4].transAxes, family='monospace')
        axes[i, 4].set_title('Metrics')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Adaptive Noise on MVTec')
    parser.add_argument('--data-root', type=str, default='data/mvtec',
                        help='Path to MVTec dataset root')
    parser.add_argument('--class-name', type=str, default='bottle',
                        help='MVTec class name')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='noise_vis_output',
                        help='Output directory for visualizations')
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Paths
    class_dir = os.path.join(args.data_root, args.class_name)
    train_dir = os.path.join(class_dir, 'train', 'good')
    test_dir = os.path.join(class_dir, 'test')
    gt_dir = os.path.join(class_dir, 'ground_truth')
    
    print(f"\nDataset paths:")
    print(f"  Train (good): {train_dir}")
    print(f"  Test: {test_dir}")
    print(f"  Ground truth: {gt_dir}")
    
    # Check paths exist
    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory not found: {train_dir}")
        print("Please ensure MVTec dataset is downloaded to data/mvtec/")
        return
    
    # Create model
    print("\nCreating model...")
    model = create_model(args.checkpoint, device)
    
    # Build memory bank from normal images
    if not model.memory_bank_built:
        model = build_memory_bank_from_normal_images(model, train_dir, device)
    
    # Image transforms
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    
    mask_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    
    # Find anomaly types
    defect_types = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    print(f"\nDefect types found: {defect_types}")
    
    images_info = []
    
    for defect_type in defect_types:
        defect_dir = os.path.join(test_dir, defect_type)
        image_files = [f for f in os.listdir(defect_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        n_samples_type = min(args.n_samples, len(image_files))
        
        for img_file in image_files[:n_samples_type]:
            img_path = os.path.join(defect_dir, img_file)
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            # Load ground truth mask
            if defect_type == 'good':
                gt_mask = torch.zeros(256, 256)
            else:
                gt_path = os.path.join(gt_dir, defect_type, img_file.replace('.png', '_mask.png'))
                if not os.path.exists(gt_path):
                    # Try other extensions
                    gt_path = os.path.join(gt_dir, defect_type, img_file.split('.')[0] + '_mask.png')
                
                if os.path.exists(gt_path):
                    gt_img = Image.open(gt_path).convert('L')
                    gt_mask = mask_transform(gt_img).squeeze(0)
                else:
                    print(f"  Warning: GT mask not found for {img_file}")
                    gt_mask = torch.zeros(256, 256)
            
            # Extract noise maps
            noise_maps = extract_noise_maps(model, img_tensor, device)
            
            # Compute metrics
            metrics = compute_correlation_with_mask(noise_maps['noise_std'][0], gt_mask)
            
            name = f"{args.class_name}/{defect_type}/{img_file}"
            print(f"\n{name}")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  Noise Ratio (anomaly/normal): {metrics['noise_ratio']:.4f}")
            print(f"  IoU (high-noise vs anomaly): {metrics['iou']:.4f}")
            
            images_info.append({
                'name': name,
                'image': img_tensor[0],
                'gt_mask': gt_mask,
                'noise_maps': noise_maps,
                'metrics': metrics,
            })
            
            # Individual visualization
            save_path = os.path.join(args.output_dir, f"{args.class_name}_{defect_type}_{img_file}")
            visualize_with_gt(
                img_tensor[0], noise_maps, gt_mask,
                title=name, save_path=save_path
            )
    
    # Overall comparison
    if len(images_info) > 1:
        comparison_path = os.path.join(args.output_dir, f"{args.class_name}_comparison.png")
        visualize_comparison(images_info, save_path=comparison_path)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    anomaly_correlations = [info['metrics']['correlation'] for info in images_info 
                           if info['gt_mask'].max() > 0]
    anomaly_ratios = [info['metrics']['noise_ratio'] for info in images_info 
                      if info['gt_mask'].max() > 0]
    anomaly_ious = [info['metrics']['iou'] for info in images_info 
                    if info['gt_mask'].max() > 0]
    
    if anomaly_correlations:
        print(f"Anomaly samples ({len(anomaly_correlations)}):")
        print(f"  Mean Correlation: {np.mean(anomaly_correlations):.4f} ± {np.std(anomaly_correlations):.4f}")
        print(f"  Mean Noise Ratio: {np.mean(anomaly_ratios):.4f} ± {np.std(anomaly_ratios):.4f}")
        print(f"  Mean IoU: {np.mean(anomaly_ious):.4f} ± {np.std(anomaly_ious):.4f}")
    
    normal_samples = [info for info in images_info if info['gt_mask'].max() == 0]
    if normal_samples:
        normal_noise_means = [info['noise_maps']['noise_std'][0].mean().item() for info in normal_samples]
        print(f"\nNormal samples ({len(normal_samples)}):")
        print(f"  Mean Noise STD: {np.mean(normal_noise_means):.4f}")
    
    print("\nVisualization complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
