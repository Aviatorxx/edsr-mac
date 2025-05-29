#!/usr/bin/env python3
"""
EDSR vs Enhanced EDSR Comprehensive Comparison Test
Generate comparison images and curves for original EDSR and Enhanced EDSR
Fixed version with correct paths and PSNR calculation - MPS Accelerated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Songti SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# Original EDSR Model Definition (Same as bicubic comparison)
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Module):
    def __init__(self, scale, n_feats, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(np.log2(scale))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        return self.body(x)

class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, n_colors=3, res_scale=1):
        super(EDSR, self).__init__()
        self.scale = scale
        
        kernel_size = 3
        act = nn.ReLU(True)
        
        # define head module
        m_head = [nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size//2)]
        
        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
        
        # define tail module
        m_tail = [
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, n_colors, kernel_size, padding=kernel_size//2)
        ]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

# =============================================================================
# Enhanced EDSR Model Definition
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(n_feats // reduction, n_feats, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_cat = self.conv1(x_cat)
        return self.sigmoid(x_cat)

class EnhancedResidualDenseBlock(nn.Module):
    def __init__(self, n_feats, growth_rate=32, res_scale=0.2):
        super(EnhancedResidualDenseBlock, self).__init__()
        self.res_scale = res_scale

        # Dense connections
        self.conv1 = nn.Conv2d(n_feats, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(n_feats + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(n_feats + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(n_feats + 3 * growth_rate, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(n_feats + 4 * growth_rate, n_feats, 3, padding=1)

        # Attention modules
        self.ca = ChannelAttention(n_feats)
        self.sa = SpatialAttention()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))

        # Apply attention
        x5 = x5 * self.ca(x5)
        x5 = x5 * self.sa(x5)

        return x + x5 * self.res_scale

class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self, n_feats):
        super(MultiScaleFeatureExtraction, self).__init__()
        self.conv1x1 = nn.Conv2d(n_feats, n_feats // 4, 1)
        self.conv3x3 = nn.Conv2d(n_feats, n_feats // 4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(n_feats, n_feats // 4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(n_feats, n_feats // 4, 7, padding=3)
        self.fusion = nn.Conv2d(n_feats, n_feats, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1x1(x))
        x2 = self.relu(self.conv3x3(x))
        x3 = self.relu(self.conv5x5(x))
        x4 = self.relu(self.conv7x7(x))

        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.fusion(out)

        return out + x

class ProgressiveUpsampler(nn.Module):
    def __init__(self, scale, n_feats, n_colors=3):
        super(ProgressiveUpsampler, self).__init__()
        self.scale = scale

        if scale == 4:
            # 4x upsampling: 2x -> 2x
            self.up1 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
            self.up2 = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
            # Refinement after each upsampling
            self.refine1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.refine2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
            self.final = nn.Conv2d(n_feats, n_colors, 3, padding=1)
        else:
            raise NotImplementedError(f"Scale {scale} not implemented")

    def forward(self, x):
        if self.scale == 4:
            x = self.up1(x)
            x = self.refine1(x) + x
            x = self.up2(x)
            x = self.refine2(x) + x
            x = self.final(x)
        return x

class EAD_EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, n_colors=3, growth_rate=32):
        super(EAD_EDSR, self).__init__()

        # Feature extraction
        self.head = nn.Conv2d(n_colors, n_feats, 3, padding=1)

        # Multi-scale feature extraction
        self.ms_extract = MultiScaleFeatureExtraction(n_feats)

        # Enhanced residual dense blocks
        self.body = nn.ModuleList([
            EnhancedResidualDenseBlock(n_feats, growth_rate)
            for _ in range(n_resblocks)
        ])

        # Global feature fusion
        self.global_fusion = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # Global residual learning
        self.global_res = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # Progressive upsampling
        self.tail = ProgressiveUpsampler(scale, n_feats, n_colors)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        x = self.head(x)
        head_output = x

        # Multi-scale feature extraction
        x = self.ms_extract(x)

        # Enhanced residual dense blocks
        for block in self.body:
            x = block(x)

        # Global feature fusion and residual
        x = self.global_fusion(x)
        x = self.global_res(x) + head_output

        # Progressive upsampling
        x = self.tail(x)

        return x

# =============================================================================
# Utility Functions (Same as bicubic comparison)
# =============================================================================

def calculate_psnr_tensor(img1, img2, max_val=1.0):
    """Calculate PSNR between two tensor images (same as edsr_bicubic_comparison.py)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim_tensor(img1, img2):
    """Calculate SSIM between two tensor images (same as edsr_bicubic_comparison.py)"""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    if len(img1_np.shape) == 4:
        img1_np = img1_np[0]
        img2_np = img2_np[0]
    
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))
    
    ssim_vals = []
    for i in range(3):
        ssim_val = structural_similarity(
            img1_np[:, :, i], img2_np[:, :, i], 
            data_range=1.0
        )
        ssim_vals.append(ssim_val)
    
    return np.mean(ssim_vals)

# =============================================================================
# Model Processing Functions
# =============================================================================

def process_edsr_model(lr_path, model, device):
    """Process with EDSR model (unified function) - MPS optimized"""
    to_tensor = transforms.ToTensor()
    
    lr_img = Image.open(lr_path).convert('RGB')
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
    
    try:
        with torch.no_grad():
            # Use autocast for MPS if available
            if device.type == 'mps':
                with torch.autocast(device_type='cpu'):  # MPS doesn't support autocast yet
                    sr_tensor = model(lr_tensor)
            else:
                sr_tensor = model(lr_tensor)
            
        # Move to CPU for processing
        sr_tensor_cpu = sr_tensor.cpu()
        sr_np = (sr_tensor_cpu[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        sr_img = Image.fromarray(sr_np)
        
        # Clean up GPU memory
        del lr_tensor, sr_tensor
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'lr_img': lr_img,
            'sr_img': sr_img,
            'sr_tensor': sr_tensor_cpu
        }
        
    except Exception as e:
        print(f"    ⚠️  GPU processing failed, falling back to CPU: {str(e)}")
        # Fallback to CPU processing
        lr_tensor_cpu = lr_tensor.cpu()
        model_cpu = model.cpu()
        
        with torch.no_grad():
            sr_tensor = model_cpu(lr_tensor_cpu)
        
        sr_np = (sr_tensor[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        sr_img = Image.fromarray(sr_np)
        
        # Move model back to original device
        model.to(device)
        
        return {
            'lr_img': lr_img,
            'sr_img': sr_img,
            'sr_tensor': sr_tensor
        }

def calculate_edsr_metrics(hr_path, sr_tensor):
    """Calculate metrics between HR and SR result (same as edsr_bicubic_comparison.py) - MPS optimized"""
    to_tensor = transforms.ToTensor()
    
    hr_img = Image.open(hr_path).convert('RGB')
    hr_tensor = to_tensor(hr_img).unsqueeze(0)
    
    # Ensure both tensors are on CPU for metric calculation
    if sr_tensor.device.type != 'cpu':
        sr_tensor = sr_tensor.cpu()
    if hr_tensor.device.type != 'cpu':
        hr_tensor = hr_tensor.cpu()
    
    # Handle size mismatch by cropping/resizing like in edsr_bicubic_comparison.py
    if hr_tensor.shape != sr_tensor.shape:
        h_hr, w_hr = hr_tensor.shape[2:]
        h_sr, w_sr = sr_tensor.shape[2:]
        
        if h_hr > h_sr or w_hr > w_sr:
            # Crop HR to match SR
            h_min = min(h_hr, h_sr)
            w_min = min(w_hr, w_sr)
            hr_tensor = hr_tensor[:, :, :h_min, :w_min]
            sr_tensor = sr_tensor[:, :, :h_min, :w_min]
        else:
            # Resize SR to match HR
            sr_tensor = F.interpolate(sr_tensor, size=(h_hr, w_hr), mode='bilinear', align_corners=False)
    
    # Calculate metrics using tensor method (same as edsr_bicubic_comparison.py)
    psnr_val = calculate_psnr_tensor(sr_tensor, hr_tensor).item()
    ssim_val = calculate_ssim_tensor(sr_tensor, hr_tensor)
    
    return psnr_val, ssim_val, hr_img

def create_4way_comparison_plot(original_result, enhanced_result, hr_img, 
                               original_psnr, original_ssim, 
                               enhanced_psnr, enhanced_ssim, 
                               filename, output_dir):
    """Create 4-image comparison plot: LR, Original EDSR, Enhanced EDSR, HR"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # LR (upsampled with nearest neighbor for visualization)
    lr_display = original_result['lr_img'].resize(hr_img.size, Image.NEAREST)
    axes[0, 0].imshow(lr_display)
    axes[0, 0].set_title('LR (Nearest Upsample)', fontsize=16, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Original EDSR
    axes[0, 1].imshow(original_result['sr_img'])
    axes[0, 1].set_title(f'Original EDSR\nPSNR: {original_psnr:.2f} dB\nSSIM: {original_ssim:.4f}', 
                        fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Enhanced EDSR
    axes[1, 0].imshow(enhanced_result['sr_img'])
    axes[1, 0].set_title(f'Enhanced EDSR\nPSNR: {enhanced_psnr:.2f} dB\nSSIM: {enhanced_ssim:.4f}', 
                        fontsize=16, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    # HR
    axes[1, 1].imshow(hr_img)
    axes[1, 1].set_title('Ground Truth (HR)', fontsize=16, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add improvement info
    psnr_improvement = enhanced_psnr - original_psnr
    ssim_improvement = enhanced_ssim - original_ssim
    
    fig.suptitle(f'{filename}\nEnhanced EDSR Improvement: +{psnr_improvement:.2f} dB PSNR, +{ssim_improvement:.4f} SSIM', 
                fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return save_path

def test_dataset(dataset_name, original_model, enhanced_model, device, output_dir):
    """Test a complete dataset with both models - MPS optimized"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name} with {device} acceleration")
    print(f"{'='*60}")
    
    hr_dir = f"../data/test/{dataset_name}/HR"
    lr_dir = f"../data/test/{dataset_name}/LR"
    
    if not (os.path.exists(hr_dir) and os.path.exists(lr_dir)):
        print(f"Dataset {dataset_name} not found!")
        return None
    
    # Get all image files
    hr_files = sorted([f for f in os.listdir(hr_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    results = []
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    
    # Progress tracking
    print(f"Processing {len(hr_files)} images...")
    
    for i, hr_file in enumerate(tqdm(hr_files, desc=f"Processing {dataset_name}")):
        lr_file = hr_file  # Assume same naming convention
        hr_path = os.path.join(hr_dir, hr_file)
        lr_path = os.path.join(lr_dir, lr_file)
        
        if not os.path.exists(lr_path):
            print(f"Warning: LR file not found for {hr_file}")
            continue
        
        try:
            # Process with original EDSR
            original_result = process_edsr_model(lr_path, original_model, device)
            
            # Process with enhanced EDSR
            enhanced_result = process_edsr_model(lr_path, enhanced_model, device)
            
            # Calculate metrics against HR
            original_psnr, original_ssim, hr_img = calculate_edsr_metrics(
                hr_path, original_result['sr_tensor'])
            enhanced_psnr, enhanced_ssim, _ = calculate_edsr_metrics(
                hr_path, enhanced_result['sr_tensor'])
            
            # Create visualization
            filename = os.path.splitext(hr_file)[0]
            save_path = create_4way_comparison_plot(
                original_result, enhanced_result, hr_img,
                original_psnr, original_ssim,
                enhanced_psnr, enhanced_ssim,
                filename, dataset_output_dir
            )
            
            # Store results
            results.append({
                'filename': hr_file,
                'original_psnr': original_psnr,
                'original_ssim': original_ssim,
                'enhanced_psnr': enhanced_psnr,
                'enhanced_ssim': enhanced_ssim,
                'psnr_improvement': enhanced_psnr - original_psnr,
                'ssim_improvement': enhanced_ssim - original_ssim,
                'comparison_path': save_path
            })
            
            # Brief progress update (only for first and last few images)
            if i < 2 or i >= len(hr_files) - 2:
                print(f"  ✓ {hr_file}: Original={original_psnr:.1f}dB, Enhanced={enhanced_psnr:.1f}dB, Gain=+{enhanced_psnr-original_psnr:.1f}dB")
            
            # Clean up memory periodically
            del original_result, enhanced_result, hr_img
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error processing {hr_file}: {str(e)}")
            continue
    
    if results:
        # Calculate averages
        avg_original_psnr = np.mean([r['original_psnr'] for r in results])
        avg_original_ssim = np.mean([r['original_ssim'] for r in results])
        avg_enhanced_psnr = np.mean([r['enhanced_psnr'] for r in results])
        avg_enhanced_ssim = np.mean([r['enhanced_ssim'] for r in results])
        avg_psnr_improvement = np.mean([r['psnr_improvement'] for r in results])
        avg_ssim_improvement = np.mean([r['ssim_improvement'] for r in results])
        
        print(f"\n🎯 {dataset_name} Summary Results:")
        print(f"{'Method':<15} {'PSNR (dB)':<12} {'SSIM':<8}")
        print("-" * 37)
        print(f"{'Original EDSR':<15} {avg_original_psnr:<12.2f} {avg_original_ssim:<8.4f}")
        print(f"{'Enhanced EDSR':<15} {avg_enhanced_psnr:<12.2f} {avg_enhanced_ssim:<8.4f}")
        print(f"{'Improvement':<15} {avg_psnr_improvement:<12.2f} {avg_ssim_improvement:<8.4f}")
        
        # Save results to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(dataset_output_dir, f"{dataset_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed results saved to: {csv_path}")
        
        return {
            'dataset': dataset_name,
            'avg_original_psnr': avg_original_psnr,
            'avg_original_ssim': avg_original_ssim,
            'avg_enhanced_psnr': avg_enhanced_psnr,
            'avg_enhanced_ssim': avg_enhanced_ssim,
            'avg_psnr_improvement': avg_psnr_improvement,
            'avg_ssim_improvement': avg_ssim_improvement,
            'individual_results': results
        }
    
    return None

def create_comparison_curves(all_results, output_dir):
    """Create comparison curves and charts"""
    print("\n📊 Creating comparison curves...")
    
    # Extract data for plotting
    datasets = [r['dataset'] for r in all_results]
    original_psnr = [r['avg_original_psnr'] for r in all_results]
    enhanced_psnr = [r['avg_enhanced_psnr'] for r in all_results]
    original_ssim = [r['avg_original_ssim'] for r in all_results]
    enhanced_ssim = [r['avg_enhanced_ssim'] for r in all_results]
    psnr_improvements = [r['avg_psnr_improvement'] for r in all_results]
    ssim_improvements = [r['avg_ssim_improvement'] for r in all_results]
    
    # Create comprehensive comparison chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # PSNR comparison
    bars1 = ax1.bar(x - width/2, original_psnr, width, label='Original EDSR', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, enhanced_psnr, width, label='Enhanced EDSR', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=14, fontweight='bold')
    ax1.set_title('PSNR Comparison: Original vs Enhanced EDSR', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # SSIM comparison
    bars3 = ax2.bar(x - width/2, original_ssim, width, label='Original EDSR', 
                   color='lightgreen', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x + width/2, enhanced_ssim, width, label='Enhanced EDSR', 
                   color='orange', alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax2.set_ylabel('SSIM', fontsize=14, fontweight='bold')
    ax2.set_title('SSIM Comparison: Original vs Enhanced EDSR', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # PSNR improvement
    bars5 = ax3.bar(x, psnr_improvements, color='gold', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax3.set_ylabel('PSNR Improvement (dB)', fontsize=14, fontweight='bold')
    ax3.set_title('PSNR Improvement (Enhanced - Original)', fontsize=16, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars5:
        height = bar.get_height()
        sign = "+" if height >= 0 else ""
        ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                f'{sign}{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # SSIM improvement
    bars6 = ax4.bar(x, ssim_improvements, color='mediumseagreen', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax4.set_ylabel('SSIM Improvement', fontsize=14, fontweight='bold')
    ax4.set_title('SSIM Improvement (Enhanced - Original)', fontsize=16, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars6:
        height = bar.get_height()
        sign = "+" if height >= 0 else ""
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.005 if height >= 0 else -0.005),
                f'{sign}{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "comparison_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Comparison curves saved to: {save_path}")

def main():
    """Main comparison function"""
    print("EDSR vs Enhanced EDSR Comprehensive Comparison (MPS Accelerated)")
    print("=" * 70)
    
    # Set device with MPS priority
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU (no GPU acceleration available)")
    
    print(f"Device: {device}")
    
    # Check model paths (corrected for metrics directory)
    original_model_path = "../models/edsr_x4/best.pth"
    enhanced_model_path = "../models/ead_edsr_x4/best.pth"
    
    # Debug path checking
    from pathlib import Path
    original_path_abs = Path(original_model_path).resolve()
    enhanced_path_abs = Path(enhanced_model_path).resolve()
    
    print(f"Looking for Original EDSR model at: {original_path_abs}")
    print(f"Looking for Enhanced EDSR model at: {enhanced_path_abs}")
    
    if not (os.path.exists(original_model_path) or original_path_abs.exists()):
        print(f"❌ Original EDSR model not found!")
        return
    
    if not (os.path.exists(enhanced_model_path) or enhanced_path_abs.exists()):
        print(f"❌ Enhanced EDSR model not found!")
        print("Please train the Enhanced EDSR model first.")
        return
    
    print(f"✓ Found Original EDSR model")
    print(f"✓ Found Enhanced EDSR model")
    
    # Load models
    print("\n🔄 Loading models...")
    
    try:
        # Load original EDSR
        original_model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(device)
        original_checkpoint = torch.load(original_model_path, map_location=device, weights_only=False)
        original_model.load_state_dict(original_checkpoint['model_state_dict'])
        original_model.eval()
        print(f"✓ Loaded Original EDSR from epoch {original_checkpoint['epoch']}")
        
        # Load enhanced EDSR
        enhanced_model = EAD_EDSR(n_resblocks=10, n_feats=64, scale=4, growth_rate=24).to(device)
        enhanced_checkpoint = torch.load(enhanced_model_path, map_location=device, weights_only=False)
        enhanced_model.load_state_dict(enhanced_checkpoint['model_state_dict'])
        enhanced_model.eval()
        print(f"✓ Loaded Enhanced EDSR from epoch {enhanced_checkpoint['epoch']}")
        
        # Warm up GPU
        if device.type in ['mps', 'cuda']:
            print("🔥 Warming up GPU...")
            dummy_input = torch.randn(1, 3, 64, 64).to(device)
            with torch.no_grad():
                _ = original_model(dummy_input)
                _ = enhanced_model(dummy_input)
            del dummy_input
            if device.type == 'mps':
                torch.mps.empty_cache()
            elif device.type == 'cuda':
                torch.cuda.empty_cache()
            print("✓ GPU warmed up")
            
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print("Falling back to CPU...")
        device = torch.device("cpu")
        
        # Reload on CPU
        original_model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(device)
        original_checkpoint = torch.load(original_model_path, map_location=device, weights_only=False)
        original_model.load_state_dict(original_checkpoint['model_state_dict'])
        original_model.eval()
        
        enhanced_model = EAD_EDSR(n_resblocks=10, n_feats=64, scale=4, growth_rate=24).to(device)
        enhanced_checkpoint = torch.load(enhanced_model_path, map_location=device, weights_only=False)
        enhanced_model.load_state_dict(enhanced_checkpoint['model_state_dict'])
        enhanced_model.eval()
    
    # Output directory
    output_dir = "../results/edsr_and_enhanced"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Results will be saved to: {output_dir}")
    
    # Test datasets
    datasets = ["Set5", "Set14"]
    all_results = []
    
    for dataset in datasets:
        result = test_dataset(dataset, original_model, enhanced_model, device, output_dir)
        if result:
            all_results.append(result)
    
    # Create overall summary and curves
    if all_results:
        print(f"\n{'='*70}")
        print("OVERALL COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        # Create comparison curves
        create_comparison_curves(all_results, output_dir)
        
        # Calculate overall averages
        overall_original_psnr = np.mean([r['avg_original_psnr'] for r in all_results])
        overall_original_ssim = np.mean([r['avg_original_ssim'] for r in all_results])
        overall_enhanced_psnr = np.mean([r['avg_enhanced_psnr'] for r in all_results])
        overall_enhanced_ssim = np.mean([r['avg_enhanced_ssim'] for r in all_results])
        overall_psnr_improvement = np.mean([r['avg_psnr_improvement'] for r in all_results])
        overall_ssim_improvement = np.mean([r['avg_ssim_improvement'] for r in all_results])
        
        # Create and save summary
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Dataset': result['dataset'],
                'Original PSNR': f"{result['avg_original_psnr']:.2f}",
                'Enhanced PSNR': f"{result['avg_enhanced_psnr']:.2f}",
                'PSNR Improvement': f"{result['avg_psnr_improvement']:.2f}",
                'Original SSIM': f"{result['avg_original_ssim']:.4f}",
                'Enhanced SSIM': f"{result['avg_enhanced_ssim']:.4f}",
                'SSIM Improvement': f"{result['avg_ssim_improvement']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "overall_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(summary_df.to_string(index=False))
        print(f"\n✓ Overall summary saved to: {summary_path}")
        
        print(f"\n🎯 Final Results:")
        print(f"Original EDSR:  PSNR={overall_original_psnr:.2f} dB, SSIM={overall_original_ssim:.4f}")
        print(f"Enhanced EDSR:  PSNR={overall_enhanced_psnr:.2f} dB, SSIM={overall_enhanced_ssim:.4f}")
        print(f"Improvement:    PSNR=+{overall_psnr_improvement:.2f} dB, SSIM=+{overall_ssim_improvement:.4f}")
        
        print(f"\n📁 All results saved to: {output_dir}/")
        print("Generated files:")
        print("- Individual comparison images for each test image")
        print("- comparison_curves.png (overall comparison charts)")
        print("- Set5_results.csv and Set14_results.csv (detailed results)")
        print("- overall_summary.csv (summary table)")

if __name__ == "__main__":
    main()
