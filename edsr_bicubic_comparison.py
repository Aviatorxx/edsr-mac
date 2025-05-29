#!/usr/bin/env python3
"""
EDSR vs Bicubic Comprehensive Comparison Test
Separate processing for EDSR and Bicubic to avoid size mismatch issues
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Songti SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
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
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
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
        m_head = [nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size // 2)]

        # define body module
        m_body = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2))

        # define tail module
        m_tail = [
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, n_colors, kernel_size, padding=kernel_size // 2)
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


def rgb2y(img):
    """Convert RGB to Y channel"""
    return 16.0 + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.0


def crop_to_same_size(*images):
    """Crop all images to the same size (minimum common size)"""
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)

    cropped_images = []
    for img in images:
        if len(img.shape) == 3:
            cropped = img[:min_h, :min_w, :]
        else:
            cropped = img[:min_h, :min_w]
        cropped_images.append(cropped)

    return cropped_images


def calculate_psnr_tensor(img1, img2, max_val=1.0):
    """Calculate PSNR between two tensor images (like original edsr_test_only.py)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim_tensor(img1, img2):
    """Calculate SSIM between two tensor images"""
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


def calculate_metrics_bicubic(img1, img2):
    """Calculate PSNR and SSIM for bicubic (using Y channel like bicubic_test.py)"""
    # Convert to Y channel for PSNR calculation
    img1_y = rgb2y(img1).astype(np.float32)
    img2_y = rgb2y(img2).astype(np.float32)

    # Crop to same size if needed
    if img1_y.shape != img2_y.shape:
        img1_y, img2_y = crop_to_same_size(img1_y, img2_y)

    # Calculate PSNR on Y channel (like bicubic_test.py)
    psnr_val = peak_signal_noise_ratio(img1_y, img2_y, data_range=img1_y.max() - img1_y.min())

    # Calculate SSIM on RGB channels
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2.astype(np.float32) / 255.0

    # Crop RGB images to same size if needed
    if img1_norm.shape != img2_norm.shape:
        img1_norm, img2_norm = crop_to_same_size(img1_norm, img2_norm)

    ssim_vals = []
    for i in range(3):  # RGB channels
        ssim_val = structural_similarity(
            img1_norm[:, :, i], img2_norm[:, :, i],
            data_range=1.0
        )
        ssim_vals.append(ssim_val)
    ssim_val = np.mean(ssim_vals)

    return psnr_val, ssim_val


def process_bicubic(hr_path, scale=4):
    """Process bicubic interpolation"""
    hr_img = Image.open(hr_path).convert('RGB')

    # Generate LR by downsampling HR
    lr_img = hr_img.resize((hr_img.width // scale, hr_img.height // scale), Image.BICUBIC)

    # Generate bicubic upsampling
    bicubic_img = lr_img.resize(hr_img.size, Image.BICUBIC)

    # Convert to numpy for metric calculation
    hr_np = np.array(hr_img)
    bicubic_np = np.array(bicubic_img)

    # Calculate metrics
    psnr_val, ssim_val = calculate_metrics_bicubic(hr_np, bicubic_np)

    return {
        'lr_img': lr_img,
        'bicubic_img': bicubic_img,
        'hr_img': hr_img,
        'psnr': psnr_val,
        'ssim': ssim_val
    }


def process_edsr(lr_path, model, device):
    """Process EDSR super-resolution"""
    to_tensor = transforms.ToTensor()

    # Load LR image
    lr_img = Image.open(lr_path).convert('RGB')
    lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

    # Generate EDSR super-resolution
    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    # Convert to PIL Image for visualization (but keep tensor for metrics)
    sr_np = (sr_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    sr_img = Image.fromarray(sr_np)

    return {
        'lr_img': lr_img,
        'sr_img': sr_img,
        'sr_tensor': sr_tensor  # Keep tensor for accurate metric calculation
    }


def calculate_edsr_metrics(hr_path, edsr_result):
    """Calculate metrics between HR and EDSR result using tensors (like edsr_test_only.py)"""
    to_tensor = transforms.ToTensor()

    hr_img = Image.open(hr_path).convert('RGB')
    hr_tensor = to_tensor(hr_img).unsqueeze(0)
    sr_tensor = edsr_result['sr_tensor'].cpu()

    # Handle size mismatch by cropping/resizing like in edsr_test_only.py
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

    # Calculate metrics using tensor method (like edsr_test_only.py)
    psnr_val = calculate_psnr_tensor(sr_tensor, hr_tensor).item()
    ssim_val = calculate_ssim_tensor(sr_tensor, hr_tensor)

    return psnr_val, ssim_val, hr_img


def create_comparison_plot(bicubic_result, edsr_result, hr_img, edsr_psnr, edsr_ssim, filename, output_dir):
    """Create 4-image comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # LR (upsampled with nearest neighbor for visualization)
    lr_display = bicubic_result['lr_img'].resize(hr_img.size, Image.NEAREST)
    axes[0, 0].imshow(lr_display)
    axes[0, 0].set_title('LR (Nearest Upsample)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Bicubic
    axes[0, 1].imshow(bicubic_result['bicubic_img'])
    axes[0, 1].set_title(f'Bicubic\nPSNR: {bicubic_result["psnr"]:.2f} dB\nSSIM: {bicubic_result["ssim"]:.4f}',
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # EDSR
    axes[1, 0].imshow(edsr_result['sr_img'])
    axes[1, 0].set_title(f'EDSR\nPSNR: {edsr_psnr:.2f} dB\nSSIM: {edsr_ssim:.4f}',
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # HR
    axes[1, 1].imshow(hr_img)
    axes[1, 1].set_title('Ground Truth (HR)', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{filename}_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    return save_path


def test_dataset(dataset_name, model, device, output_dir):
    """Test a complete dataset"""
    print(f"\n{'=' * 50}")
    print(f"Testing {dataset_name}")
    print(f"{'=' * 50}")

    hr_dir = f"data/test/{dataset_name}/HR"
    lr_dir = f"data/test/{dataset_name}/LR"

    if not (os.path.exists(hr_dir) and os.path.exists(lr_dir)):
        print(f"Dataset {dataset_name} not found!")
        return None

    # Get all image files
    hr_files = sorted([f for f in os.listdir(hr_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    results = []
    dataset_output_dir = os.path.join(output_dir, dataset_name)

    for hr_file in tqdm(hr_files, desc=f"Processing {dataset_name}"):
        lr_file = hr_file  # Assume same naming convention
        hr_path = os.path.join(hr_dir, hr_file)
        lr_path = os.path.join(lr_dir, lr_file)

        if not os.path.exists(lr_path):
            print(f"Warning: LR file not found for {hr_file}")
            continue

        try:
            # Process bicubic (using HR to generate LR, then bicubic upsampling)
            bicubic_result = process_bicubic(hr_path, scale=4)

            # Process EDSR (using existing LR image)
            edsr_result = process_edsr(lr_path, model, device)

            # Calculate EDSR metrics against HR
            edsr_psnr, edsr_ssim, hr_img = calculate_edsr_metrics(hr_path, edsr_result)

            # Create visualization
            filename = os.path.splitext(hr_file)[0]
            save_path = create_comparison_plot(
                bicubic_result, edsr_result, hr_img,
                edsr_psnr, edsr_ssim, filename, dataset_output_dir
            )

            # Store results
            results.append({
                'filename': hr_file,
                'bicubic_psnr': bicubic_result['psnr'],
                'bicubic_ssim': bicubic_result['ssim'],
                'edsr_psnr': edsr_psnr,
                'edsr_ssim': edsr_ssim,
                'comparison_path': save_path
            })

            print(f"{hr_file}:")
            print(f"  Bicubic - PSNR: {bicubic_result['psnr']:.2f} dB, SSIM: {bicubic_result['ssim']:.4f}")
            print(f"  EDSR    - PSNR: {edsr_psnr:.2f} dB, SSIM: {edsr_ssim:.4f}")
            print(f"  Improvement: +{edsr_psnr - bicubic_result['psnr']:.2f} dB PSNR")

        except Exception as e:
            print(f"Error processing {hr_file}: {str(e)}")
            continue

    if results:
        # Calculate averages
        avg_bicubic_psnr = np.mean([r['bicubic_psnr'] for r in results])
        avg_bicubic_ssim = np.mean([r['bicubic_ssim'] for r in results])
        avg_edsr_psnr = np.mean([r['edsr_psnr'] for r in results])
        avg_edsr_ssim = np.mean([r['edsr_ssim'] for r in results])

        print(f"\n{dataset_name} Summary Results:")
        print(f"{'Method':<10} {'PSNR (dB)':<12} {'SSIM':<8}")
        print("-" * 32)
        print(f"{'Bicubic':<10} {avg_bicubic_psnr:<12.2f} {avg_bicubic_ssim:<8.4f}")
        print(f"{'EDSR':<10} {avg_edsr_psnr:<12.2f} {avg_edsr_ssim:<8.4f}")
        print(f"{'Improvement':<10} {avg_edsr_psnr - avg_bicubic_psnr:<12.2f} {avg_edsr_ssim - avg_bicubic_ssim:<8.4f}")

        # Save results to CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(dataset_output_dir, f"{dataset_name}_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")

        return {
            'dataset': dataset_name,
            'avg_bicubic_psnr': avg_bicubic_psnr,
            'avg_bicubic_ssim': avg_bicubic_ssim,
            'avg_edsr_psnr': avg_edsr_psnr,
            'avg_edsr_ssim': avg_edsr_ssim,
            'individual_results': results
        }

    return None


def main():
    """Main testing function"""
    print("EDSR vs Bicubic Comprehensive Comparison (Separate Processing)")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained EDSR model
    model_path = "models/edsr_x4/best.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please ensure you have trained the EDSR model first.")
        return

    # Create model and load weights
    model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Loaded EDSR model from epoch {checkpoint['epoch']}")
        if 'psnr' in checkpoint:
            print(f"  Training PSNR: {checkpoint['psnr']:.4f} dB")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Output directory
    output_dir = "results/edsr_and_bicubic"
    os.makedirs(output_dir, exist_ok=True)

    # Test datasets
    datasets = ["Set5", "Set14"]
    all_results = []

    for dataset in datasets:
        result = test_dataset(dataset, model, device, output_dir)
        if result:
            all_results.append(result)

    # Create overall summary
    if all_results:
        print(f"\n{'=' * 60}")
        print("OVERALL SUMMARY")
        print(f"{'=' * 60}")

        summary_data = []
        for result in all_results:
            summary_data.append({
                'Dataset': result['dataset'],
                'Bicubic PSNR': f"{result['avg_bicubic_psnr']:.2f}",
                'Bicubic SSIM': f"{result['avg_bicubic_ssim']:.4f}",
                'EDSR PSNR': f"{result['avg_edsr_psnr']:.2f}",
                'EDSR SSIM': f"{result['avg_edsr_ssim']:.4f}",
                'PSNR Gain': f"{result['avg_edsr_psnr'] - result['avg_bicubic_psnr']:.2f}",
                'SSIM Gain': f"{result['avg_edsr_ssim'] - result['avg_bicubic_ssim']:.4f}"
            })

        # Save overall summary
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "overall_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(summary_df.to_string(index=False))
        print(f"\n✓ Overall summary saved to: {summary_path}")
        print(f"✓ All comparison images saved in: {output_dir}")

        # Calculate overall averages
        overall_bicubic_psnr = np.mean([r['avg_bicubic_psnr'] for r in all_results])
        overall_bicubic_ssim = np.mean([r['avg_bicubic_ssim'] for r in all_results])
        overall_edsr_psnr = np.mean([r['avg_edsr_psnr'] for r in all_results])
        overall_edsr_ssim = np.mean([r['avg_edsr_ssim'] for r in all_results])

        print(f"\nOverall Average Performance:")
        print(f"Bicubic: PSNR={overall_bicubic_psnr:.2f} dB, SSIM={overall_bicubic_ssim:.4f}")
        print(f"EDSR:    PSNR={overall_edsr_psnr:.2f} dB, SSIM={overall_edsr_ssim:.4f}")
        print(
            f"Gain:    PSNR=+{overall_edsr_psnr - overall_bicubic_psnr:.2f} dB, SSIM=+{overall_edsr_ssim - overall_bicubic_ssim:.4f}")


if __name__ == "__main__":
    main()