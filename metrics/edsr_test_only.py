#!/usr/bin/env python3
"""
Test-only script for evaluating the trained EDSR model
Moved to metrics directory with adjusted paths
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

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    if len(img1_np.shape) == 4:
        img1_np = img1_np[0]
        img2_np = img2_np[0]
    
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))
    
    ssim_vals = []
    for i in range(img1_np.shape[2]):
        ssim_val = structural_similarity(img1_np[:, :, i], img2_np[:, :, i], data_range=1.0)
        ssim_vals.append(ssim_val)
    
    return np.mean(ssim_vals)

def test_model():
    """Test the trained model"""
    
    # Set device (use CPU to avoid MPS issues during testing)
    device = torch.device("cpu")  # Use CPU for compatibility
    
    # Load model (adjusted path for metrics directory)
    model_path = "../models/edsr_x4/best.pth"
    
    # Debug: Check absolute path
    from pathlib import Path
    model_path_abs = Path(model_path).resolve()
    print(f"Looking for model at: {model_path_abs}")
    
    if not os.path.exists(model_path) and not model_path_abs.exists():
        print(f"Model not found: {model_path}")
        print(f"Absolute path: {model_path_abs}")
        return
    
    print(f"✓ Found model at: {model_path}")
    
    # Create model and load weights
    model = EDSR(n_resblocks=16, n_feats=64, scale=4).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with PSNR: {checkpoint['psnr']:.4f}")
    
    # Test datasets (adjusted paths for metrics directory)
    test_sets = [
        ("../data/test/Set5/HR", "../data/test/Set5/LR", "Set5"),
        ("../data/test/Set14/HR", "../data/test/Set14/LR", "Set14")
    ]
    
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    for test_hr_dir, test_lr_dir, set_name in test_sets:
        if not (os.path.exists(test_hr_dir) and os.path.exists(test_lr_dir)):
            print(f"Test set {set_name} not found")
            continue
            
        print(f"\nEvaluating {set_name}...")
        
        hr_files = sorted([f for f in os.listdir(test_hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        results = []
        
        with torch.no_grad():
            for hr_file in tqdm(hr_files, desc=f"Processing {set_name}"):
                lr_file = hr_file  # Assume same naming
                lr_path = os.path.join(test_lr_dir, lr_file)
                hr_path = os.path.join(test_hr_dir, hr_file)
                
                if not os.path.exists(lr_path):
                    print(f"LR file not found: {lr_file}")
                    continue
                
                # Load images
                hr_img = Image.open(hr_path).convert('RGB')
                lr_img = Image.open(lr_path).convert('RGB')
                
                # Convert to tensors
                hr_tensor = to_tensor(hr_img).unsqueeze(0).to(device)
                lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
                
                # Generate super-resolution
                sr_tensor = model(lr_tensor)
                
                # Handle size mismatch - crop or pad to match
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
                        # Pad SR to match HR (simple approach)
                        sr_tensor = F.interpolate(sr_tensor, size=(h_hr, w_hr), mode='bilinear', align_corners=False)
                
                # Calculate metrics
                psnr = calculate_psnr(sr_tensor, hr_tensor).item()
                ssim = calculate_ssim(sr_tensor, hr_tensor)
                
                results.append({
                    'filename': hr_file,
                    'psnr': psnr,
                    'ssim': ssim
                })
                
                print(f"{hr_file}: PSNR={psnr:.4f}, SSIM={ssim:.4f}")
        
        # Calculate averages
        if results:
            avg_psnr = np.mean([r['psnr'] for r in results])
            avg_ssim = np.mean([r['ssim'] for r in results])
            
            print(f"\n{set_name} Results:")
            print(f"Average PSNR: {avg_psnr:.4f} dB")
            print(f"Average SSIM: {avg_ssim:.4f}")
        else:
            print(f"No results for {set_name}")

if __name__ == "__main__":
    test_model()
