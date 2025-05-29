#!/usr/bin/env python3
"""
Standalone evaluation script for Enhanced EDSR
Run this to test your trained enhanced model
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# Import model classes (copy from enhanced_edsr_simple.py or import)
from enhanced_edsr import EAD_EDSR, calculate_psnr, calculate_ssim

def evaluate_enhanced_model():
    """Evaluate the trained enhanced model"""
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configuration (must match training config)
    model_config = {
        'n_resblocks': 10,
        'n_feats': 64,
        'scale': 4,
        'growth_rate': 24,  # This must match training!
    }
    
    # Load model
    model_path = "../models/ead_edsr_x4/best.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run training first: python enhanced_edsr_simple.py")
        return
    
    # Create model with correct config
    model = EAD_EDSR(
        n_resblocks=model_config['n_resblocks'],
        n_feats=model_config['n_feats'],
        scale=model_config['scale'],
        growth_rate=model_config['growth_rate']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded Enhanced EDSR model:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   PSNR: {checkpoint['psnr']:.4f} dB")
    print(f"   Config: {model_config}")
    
    # Test datasets
    test_sets = [
        ("data/test/Set5/HR", "data/test/Set5/LR", "Set5"),
        ("data/test/Set14/HR", "data/test/Set14/LR", "Set14")
    ]
    
    all_results = {}
    to_tensor = transforms.ToTensor()
    
    for test_hr_dir, test_lr_dir, set_name in test_sets:
        if not (os.path.exists(test_hr_dir) and os.path.exists(test_lr_dir)):
            print(f"⚠️  Test set {set_name} not found")
            continue
        
        print(f"\n🔍 Evaluating {set_name}...")
        
        # Get test files
        hr_files = sorted([f for f in os.listdir(test_hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        lr_files = sorted([f for f in os.listdir(test_lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        results = []
        
        with torch.no_grad():
            for hr_file in tqdm(hr_files, desc=f"Processing {set_name}"):
                lr_file = hr_file  # Assume same naming
                
                if lr_file not in lr_files:
                    print(f"⚠️  LR file not found: {lr_file}")
                    continue
                
                # Load images
                hr_path = os.path.join(test_hr_dir, hr_file)
                lr_path = os.path.join(test_lr_dir, lr_file)
                
                hr_img = Image.open(hr_path).convert('RGB')
                lr_img = Image.open(lr_path).convert('RGB')
                
                print(f"   📷 {hr_file}: HR{hr_img.size} LR{lr_img.size}")
                
                # Convert to tensors
                hr_tensor = to_tensor(hr_img).unsqueeze(0).to(device)
                lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
                
                # Generate super-resolution
                sr_tensor = model(lr_tensor)
                
                print(f"      SR shape: {sr_tensor.shape[2:]} HR shape: {hr_tensor.shape[2:]}")
                
                # Handle size mismatch with MPS compatibility
                if hr_tensor.shape != sr_tensor.shape:
                    print(f"      🔧 Resizing HR to match SR...")
                    if device.type == 'mps':
                        hr_tensor = F.interpolate(hr_tensor, size=sr_tensor.shape[2:], mode='bilinear', align_corners=False)
                    else:
                        hr_tensor = F.interpolate(hr_tensor, size=sr_tensor.shape[2:], mode='bicubic', align_corners=False)
                
                # Calculate metrics
                psnr = calculate_psnr(sr_tensor, hr_tensor).item()
                ssim = calculate_ssim(sr_tensor, hr_tensor)
                
                results.append({
                    'filename': hr_file,
                    'psnr': psnr,
                    'ssim': ssim
                })
                
                print(f"      📊 PSNR: {psnr:.3f} dB, SSIM: {ssim:.3f}")
        
        # Calculate averages
        if results:
            avg_psnr = np.mean([r['psnr'] for r in results])
            avg_ssim = np.mean([r['ssim'] for r in results])
            
            all_results[set_name] = {
                'avg_psnr': avg_psnr,
                'avg_ssim': avg_ssim,
                'results': results
            }
            
            print(f"\n📈 {set_name} Summary:")
            print(f"   Average PSNR: {avg_psnr:.4f} dB")
            print(f"   Average SSIM: {avg_ssim:.4f}")
            
            print(f"\n📋 {set_name} Detailed Results:")
            for result in results:
                print(f"   {result['filename']}: PSNR={result['psnr']:.3f}dB, SSIM={result['ssim']:.3f}")
    
    # Final summary
    print(f"\n" + "="*60)
    print("🏆 ENHANCED EDSR EVALUATION SUMMARY")
    print("="*60)
    
    for set_name, data in all_results.items():
        print(f"{set_name:>8}: {data['avg_psnr']:6.3f} dB PSNR, {data['avg_ssim']:.3f} SSIM")
    
    print(f"\n🎯 vs Original EDSR (30.29 dB):")
    for set_name, data in all_results.items():
        improvement = data['avg_psnr'] - 30.29
        print(f"{set_name:>8}: {improvement:+.3f} dB improvement")
    
    print(f"\n✅ Enhanced model evaluation completed!")
    
    return all_results

if __name__ == "__main__":
    evaluate_enhanced_model()
