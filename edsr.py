import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set device for M4 MacBook
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class ResBlock(nn.Module):
    """Residual Block for EDSR"""
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
    """Upsampling module"""
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
    """Enhanced Deep Super-Resolution Network"""
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, n_colors=3, res_scale=1):
        super(EDSR, self).__init__()
        
        kernel_size = 3
        act = nn.ReLU(True)
        
        # Define head module
        m_head = [nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size//2)]
        
        # Define body module
        m_body = [
            ResBlock(n_feats, kernel_size, res_scale=res_scale) 
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
        
        # Define tail module
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

class SRDataset(Dataset):
    """Super-Resolution Dataset"""
    def __init__(self, hr_dir, lr_dir, patch_size=48, scale=4, training=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.training = training
        
        # Get all image files
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.hr_files)} HR images and {len(self.lr_files)} LR images")
        
        # Ensure we have matching pairs
        self.files = []
        for hr_file in self.hr_files:
            # Try different naming conventions
            lr_candidates = [
                hr_file.replace('.png', 'x4.png'),
                hr_file.replace('.png', '_x4.png'),
                hr_file,
                hr_file.replace('.png', '.png')
            ]
            
            for lr_candidate in lr_candidates:
                if lr_candidate in self.lr_files:
                    self.files.append((hr_file, lr_candidate))
                    break
        
        print(f"Found {len(self.files)} matching HR-LR pairs")
        
        self.to_tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        hr_file, lr_file = self.files[idx]
        
        # Load images
        hr_path = os.path.join(self.hr_dir, hr_file)
        lr_path = os.path.join(self.lr_dir, lr_file)
        
        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')
        
        # Convert to tensors
        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)
        
        if self.training:
            # Random crop for training
            h, w = lr_tensor.shape[1:]
            if h >= self.patch_size and w >= self.patch_size:
                top = np.random.randint(0, h - self.patch_size + 1)
                left = np.random.randint(0, w - self.patch_size + 1)
                
                lr_tensor = lr_tensor[:, top:top+self.patch_size, left:left+self.patch_size]
                hr_tensor = hr_tensor[:, top*self.scale:(top+self.patch_size)*self.scale, 
                                    left*self.scale:(left+self.patch_size)*self.scale]
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [2])
                hr_tensor = torch.flip(hr_tensor, [2])
        
        return lr_tensor, hr_tensor, hr_file

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # Convert tensors to numpy arrays
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(img1_np.shape) == 4:
        img1_np = img1_np[0]
        img2_np = img2_np[0]
    
    # Convert from CHW to HWC
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))
    
    # Calculate SSIM for each channel and take the mean
    ssim_vals = []
    for i in range(img1_np.shape[2]):
        ssim_val = structural_similarity(img1_np[:, :, i], img2_np[:, :, i], data_range=1.0)
        ssim_vals.append(ssim_val)
    
    return np.mean(ssim_vals)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, model_name="edsr"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        
        # Create model directory
        self.model_dir = f"models/{model_name}"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_psnrs = []
        self.val_ssims = []
        
        self.best_psnr = 0
        self.best_epoch = 0
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for lr_batch, hr_batch, _ in pbar:
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            sr_batch = self.model(lr_batch)
            loss = self.criterion(sr_batch, hr_batch)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for lr_batch, hr_batch, _ in tqdm(self.val_loader, desc="Validation"):
                lr_batch = lr_batch.to(self.device)
                hr_batch = hr_batch.to(self.device)
                
                sr_batch = self.model(lr_batch)
                loss = self.criterion(sr_batch, hr_batch)
                
                # Calculate metrics
                psnr = calculate_psnr(sr_batch, hr_batch)
                ssim = calculate_ssim(sr_batch[0], hr_batch[0])
                
                total_loss += loss.item()
                total_psnr += psnr.item()
                total_ssim += ssim
        
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def save_model(self, epoch, psnr, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'psnr': psnr,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_psnrs': self.val_psnrs,
            'val_ssims': self.val_ssims,
        }
        
        # Save latest model
        torch.save(checkpoint, os.path.join(self.model_dir, 'latest.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.model_dir, 'best.pth'))
    
    def train(self, epochs=50):
        print(f"Starting training for {epochs} epochs...")
        print(f"Model will be saved to: {self.model_dir}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_psnr, val_ssim = self.validate()
            self.val_losses.append(val_loss)
            self.val_psnrs.append(val_psnr)
            self.val_ssims.append(val_ssim)
            
            # Update learning rate
            self.scheduler.step()
            
            # Check if best model
            is_best = val_psnr > self.best_psnr
            if is_best:
                self.best_psnr = val_psnr
                self.best_epoch = epoch + 1
            
            # Save model
            self.save_model(epoch + 1, val_psnr, is_best)
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
            print(f"Best PSNR: {self.best_psnr:.4f} (Epoch {self.best_epoch})")
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"\nTraining completed!")
        print(f"Best model saved at epoch {self.best_epoch} with PSNR: {self.best_psnr:.4f}")
        
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # PSNR curve
        ax2.plot(epochs, self.val_psnrs, 'g-', label='Val PSNR')
        ax2.set_title('Validation PSNR')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.legend()
        ax2.grid(True)
        
        # SSIM curve
        ax3.plot(epochs, self.val_ssims, 'm-', label='Val SSIM')
        ax3.set_title('Validation SSIM')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('SSIM')
        ax3.legend()
        ax3.grid(True)
        
        # Combined metrics
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, self.val_psnrs, 'g-', label='PSNR')
        line2 = ax4_twin.plot(epochs, self.val_ssims, 'm-', label='SSIM')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('PSNR (dB)', color='g')
        ax4_twin.set_ylabel('SSIM', color='m')
        ax4.set_title('PSNR and SSIM Progress')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()

class Evaluator:
    def __init__(self, model_path, device):
        self.device = device
        self.model = EDSR().to(device)
        
        # Load best model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from epoch {checkpoint['epoch']} with PSNR: {checkpoint['psnr']:.4f}")
    
    def evaluate_dataset(self, test_dir_hr, test_dir_lr, save_results=True):
        """Evaluate on test dataset"""
        print(f"Evaluating on test dataset...")
        print(f"HR dir: {test_dir_hr}")
        print(f"LR dir: {test_dir_lr}")
        
        # Get test files
        hr_files = sorted([f for f in os.listdir(test_dir_hr) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        lr_files = sorted([f for f in os.listdir(test_dir_lr) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        results = []
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()
        
        with torch.no_grad():
            for hr_file in tqdm(hr_files, desc="Evaluating"):
                # Find corresponding LR file
                lr_file = hr_file  # Assume same name for Set5/Set14
                
                if lr_file not in lr_files:
                    print(f"Warning: No LR file found for {hr_file}")
                    continue
                
                # Load images
                hr_path = os.path.join(test_dir_hr, hr_file)
                lr_path = os.path.join(test_dir_lr, lr_file)
                
                hr_img = Image.open(hr_path).convert('RGB')
                lr_img = Image.open(lr_path).convert('RGB')
                
                # Convert to tensors
                hr_tensor = to_tensor(hr_img).unsqueeze(0).to(self.device)
                lr_tensor = to_tensor(lr_img).unsqueeze(0).to(self.device)
                
                # Generate super-resolution
                sr_tensor = self.model(lr_tensor)
                
                # Resize HR to match SR if needed
                if hr_tensor.shape != sr_tensor.shape:
                    hr_tensor = F.interpolate(hr_tensor, size=sr_tensor.shape[2:], mode='bicubic', align_corners=False)
                
                # Calculate metrics
                psnr = calculate_psnr(sr_tensor, hr_tensor).item()
                ssim = calculate_ssim(sr_tensor, hr_tensor)
                
                results.append({
                    'filename': hr_file,
                    'psnr': psnr,
                    'ssim': ssim
                })
                
                # Save result images
                if save_results:
                    result_dir = os.path.join(self.model.training.model_dir if hasattr(self.model, 'training') else 'results', 
                                            'test_results')
                    os.makedirs(result_dir, exist_ok=True)
                    
                    # Save SR image
                    sr_img = to_pil(sr_tensor.squeeze(0).cpu())
                    sr_img.save(os.path.join(result_dir, f"sr_{hr_file}"))
        
        # Calculate average metrics
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        
        print(f"\nTest Results:")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("\nPer-image results:")
        for result in results:
            print(f"{result['filename']}: PSNR={result['psnr']:.4f}, SSIM={result['ssim']:.4f}")
        
        return results, avg_psnr, avg_ssim

def main():
    """Main training and evaluation pipeline"""
    
    # Configuration
    config = {
        'model_name': 'edsr_x4',
        'scale': 4,
        'epochs': 50,
        'batch_size': 8,
        'patch_size': 48,
        'n_resblocks': 16,
        'n_feats': 64,
    }
    
    print("Image Super-Resolution System")
    print("=" * 50)
    print(f"Configuration: {config}")
    print(f"Device: {device}")
    
    # Data paths - adjust these to your actual paths
    train_hr_dir = "data/train/HR"
    train_lr_dir = "data/train/LR/X4"
    
    # Test datasets
    test_sets = [
        ("data/test/Set5/HR", "data/test/Set5/LR"),
        ("data/test/Set14/HR", "data/test/Set14/LR")
    ]
    
    # Check if data directories exist
    if not os.path.exists(train_hr_dir):
        print(f"Warning: Training HR directory not found: {train_hr_dir}")
        print("Please adjust the data paths in the main() function")
        return
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SRDataset(train_hr_dir, train_lr_dir, 
                             patch_size=config['patch_size'], 
                             scale=config['scale'], 
                             training=True)
    
    # Split training data for validation (80-20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    
    # Create model
    model = EDSR(n_resblocks=config['n_resblocks'], 
                 n_feats=config['n_feats'], 
                 scale=config['scale'])
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, device, config['model_name'])
    trainer.train(epochs=config['epochs'])
    
    # Evaluate on test sets
    print("\n" + "="*50)
    print("EVALUATION ON TEST SETS")
    print("="*50)
    
    evaluator = Evaluator(os.path.join(trainer.model_dir, 'best.pth'), device)
    
    for i, (test_hr_dir, test_lr_dir) in enumerate(test_sets):
        if os.path.exists(test_hr_dir) and os.path.exists(test_lr_dir):
            print(f"\nEvaluating on test set {i+1}: {os.path.basename(test_hr_dir)}")
            results, avg_psnr, avg_ssim = evaluator.evaluate_dataset(test_hr_dir, test_lr_dir)
        else:
            print(f"Test set {i+1} not found: {test_hr_dir}")

if __name__ == "__main__":
    main()
