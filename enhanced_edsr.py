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
import math

warnings.filterwarnings("ignore")

# Set device for M4 MacBook
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class ChannelAttention(nn.Module):
    """Channel Attention Module"""

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
    """Spatial Attention Module"""

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
    """Enhanced Residual Dense Block with Attention"""

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
    """Multi-scale feature extraction module"""

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
    """Progressive upsampling with residual learning"""

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
    """Enhanced Attention-Dense EDSR"""

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


class EnhancedLoss(nn.Module):
    """Enhanced loss with L1 + SSIM (no perceptual to avoid device issues)"""

    def __init__(self, l1_weight=1.0, ssim_weight=0.1):
        super(EnhancedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.l1_loss = nn.L1Loss()

    def ssim_loss(self, x, y):
        """SSIM-based loss"""
        # Simple SSIM approximation using correlation
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))

        return 1 - ssim_map.mean()

    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.l1_weight * l1 + self.ssim_weight * ssim


class SRDataset(Dataset):
    """Enhanced Super-Resolution Dataset with better augmentation"""

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
            lr_candidates = [
                hr_file.replace('.png', 'x4.png'),
                hr_file.replace('.png', '_x4.png'),
                hr_file,
            ]

            for lr_candidate in lr_candidates:
                if lr_candidate in self.lr_files:
                    self.files.append((hr_file, lr_candidate))
                    break

        print(f"Found {len(self.files)} matching HR-LR pairs")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files) * (4 if self.training else 1)  # 4x augmentation during training

    def __getitem__(self, idx):
        if self.training:
            idx = idx % len(self.files)

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
            # Enhanced data augmentation
            h, w = lr_tensor.shape[1:]
            if h >= self.patch_size and w >= self.patch_size:
                top = np.random.randint(0, h - self.patch_size + 1)
                left = np.random.randint(0, w - self.patch_size + 1)

                lr_tensor = lr_tensor[:, top:top + self.patch_size, left:left + self.patch_size]
                hr_tensor = hr_tensor[:, top * self.scale:(top + self.patch_size) * self.scale,
                            left * self.scale:(left + self.patch_size) * self.scale]

            # Random augmentations
            if np.random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [2])
                hr_tensor = torch.flip(hr_tensor, [2])

            if np.random.random() > 0.5:
                lr_tensor = torch.flip(lr_tensor, [1])
                hr_tensor = torch.flip(hr_tensor, [1])

            if np.random.random() > 0.5:
                k = np.random.randint(1, 4)
                lr_tensor = torch.rot90(lr_tensor, k, [1, 2])
                hr_tensor = torch.rot90(hr_tensor, k, [1, 2])

        return lr_tensor, hr_tensor, hr_file


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


class EnhancedTrainer:
    def __init__(self, model, train_loader, val_loader, device, model_name="ead_edsr"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name

        # Create model directory
        self.model_dir = f"models/{model_name}"
        os.makedirs(self.model_dir, exist_ok=True)

        # Enhanced loss function (simple version)
        self.criterion = EnhancedLoss(l1_weight=1.0, ssim_weight=0.1)
        print("✅ Using Enhanced Loss (L1 + SSIM)")

        # Advanced optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-4)

        # Advanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'})

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
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        print(f"Starting enhanced training for {epochs} epochs...")
        print(f"Model will be saved to: {self.model_dir}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
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
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")

        # Plot training curves
        self.plot_training_curves()

        print(f"\nEnhanced training completed!")
        print(f"Best model saved at epoch {self.best_epoch} with PSNR: {self.best_psnr:.4f}")

    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PSNR curve
        ax2.plot(epochs, self.val_psnrs, 'g-', label='Val PSNR', linewidth=2)
        best_epoch_idx = self.best_epoch - 1
        ax2.scatter([self.best_epoch], [self.best_psnr], color='red', s=100, zorder=5, label='Best PSNR')
        ax2.set_title('Validation PSNR Progress', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('PSNR (dB)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # SSIM curve
        ax3.plot(epochs, self.val_ssims, 'm-', label='Val SSIM', linewidth=2)
        ax3.set_title('Validation SSIM Progress', fontsize=14)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('SSIM')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Combined metrics
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, self.val_psnrs, 'g-', linewidth=2, label='PSNR')
        line2 = ax4_twin.plot(epochs, self.val_ssims, 'm-', linewidth=2, label='SSIM')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('PSNR (dB)', color='g')
        ax4_twin.set_ylabel('SSIM', color='m')
        ax4.set_title('PSNR and SSIM Combined', fontsize=14)

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'enhanced_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()


class EnhancedEvaluator:
    def __init__(self, model_path, device):
        self.device = device
        self.model = EAD_EDSR().to(device)

        # Load best model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded enhanced model from epoch {checkpoint['epoch']} with PSNR: {checkpoint['psnr']:.4f}")

    def evaluate_dataset(self, test_dir_hr, test_dir_lr, save_results=True):
        """Evaluate on test dataset with MPS compatibility"""
        print(f"Evaluating enhanced model on test dataset...")

        # Get test files
        hr_files = sorted([f for f in os.listdir(test_dir_hr) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        lr_files = sorted([f for f in os.listdir(test_dir_lr) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        results = []
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        with torch.no_grad():
            for hr_file in tqdm(hr_files, desc="Enhanced Evaluation"):
                lr_file = hr_file

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

                # Handle size mismatch with MPS compatibility
                if hr_tensor.shape != sr_tensor.shape:
                    if self.device.type == 'mps':
                        hr_tensor = F.interpolate(hr_tensor, size=sr_tensor.shape[2:], mode='bilinear',
                                                  align_corners=False)
                    else:
                        hr_tensor = F.interpolate(hr_tensor, size=sr_tensor.shape[2:], mode='bicubic',
                                                  align_corners=False)

                # Calculate metrics
                psnr = calculate_psnr(sr_tensor, hr_tensor).item()
                ssim = calculate_ssim(sr_tensor, hr_tensor)

                results.append({
                    'filename': hr_file,
                    'psnr': psnr,
                    'ssim': ssim
                })

        # Calculate average metrics
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])

        print(f"\nEnhanced Test Results:")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print("\nPer-image enhanced results:")
        for result in results:
            print(f"{result['filename']}: PSNR={result['psnr']:.4f}, SSIM={result['ssim']:.4f}")

        return results, avg_psnr, avg_ssim


def main():
    """Enhanced main training and evaluation pipeline"""

    # Enhanced configuration (optimized for reliability)
    config = {
        'model_name': 'ead_edsr_x4',
        'scale': 4,
        'epochs': 50,
        'batch_size': 4,
        'patch_size': 48,
        'n_resblocks': 10,
        'n_feats': 64,
        'growth_rate': 24,
    }

    print("Enhanced Attention-Dense EDSR System (Simple Loss)")
    print("=" * 60)
    print(f"Enhanced Configuration: {config}")
    print(f"Device: {device}")

    # Data paths
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
        return

    # Create enhanced datasets
    print("\nCreating enhanced datasets...")
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

    # Create enhanced model
    model = EAD_EDSR(n_resblocks=config['n_resblocks'],
                     n_feats=config['n_feats'],
                     scale=config['scale'],
                     growth_rate=config['growth_rate'])

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nEnhanced Model Statistics:")
    print(f"Total parameters: {total_params:,}")

    # Create enhanced trainer and start training
    trainer = EnhancedTrainer(model, train_loader, val_loader, device, config['model_name'])
    trainer.train(epochs=config['epochs'])

    # Evaluate on test sets
    print("\n" + "=" * 60)
    print("ENHANCED EVALUATION ON TEST SETS")
    print("=" * 60)

    evaluator = EnhancedEvaluator(os.path.join(trainer.model_dir, 'best.pth'), device)

    for i, (test_hr_dir, test_lr_dir) in enumerate(test_sets):
        if os.path.exists(test_hr_dir) and os.path.exists(test_lr_dir):
            print(f"\nEvaluating enhanced model on test set {i + 1}: {os.path.basename(test_hr_dir)}")
            results, avg_psnr, avg_ssim = evaluator.evaluate_dataset(test_hr_dir, test_lr_dir)
        else:
            print(f"Test set {i + 1} not found: {test_hr_dir}")


if __name__ == "__main__":
    main()