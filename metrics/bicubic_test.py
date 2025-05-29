import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

plt.rc('font', family='Songti SC', size=13)

def rgb2y(img):
    return 16.0 + (65.738*img[:,:,0] + 129.057*img[:,:,1] + 25.064*img[:,:,2]) / 256.0

def eval_and_visualize(hr_path, scale, out_dir):
    hr = Image.open(hr_path).convert('RGB')
    # 方法一：从 HR 生成真正的 LR
    lr = hr.resize((hr.width//scale, hr.height//scale), Image.BICUBIC)
    up = lr.resize(hr.size, Image.BICUBIC)

    # 计算 PSNR（Y 通道）
    hr_np = np.array(hr)
    up_np = np.array(up)
    hr_y = rgb2y(hr_np).astype(np.float32)
    up_y = rgb2y(up_np).astype(np.float32)
    p = psnr(hr_y, up_y, data_range=hr_y.max()-hr_y.min())

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lr.resize(hr.size, Image.NEAREST))
    axes[0].set_title('LR Nearest Upsample', fontsize=13)
    axes[1].imshow(up)
    axes[1].set_title(f'Bicubic\nPSNR={p:.2f} dB', fontsize=13)
    axes[2].imshow(hr)
    axes[2].set_title('HR', fontsize=13)
    for ax in axes:
        ax.axis('off')

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(hr_path))[0]
    fig.savefig(os.path.join(out_dir, f"{name}_compare.png"), bbox_inches='tight')
    plt.close(fig)

    return p

def eval_dataset(hr_dir, scale, out_dir):
    psnrs = []
    for fn in sorted(os.listdir(hr_dir)):
        if not fn.lower().endswith(('png','jpg','jpeg')): continue
        p = eval_and_visualize(os.path.join(hr_dir, fn), scale, out_dir)
        psnrs.append(p)
    print(f"平均 PSNR = {np.mean(psnrs):.2f} dB")

if __name__ == "__main__":
    for dataset in ["Set5", "Set14"]:
        print(f"===== {dataset} =====")
        eval_dataset(f"../data/test/{dataset}/HR", scale=4, out_dir=f"../results/{dataset}")
