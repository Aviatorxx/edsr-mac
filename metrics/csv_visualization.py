#!/usr/bin/env python3
"""
CSV Results Visualization Generator
Generate charts from EDSR vs Bicubic comparison results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Songti SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ResultsVisualizer:
    def __init__(self, results_dir="results/edsr_and_bicubic"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self.load_data()

    def load_data(self):
        """Load all CSV files"""
        try:
            # Load individual dataset results
            self.set5_data = pd.read_csv(self.results_dir / "Set5" / "Set5_results.csv")
            self.set14_data = pd.read_csv(self.results_dir / "Set14" / "Set14_results.csv")

            # Load overall summary
            self.summary_data = pd.read_csv(self.results_dir / "overall_summary.csv")

            print("✓ Successfully loaded all CSV files")
            print(f"  Set5: {len(self.set5_data)} images")
            print(f"  Set14: {len(self.set14_data)} images")

        except FileNotFoundError as e:
            print(f"Error: Could not find CSV files. Please run the comparison script first.")
            print(f"Missing file: {e}")
            return False

        return True

    def plot_summary_comparison(self):
        """Plot overall summary comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Extract numeric values
        datasets = self.summary_data['Dataset'].values
        bicubic_psnr = [float(x) for x in self.summary_data['Bicubic PSNR'].values]
        edsr_psnr = [float(x) for x in self.summary_data['EDSR PSNR'].values]
        bicubic_ssim = [float(x) for x in self.summary_data['Bicubic SSIM'].values]
        edsr_ssim = [float(x) for x in self.summary_data['EDSR SSIM'].values]

        # PSNR comparison
        x = np.arange(len(datasets))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, bicubic_psnr, width, label='Bicubic', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width / 2, edsr_psnr, width, label='EDSR', color='lightcoral', alpha=0.8)

        ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax1.set_title('PSNR Comparison by Dataset', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

        # SSIM comparison
        bars3 = ax2.bar(x - width / 2, bicubic_ssim, width, label='Bicubic', color='lightgreen', alpha=0.8)
        bars4 = ax2.bar(x + width / 2, edsr_ssim, width, label='EDSR', color='orange', alpha=0.8)

        ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.set_title('SSIM Comparison by Dataset', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / "summary_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Summary comparison saved to: {save_path}")

    def plot_detailed_results(self, dataset_name, data):
        """Plot detailed results for a specific dataset"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Extract image names (remove file extensions for cleaner display)
        image_names = [os.path.splitext(name)[0] for name in data['filename']]

        # PSNR comparison
        ax1.plot(image_names, data['bicubic_psnr'], 'o-', label='Bicubic', linewidth=2, markersize=8, color='skyblue')
        ax1.plot(image_names, data['edsr_psnr'], 's-', label='EDSR', linewidth=2, markersize=8, color='lightcoral')
        ax1.set_title(f'{dataset_name} - PSNR Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # SSIM comparison
        ax2.plot(image_names, data['bicubic_ssim'], 'o-', label='Bicubic', linewidth=2, markersize=8,
                 color='lightgreen')
        ax2.plot(image_names, data['edsr_ssim'], 's-', label='EDSR', linewidth=2, markersize=8, color='orange')
        ax2.set_title(f'{dataset_name} - SSIM Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # PSNR improvement bar chart
        psnr_improvement = data['edsr_psnr'] - data['bicubic_psnr']
        bars1 = ax3.bar(image_names, psnr_improvement, color='gold', alpha=0.7)
        ax3.set_title(f'{dataset_name} - PSNR Improvement (EDSR - Bicubic)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('PSNR Improvement (dB)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels on improvement bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # SSIM improvement bar chart
        ssim_improvement = data['edsr_ssim'] - data['bicubic_ssim']
        bars2 = ax4.bar(image_names, ssim_improvement, color='mediumseagreen', alpha=0.7)
        ax4.set_title(f'{dataset_name} - SSIM Improvement (EDSR - Bicubic)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('SSIM Improvement', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels on improvement bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                     f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / f"{dataset_name.lower()}_detailed.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ {dataset_name} detailed results saved to: {save_path}")

    def plot_performance_radar(self):
        """Create radar chart comparing overall performance"""
        from math import pi

        # Prepare data
        metrics = ['Avg PSNR', 'Max PSNR', 'Min PSNR', 'Avg SSIM', 'Max SSIM', 'Min SSIM']

        # Calculate metrics for both methods
        all_data = pd.concat([self.set5_data, self.set14_data])

        bicubic_values = [
            all_data['bicubic_psnr'].mean(),
            all_data['bicubic_psnr'].max(),
            all_data['bicubic_psnr'].min(),
            all_data['bicubic_ssim'].mean(),
            all_data['bicubic_ssim'].max(),
            all_data['bicubic_ssim'].min()
        ]

        edsr_values = [
            all_data['edsr_psnr'].mean(),
            all_data['edsr_psnr'].max(),
            all_data['edsr_psnr'].min(),
            all_data['edsr_ssim'].mean(),
            all_data['edsr_ssim'].max(),
            all_data['edsr_ssim'].min()
        ]

        # Normalize values for radar chart (0-1 scale)
        psnr_max = max(max(bicubic_values[:3]), max(edsr_values[:3]))
        psnr_min = min(min(bicubic_values[:3]), min(edsr_values[:3]))
        ssim_max = max(max(bicubic_values[3:]), max(edsr_values[3:]))
        ssim_min = min(min(bicubic_values[3:]), min(edsr_values[3:]))

        # Normalize PSNR values
        for i in range(3):
            bicubic_values[i] = (bicubic_values[i] - psnr_min) / (psnr_max - psnr_min)
            edsr_values[i] = (edsr_values[i] - psnr_min) / (psnr_max - psnr_min)

        # Normalize SSIM values
        for i in range(3, 6):
            bicubic_values[i] = (bicubic_values[i] - ssim_min) / (ssim_max - ssim_min)
            edsr_values[i] = (edsr_values[i] - ssim_min) / (ssim_max - ssim_min)

        # Create radar chart
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Plot Bicubic
        bicubic_values += bicubic_values[:1]
        ax.plot(angles, bicubic_values, 'o-', linewidth=2, label='Bicubic', color='skyblue')
        ax.fill(angles, bicubic_values, alpha=0.25, color='skyblue')

        # Plot EDSR
        edsr_values += edsr_values[:1]
        ax.plot(angles, edsr_values, 's-', linewidth=2, label='EDSR', color='lightcoral')
        ax.fill(angles, edsr_values, alpha=0.25, color='lightcoral')

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Radar Chart\n(Normalized Values)', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        save_path = self.output_dir / "performance_radar.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Performance radar chart saved to: {save_path}")

    def create_improvement_heatmap(self):
        """Create heatmap showing improvement across all images"""
        # Combine all data
        set5_improvement = pd.DataFrame({
            'Image': [f"Set5_{os.path.splitext(name)[0]}" for name in self.set5_data['filename']],
            'PSNR_Improvement': self.set5_data['edsr_psnr'] - self.set5_data['bicubic_psnr'],
            'SSIM_Improvement': self.set5_data['edsr_ssim'] - self.set5_data['bicubic_ssim'],
            'Dataset': 'Set5'
        })

        set14_improvement = pd.DataFrame({
            'Image': [f"Set14_{os.path.splitext(name)[0]}" for name in self.set14_data['filename']],
            'PSNR_Improvement': self.set14_data['edsr_psnr'] - self.set14_data['bicubic_psnr'],
            'SSIM_Improvement': self.set14_data['edsr_ssim'] - self.set14_data['bicubic_ssim'],
            'Dataset': 'Set14'
        })

        all_improvement = pd.concat([set5_improvement, set14_improvement])

        # Create heatmap data
        heatmap_data = all_improvement[['PSNR_Improvement', 'SSIM_Improvement']].T
        heatmap_data.columns = all_improvement['Image']

        # Create heatmap
        fig, ax = plt.subplots(figsize=(18, 6))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                    ax=ax, cbar_kws={'label': 'Improvement Value'})

        ax.set_title('EDSR Improvement Heatmap (EDSR - Bicubic)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Images', fontsize=12)
        ax.set_ylabel('Metrics', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        save_path = self.output_dir / "improvement_heatmap.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Improvement heatmap saved to: {save_path}")

    def generate_statistics_table(self):
        """Generate and save detailed statistics table"""
        # Calculate overall statistics
        all_data = pd.concat([self.set5_data, self.set14_data])

        stats = {
            'Method': ['Bicubic', 'EDSR', 'Improvement'],
            'Mean PSNR': [
                f"{all_data['bicubic_psnr'].mean():.2f}",
                f"{all_data['edsr_psnr'].mean():.2f}",
                f"{(all_data['edsr_psnr'] - all_data['bicubic_psnr']).mean():.2f}"
            ],
            'Std PSNR': [
                f"{all_data['bicubic_psnr'].std():.2f}",
                f"{all_data['edsr_psnr'].std():.2f}",
                f"{(all_data['edsr_psnr'] - all_data['bicubic_psnr']).std():.2f}"
            ],
            'Mean SSIM': [
                f"{all_data['bicubic_ssim'].mean():.4f}",
                f"{all_data['edsr_ssim'].mean():.4f}",
                f"{(all_data['edsr_ssim'] - all_data['bicubic_ssim']).mean():.4f}"
            ],
            'Std SSIM': [
                f"{all_data['bicubic_ssim'].std():.4f}",
                f"{all_data['edsr_ssim'].std():.4f}",
                f"{(all_data['edsr_ssim'] - all_data['bicubic_ssim']).std():.4f}"
            ]
        }

        stats_df = pd.DataFrame(stats)

        # Save to CSV
        stats_path = self.output_dir / "detailed_statistics.csv"
        stats_df.to_csv(stats_path, index=False)

        print("\n" + "=" * 60)
        print("DETAILED STATISTICS TABLE")
        print("=" * 60)
        print(stats_df.to_string(index=False))
        print(f"\n✓ Detailed statistics saved to: {stats_path}")

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("🎨 Generating comprehensive visualizations...")
        print("=" * 50)

        # 1. Summary comparison
        print("\n1. Creating summary comparison charts...")
        self.plot_summary_comparison()

        # 2. Detailed results for each dataset
        print("\n2. Creating detailed results charts...")
        self.plot_detailed_results("Set5", self.set5_data)
        self.plot_detailed_results("Set14", self.set14_data)

        # 3. Performance radar chart
        print("\n3. Creating performance radar chart...")
        self.plot_performance_radar()

        # 4. Improvement heatmap
        print("\n4. Creating improvement heatmap...")
        self.create_improvement_heatmap()

        # 5. Statistics table
        print("\n5. Generating statistics table...")
        self.generate_statistics_table()

        print(f"\n🎉 All visualizations completed!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("- summary_comparison.png")
        print("- set5_detailed.png")
        print("- set14_detailed.png")
        print("- performance_radar.png")
        print("- improvement_heatmap.png")
        print("- detailed_statistics.csv")


def main():
    """Main function"""
    print("CSV Results Visualization Generator")
    print("=" * 50)

    # Create visualizer
    visualizer = ResultsVisualizer()

    # Generate all visualizations
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()