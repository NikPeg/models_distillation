#!/usr/bin/env python3
"""
Визуализация результатов экспериментов.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_results(csv_path, output_dir):
    """Создать графики сравнения моделей."""
    df = pd.read_csv(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Comparison', fontsize=16)
    
    models = df['Model'].values
    
    axes[0, 0].bar(models, df['Parameters'] / 1000)
    axes[0, 0].set_title('Parameters (thousands)')
    axes[0, 0].set_ylabel('Parameters (K)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(models, df['Flash (KB)'])
    axes[0, 1].axhline(y=32, color='r', linestyle='--', label='Arduino Flash (32 KB)')
    axes[0, 1].set_title('Flash Usage')
    axes[0, 1].set_ylabel('Flash (KB)')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[1, 0].bar(models, df['Latency (ms)'])
    axes[1, 0].set_title('Inference Latency')
    axes[1, 0].set_ylabel('Latency (ms)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(models, df['Accuracy'])
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/comparison.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    speedup = df['Speedup'].values
    accuracy_drop = df['Accuracy_Drop'].values * 100
    
    colors = ['blue', 'green', 'orange']
    for i, model in enumerate(models):
        ax.scatter(speedup[i], accuracy_drop[i], s=200, c=colors[i], alpha=0.7, label=model)
    
    ax.set_xlabel('Speedup (x)', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Speed vs Quality Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tradeoff.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/tradeoff.png")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    ax.bar(x - width, df['Parameters'] / df['Parameters'].iloc[0], width, label='Parameters', alpha=0.8)
    ax.bar(x, df['Flash (KB)'] / df['Flash (KB)'].iloc[0], width, label='Flash Size', alpha=0.8)
    ax.bar(x + width, df['Latency (ms)'] / df['Latency (ms)'].iloc[0], width, label='Latency', alpha=0.8)
    
    ax.set_ylabel('Normalized (relative to baseline)')
    ax.set_title('Compression and Speedup')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'normalized.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/normalized.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--input', type=str, default='../results/metrics.csv',
                       help='Input CSV file with metrics')
    parser.add_argument('--output-dir', type=str, default='../results/plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    plot_results(args.input, args.output_dir)
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()
