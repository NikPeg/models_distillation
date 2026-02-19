#!/usr/bin/env python3
"""
Визуализация результатов benchmark на реальных данных.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_results(filepath):
    """Загрузить результаты из JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(results, output_path):
    """График сравнения accuracy."""
    models = [r['name'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_params_comparison(results, output_path):
    """График сравнения числа параметров."""
    models = [r['name'] for r in results]
    params = [r['parameters'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(models, params, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, p in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2000,
                f'{p:,}',
                ha='center', va='bottom', fontsize=11)
    
    ax.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(params) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_performance_comparison(results, output_path):
    """График сравнения производительности."""
    models = [r['name'] for r in results]
    flash = [r['performance']['flash_kb'] for r in results]
    latency = [r['performance']['latency_ms'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars1 = ax1.bar(models, flash, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=32, color='red', linestyle='--', linewidth=2, label='Arduino Flash Limit (32 KB)')
    
    for bar, f in zip(bars1, flash):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{f:.1f} KB',
                ha='center', va='bottom', fontsize=11)
    
    ax1.set_ylabel('Flash Usage (KB)', fontsize=12, fontweight='bold')
    ax1.set_title('Flash Memory Usage', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(flash) * 1.2)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    bars2 = ax2.bar(models, latency, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, lat in zip(bars2, latency):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{lat:.0f} ms',
                ha='center', va='bottom', fontsize=11)
    
    ax2.set_ylabel('Inference Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Inference Speed', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(latency) * 1.15)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_tradeoff(results, output_path):
    """График trade-off между качеством и размером."""
    models = [r['name'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    params_ratio = [r['parameters'] / results[0]['parameters'] * 100 for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    sizes = [300, 250, 250]
    
    for i, (model, acc, ratio, color, size) in enumerate(zip(models, accuracies, params_ratio, colors, sizes)):
        ax.scatter(ratio, acc, s=size, c=color, alpha=0.7, edgecolors='black', linewidths=2, label=model)
        ax.annotate(f'{model}\n{acc:.1f}% acc\n{ratio:.0f}% params',
                   (ratio, acc),
                   xytext=(15, 15) if i == 0 else (-15, -40) if i == 1 else (15, -40),
                   textcoords='offset points',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))
    
    ax.set_xlabel('Model Size (% of Baseline)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Quality vs Size Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    ax.set_xlim(0, 110)
    ax.set_ylim(0, max(accuracies) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_combined_metrics(results, output_path):
    """Комбинированный график всех метрик."""
    models = [r['name'] for r in results]
    
    baseline = results[0]
    
    metrics = {
        'Accuracy': [r['accuracy'] / baseline['accuracy'] * 100 for r in results],
        'Parameters': [r['parameters'] / baseline['parameters'] * 100 for r in results],
        'Flash': [r['performance']['flash_kb'] / baseline['performance']['flash_kb'] * 100 for r in results],
        'Latency': [r['performance']['latency_ms'] / baseline['performance']['latency_ms'] * 100 for r in results]
    }
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, (metric, values) in enumerate(metrics.items()):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{val:.0f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('% of Baseline', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Metrics Comparison (Baseline = 100%)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 130)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--input', type=str, default='../results/benchmark_real.json',
                       help='Path to benchmark results JSON')
    parser.add_argument('--output-dir', type=str, default='../results/plots',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading results...")
    results = load_results(args.input)
    
    print("\nGenerating plots...")
    
    plot_accuracy_comparison(results, f'{args.output_dir}/accuracy_real.png')
    plot_params_comparison(results, f'{args.output_dir}/parameters_real.png')
    plot_performance_comparison(results, f'{args.output_dir}/performance_real.png')
    plot_tradeoff(results, f'{args.output_dir}/tradeoff_real.png')
    plot_combined_metrics(results, f'{args.output_dir}/normalized_real.png')
    
    print("\n✓ All plots generated!")


if __name__ == '__main__':
    main()
