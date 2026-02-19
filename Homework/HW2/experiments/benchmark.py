#!/usr/bin/env python3
"""
Benchmark всех моделей и сбор метрик.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from baseline_model import BaselineModel
from quantization import BinaryModel
from pruning import PrunedModel
from inference_simulator import ArduinoInferenceSimulator
from train_baseline import generate_synthetic_data
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model_accuracy(model, X_test, y_test, device, batch_size=32):
    """Оценить accuracy модели на тестовых данных."""
    model.eval()
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            if hasattr(model, 'forward') and 'quant_temp' in model.forward.__code__.co_varnames:
                outputs = model(batch_x, quant_temp=1.0)
            else:
                outputs = model(batch_x)
            
            predictions = outputs.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    return correct / total if total > 0 else 0.0


def benchmark_all_models(args):
    """Провести benchmark всех моделей."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("[1] Loading models...")
    
    baseline = BaselineModel(
        input_size=args.input_size,
        hidden_sizes=args.baseline_hidden,
        num_classes=args.num_classes
    )
    baseline.load_state_dict(torch.load(args.baseline_path))
    baseline = baseline.to(device)
    print(f"  ✓ Baseline loaded")
    
    pruned = PrunedModel(
        input_size=args.input_size,
        hidden_sizes=args.pruned_hidden,
        num_classes=args.num_classes
    )
    pruned.load_state_dict(torch.load(args.pruned_path), strict=False)
    pruned = pruned.to(device)
    print(f"  ✓ Pruned loaded")
    
    binary = BinaryModel(
        input_size=args.input_size,
        hidden_sizes=args.pruned_hidden,
        num_classes=args.num_classes
    )
    binary.load_state_dict(torch.load(args.binary_path), strict=False)
    binary = binary.to(device)
    print(f"  ✓ Binary loaded")
    
    print("\n[2] Generating test data...")
    X, y = generate_synthetic_data(
        num_samples=args.num_samples,
        input_size=args.input_size,
        num_classes=args.num_classes,
        seed=args.seed
    )
    
    split_idx = int(0.8 * len(X))
    X_test, y_test = X[split_idx:], y[split_idx:]
    X_test, y_test = X_test.to(device), y_test.to(device)
    print(f"  Test samples: {len(X_test)}")
    
    print("\n[3] Evaluating accuracy...")
    
    baseline_acc = evaluate_model_accuracy(baseline, X_test, y_test, device)
    print(f"  Baseline: {baseline_acc:.4f}")
    
    pruned_acc = evaluate_model_accuracy(pruned, X_test, y_test, device)
    print(f"  Pruned: {pruned_acc:.4f}")
    
    binary_acc = evaluate_model_accuracy(binary, X_test, y_test, device)
    print(f"  Binary: {binary_acc:.4f}")
    
    print("\n[4] Simulating Arduino inference...")
    simulator = ArduinoInferenceSimulator()
    
    baseline_metrics = simulator.measure_inference(baseline, args.input_size, bits_per_weight=2)
    print(f"  ✓ Baseline simulated")
    
    pruned_metrics = simulator.measure_inference(pruned, args.input_size, bits_per_weight=2)
    print(f"  ✓ Pruned simulated")
    
    binary_metrics = simulator.measure_inference(binary, args.input_size, bits_per_weight=1)
    print(f"  ✓ Binary simulated")
    
    print("\n[5] Collecting results...")
    
    results = {
        'Model': ['Baseline (2-bit)', 'Pruned (2-bit)', 'Binary (1-bit)'],
        'Architecture': [
            baseline.get_architecture_str(),
            pruned.get_architecture_str(),
            binary.get_architecture_str()
        ],
        'Parameters': [
            baseline.count_parameters(),
            pruned.count_parameters(),
            binary.count_parameters()
        ],
        'Flash (KB)': [
            baseline_metrics['flash_kb'],
            pruned_metrics['flash_kb'],
            binary_metrics['flash_kb']
        ],
        'SRAM (bytes)': [
            baseline_metrics['sram_bytes'],
            pruned_metrics['sram_bytes'],
            binary_metrics['sram_bytes']
        ],
        'Latency (ms)': [
            baseline_metrics['latency_ms'],
            pruned_metrics['latency_ms'],
            binary_metrics['latency_ms']
        ],
        'Throughput (inf/s)': [
            baseline_metrics['throughput_per_sec'],
            pruned_metrics['throughput_per_sec'],
            binary_metrics['throughput_per_sec']
        ],
        'Accuracy': [baseline_acc, pruned_acc, binary_acc],
        'Fits_Arduino': [
            baseline_metrics['fits_on_arduino'],
            pruned_metrics['fits_on_arduino'],
            binary_metrics['fits_on_arduino']
        ]
    }
    
    df = pd.DataFrame(results)
    
    df['Speedup'] = df['Latency (ms)'].iloc[0] / df['Latency (ms)']
    df['Compression'] = df['Parameters'].iloc[0] / df['Parameters']
    df['Accuracy_Drop'] = df['Accuracy'].iloc[0] - df['Accuracy']
    
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to: {args.output}")
    
    print("\n[6] Summary:")
    print(f"  Best compression: {df['Compression'].max():.2f}x (Binary)")
    print(f"  Best speedup: {df['Speedup'].max():.2f}x (Binary)")
    print(f"  Accuracy drop: {df['Accuracy_Drop'].iloc[2]:.4f} (Binary)")
    print(f"  Models fitting on Arduino: {df['Fits_Arduino'].sum()}/3")
    
    baseline_efficiency = baseline_acc / (baseline_metrics['latency_ms'] * baseline_metrics['flash_kb'])
    binary_efficiency = binary_acc / (binary_metrics['latency_ms'] * binary_metrics['flash_kb'])
    
    print(f"\n  Efficiency improvement: {binary_efficiency / baseline_efficiency:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Benchmark all models')
    
    parser.add_argument('--baseline-path', type=str, default='../results/baseline_model.pt')
    parser.add_argument('--pruned-path', type=str, default='../results/pruned_model.pt')
    parser.add_argument('--binary-path', type=str, default='../results/binary_model.pt')
    
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--baseline-hidden', type=int, nargs='+', default=[256, 192, 128])
    parser.add_argument('--pruned-hidden', type=int, nargs='+', default=[128, 96, 64])
    parser.add_argument('--num-classes', type=int, default=40)
    
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--output', type=str, default='../results/metrics.csv')
    
    args = parser.parse_args()
    
    benchmark_all_models(args)


if __name__ == '__main__':
    main()
