#!/usr/bin/env python3
"""
Benchmark оптимизированных моделей на реальных данных.
Сравнение baseline, pruned и binary по метрикам качества и производительности.
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
import numpy as np
import json
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from baseline_model import BaselineModel
from data_loader import create_dataloaders, TrigramEncoder, ContextEncoder
from utils import accuracy
from inference_simulator import ArduinoInferenceSimulator


def evaluate_model(model, dataloader, criterion, device):
    """Оценка модели на данных."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            total_acc += accuracy(outputs, batch_y)
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def test_inference_examples(model, charset, device, num_samples=10):
    """Протестировать inference на примерах."""
    query_encoder = TrigramEncoder(num_buckets=128)
    context_encoder = ContextEncoder(num_buckets=128)
    
    test_queries = [
        "hello", "hi", "hey",
        "how are you", "how r u",
        "are you a robot", "r u a bot",
        "what is your name",
        "bye", "see ya",
    ]
    
    idx_to_char = {i: c for i, c in enumerate(charset)}
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for query in test_queries[:num_samples]:
            query_vec = query_encoder.encode(query)
            context_vec = np.zeros(128, dtype=np.float32)
            input_vec = np.concatenate([query_vec, context_vec])
            
            x = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            
            top1_prob, top1_idx = torch.max(probs[0], dim=0)
            predicted_char = idx_to_char[top1_idx.item()]
            
            results.append({
                'query': query,
                'predicted': predicted_char,
                'confidence': top1_prob.item()
            })
    
    return results


def measure_performance(model, device):
    """Измерить производительность модели."""
    simulator = ArduinoInferenceSimulator()
    
    stats = simulator.measure_inference(
        model, 
        input_size=256,
        bits_per_weight=2
    )
    
    return stats


def benchmark_model(model_path: str, data_loader, criterion, device, charset, name: str):
    """Полный benchmark одной модели."""
    checkpoint = torch.load(model_path)
    
    model = BaselineModel(
        input_size=checkpoint['architecture']['input_size'],
        hidden_sizes=checkpoint['architecture']['hidden_sizes'],
        num_classes=checkpoint['architecture']['num_classes']
    )
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.to(device)
    
    val_loss, val_acc = evaluate_model(model, data_loader, criterion, device)
    
    inference_examples = test_inference_examples(model, charset, device, num_samples=5)
    
    perf_stats = measure_performance(model, device)
    
    results = {
        'name': name,
        'accuracy': val_acc,
        'loss': val_loss,
        'parameters': model.count_parameters(),
        'architecture': f"{model.input_size}→" + "→".join(map(str, model.hidden_sizes)) + f"→{model.num_classes}",
        'inference_examples': inference_examples,
        'performance': perf_stats
    }
    
    return results


def print_results(results_list):
    """Красиво вывести результаты."""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    baseline = results_list[0]
    
    for result in results_list:
        print(f"\n{result['name']}:")
        print(f"  Architecture: {result['architecture']}")
        print(f"  Parameters:   {result['parameters']:,}")
        print(f"  Accuracy:     {result['accuracy']:.4f}")
        print(f"  Loss:         {result['loss']:.4f}")
        
        if result['name'] != 'Baseline':
            acc_change = (result['accuracy'] - baseline['accuracy']) * 100
            param_ratio = result['parameters'] / baseline['parameters'] * 100
            print(f"  Δ Accuracy:   {acc_change:+.2f}%")
            print(f"  Δ Parameters: {param_ratio:.1f}% of baseline")
        
        print(f"\n  Performance:")
        perf = result['performance']
        print(f"    Flash:  {perf['flash_bytes']:,} bytes ({perf['flash_usage_percent']:.1f}%)")
        print(f"    SRAM:   {perf['sram_bytes']:,} bytes ({perf['sram_usage_percent']:.1f}%)")
        print(f"    Latency: {perf['latency_ms']:.2f} ms")
        
        print(f"\n  Example Predictions:")
        for ex in result['inference_examples'][:3]:
            print(f"    '{ex['query']}' → '{ex['predicted']}' ({ex['confidence']:.3f})")
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    header = f"{'Model':<12} {'Acc':<8} {'Params':<10} {'Flash (KB)':<12} {'SRAM (B)':<10} {'Latency (ms)':<12}"
    print(header)
    print("-" * 80)
    
    for result in results_list:
        perf = result['performance']
        row = (f"{result['name']:<12} "
               f"{result['accuracy']:<8.4f} "
               f"{result['parameters']:<10,} "
               f"{perf['flash_bytes']/1024:<12.2f} "
               f"{perf['sram_bytes']:<10,} "
               f"{perf['latency_ms']:<12.2f}")
        print(row)
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark optimized models')
    
    parser.add_argument('--baseline', type=str, default='../results/baseline_real.pt')
    parser.add_argument('--pruned', type=str, default='../results/pruned_real.pt')
    parser.add_argument('--binary', type=str, default='../results/binary_real.pt')
    parser.add_argument('--data', type=str,
                       default='../../../z80ai/examples/tinychat/training-data.txt.gz')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=str, default='../results/benchmark_real.json')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n[1] Loading data...")
    train_loader, val_loader, charset = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        train_split=0.8
    )
    
    criterion = nn.CrossEntropyLoss()
    
    models_to_test = [
        (args.baseline, 'Baseline'),
        (args.pruned, 'Pruned'),
        (args.binary, 'Binary')
    ]
    
    results_list = []
    
    for i, (model_path, name) in enumerate(models_to_test, 1):
        print(f"\n[{i+1}] Benchmarking {name}...")
        
        if not os.path.exists(model_path):
            print(f"  ⚠ Model not found: {model_path}")
            continue
        
        result = benchmark_model(model_path, val_loader, criterion, device, charset, name)
        results_list.append(result)
    
    print_results(results_list)
    
    with open(args.output, 'w') as f:
        json.dump(results_list, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
