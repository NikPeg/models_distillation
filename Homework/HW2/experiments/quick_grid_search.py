#!/usr/bin/env python3
"""
Быстрый grid search с уменьшенным числом эпох для ускорения экспериментов.
Подходит для быстрой оценки параметров (15 запусков за ~10-15 минут).
"""

import subprocess
import json
import csv
import time
from pathlib import Path


# Быстрые комбинации параметров (50 эпох для скорости)
QUICK_CONFIGS = [
    # Разные learning rates
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.001, 'patience': 15, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0005, 'patience': 15, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0002, 'patience': 15, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0001, 'patience': 15, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.00005, 'patience': 15, 'batch_size': 32},
    
    # Разные patience с хорошим lr
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0002, 'patience': 10, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0002, 'patience': 20, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0002, 'patience': 30, 'batch_size': 32},
    
    # Разные warmup
    {'epochs': 50, 'warmup_epochs': 5, 'lr': 0.0002, 'patience': 15, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 15, 'lr': 0.0002, 'patience': 15, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 20, 'lr': 0.0002, 'patience': 15, 'batch_size': 32},
    
    # Разные batch sizes
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0002, 'patience': 15, 'batch_size': 16},
    {'epochs': 50, 'warmup_epochs': 10, 'lr': 0.0002, 'patience': 15, 'batch_size': 64},
    
    # Комбинированные варианты
    {'epochs': 50, 'warmup_epochs': 15, 'lr': 0.0005, 'patience': 20, 'batch_size': 32},
    {'epochs': 50, 'warmup_epochs': 5, 'lr': 0.0001, 'patience': 25, 'batch_size': 64},
]


def run_training(params, run_id):
    """Запустить обучение с заданными параметрами."""
    output_path = f'../results/quick_grid/run_{run_id:02d}.pt'
    
    cmd = [
        'python3', 'train_real.py',
        '--epochs', str(params['epochs']),
        '--warmup-epochs', str(params['warmup_epochs']),
        '--lr', str(params['lr']),
        '--patience', str(params['patience']),
        '--batch-size', str(params['batch_size']),
        '--output', output_path
    ]
    
    print(f"\n[{run_id}/{len(QUICK_CONFIGS)}] Running: lr={params['lr']}, patience={params['patience']}, batch={params['batch_size']}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        elapsed_time = time.time() - start_time
        
        # Парсим accuracy
        val_acc = None
        stopped_at = None
        
        for line in result.stdout.split('\n'):
            if 'Best val accuracy:' in line:
                val_acc = float(line.split(':')[1].strip())
            elif 'Early stopping at epoch' in line:
                stopped_at = int(line.split('epoch')[1].strip())
        
        if val_acc is None:
            for line in reversed(result.stdout.split('\n')):
                if 'val_acc=' in line:
                    val_acc = float(line.split('val_acc=')[1].split(',')[0].strip())
                    break
        
        if stopped_at is None:
            stopped_at = params['epochs']
        
        success = result.returncode == 0 and val_acc is not None
        
        if success:
            print(f"  ✓ Val Acc: {val_acc:.4f} (stopped at epoch {stopped_at})")
        else:
            print(f"  ✗ Failed")
        
        return {
            'run_id': run_id,
            'success': success,
            'val_acc': val_acc if val_acc else 0.0,
            'stopped_at': stopped_at,
            'elapsed_time': elapsed_time,
            'output_path': output_path if success else None,
            **params
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            'run_id': run_id,
            'success': False,
            'val_acc': 0.0,
            'stopped_at': None,
            'elapsed_time': 0,
            'output_path': None,
            **params
        }


def save_results(results, output_dir):
    """Сохранить результаты."""
    sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    csv_path = output_dir / 'quick_grid_results.csv'
    with open(csv_path, 'w', newline='') as f:
        if sorted_results:
            writer = csv.DictWriter(f, fieldnames=sorted_results[0].keys())
            writer.writeheader()
            writer.writerows(sorted_results)
    
    json_path = output_dir / 'quick_grid_results.json'
    with open(json_path, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    
    return sorted_results, csv_path


def print_summary(results):
    """Вывести сводку."""
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("\n⚠️  No successful runs!")
        return
    
    sorted_results = sorted(successful, key=lambda x: x['val_acc'], reverse=True)
    
    print("\n" + "="*80)
    print("QUICK GRID SEARCH RESULTS")
    print("="*80)
    
    print(f"\nSuccessful runs: {len(successful)}/{len(results)}")
    
    print("\nTOP 5:")
    print("-"*80)
    print(f"{'#':<4} {'Val Acc':<10} {'LR':<12} {'Patience':<10} {'Batch':<8} {'Time':<8}")
    print("-"*80)
    
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i:<4} {r['val_acc']:<10.4f} {r['lr']:<12.6f} {r['patience']:<10} {r['batch_size']:<8} {r['elapsed_time']:<8.1f}s")
    
    print("\n" + "="*80)
    best = sorted_results[0]
    print("BEST CONFIG:")
    print(f"  Val Accuracy:  {best['val_acc']:.4f}")
    print(f"  Learning Rate: {best['lr']}")
    print(f"  Patience:      {best['patience']}")
    print(f"  Batch Size:    {best['batch_size']}")
    print(f"  Warmup:        {best['warmup_epochs']}")
    print(f"  Stopped at:    {best['stopped_at']}/{best['epochs']}")
    print("="*80)


def main():
    output_dir = Path('../results/quick_grid')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("QUICK GRID SEARCH (15 runs, ~50 epochs each)")
    print("="*80)
    print(f"\nEstimated time: ~10-15 minutes")
    
    start_time = time.time()
    results = []
    
    for i, params in enumerate(QUICK_CONFIGS, 1):
        result = run_training(params, i)
        results.append(result)
    
    total_time = time.time() - start_time
    
    sorted_results, csv_path = save_results(results, output_dir)
    print_summary(sorted_results)
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {csv_path}")
    print("\n✓ Quick grid search complete!")


if __name__ == '__main__':
    main()
