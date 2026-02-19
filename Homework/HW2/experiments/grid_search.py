#!/usr/bin/env python3
"""
Grid search для подбора гиперпараметров baseline модели.
Запускает обучение с разными комбинациями параметров и сохраняет результаты.
"""

import subprocess
import json
import csv
import time
from datetime import datetime
from pathlib import Path
import itertools


def run_training(params, run_id):
    """Запустить обучение с заданными параметрами."""
    output_path = f'../results/grid_search/run_{run_id:03d}.pt'
    
    cmd = [
        'python3', 'train_real.py',
        '--epochs', str(params['epochs']),
        '--warmup-epochs', str(params['warmup_epochs']),
        '--lr', str(params['lr']),
        '--patience', str(params['patience']),
        '--batch-size', str(params['batch_size']),
        '--output', output_path
    ]
    
    print(f"\n{'='*80}")
    print(f"RUN {run_id}: {params}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 минут максимум
        )
        
        elapsed_time = time.time() - start_time
        
        # Парсим вывод для извлечения accuracy
        output_lines = result.stdout.split('\n')
        val_acc = None
        train_acc = None
        stopped_at = None
        
        for line in output_lines:
            if 'Best val accuracy:' in line:
                val_acc = float(line.split(':')[1].strip())
            elif 'Early stopping at epoch' in line:
                stopped_at = int(line.split('epoch')[1].strip())
            elif 'Training complete!' in line:
                # Если не было early stopping, значит прошли все эпохи
                if stopped_at is None:
                    stopped_at = params['epochs']
        
        # Если не нашли accuracy в выводе, ищем в последней строке с метриками
        if val_acc is None:
            for line in reversed(output_lines):
                if 'val_acc=' in line:
                    parts = line.split('val_acc=')
                    if len(parts) > 1:
                        val_acc = float(parts[1].split(',')[0].strip())
                        break
        
        success = result.returncode == 0 and val_acc is not None
        
        return {
            'run_id': run_id,
            'success': success,
            'val_acc': val_acc if val_acc else 0.0,
            'stopped_at': stopped_at if stopped_at else params['epochs'],
            'elapsed_time': elapsed_time,
            'output_path': output_path if success else None,
            **params
        }
        
    except subprocess.TimeoutExpired:
        print(f"⚠️  Run {run_id} timed out!")
        return {
            'run_id': run_id,
            'success': False,
            'val_acc': 0.0,
            'stopped_at': None,
            'elapsed_time': 600,
            'output_path': None,
            **params
        }
    except Exception as e:
        print(f"⚠️  Run {run_id} failed: {e}")
        return {
            'run_id': run_id,
            'success': False,
            'val_acc': 0.0,
            'stopped_at': None,
            'elapsed_time': 0,
            'output_path': None,
            **params
        }


def generate_param_combinations():
    """Сгенерировать комбинации параметров для grid search."""
    
    # Определяем диапазоны параметров
    param_grid = {
        'epochs': [100, 150, 200],
        'warmup_epochs': [20, 30, 40],
        'lr': [0.001, 0.0005, 0.0002, 0.0001],
        'patience': [20, 30, 50],
        'batch_size': [32, 64]
    }
    
    # Генерируем все комбинации
    keys = param_grid.keys()
    values = param_grid.values()
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Фильтруем комбинации (warmup_epochs должен быть меньше epochs)
    valid_combinations = [
        c for c in all_combinations 
        if c['warmup_epochs'] < c['epochs']
    ]
    
    # Для быстрого старта выбираем подмножество (top 20 комбинаций)
    # Приоритет: средние значения epochs, разные lr
    priority_combinations = []
    
    # 1. Базовые комбинации с разными lr
    for lr in [0.001, 0.0005, 0.0002, 0.0001]:
        priority_combinations.append({
            'epochs': 150,
            'warmup_epochs': 30,
            'lr': lr,
            'patience': 30,
            'batch_size': 32
        })
    
    # 2. Разные patience с оптимальным lr
    for patience in [20, 30, 50]:
        priority_combinations.append({
            'epochs': 150,
            'warmup_epochs': 30,
            'lr': 0.0002,
            'patience': patience,
            'batch_size': 32
        })
    
    # 3. Разные warmup с оптимальным lr
    for warmup in [20, 30, 40]:
        priority_combinations.append({
            'epochs': 150,
            'warmup_epochs': warmup,
            'lr': 0.0002,
            'patience': 30,
            'batch_size': 32
        })
    
    # 4. Разные epochs
    for epochs in [100, 150, 200]:
        priority_combinations.append({
            'epochs': epochs,
            'warmup_epochs': min(30, epochs // 5),
            'lr': 0.0002,
            'patience': 30,
            'batch_size': 32
        })
    
    # 5. Разные batch_size
    for bs in [32, 64]:
        priority_combinations.append({
            'epochs': 150,
            'warmup_epochs': 30,
            'lr': 0.0002,
            'patience': 30,
            'batch_size': bs
        })
    
    # Удаляем дубликаты
    unique_combinations = []
    seen = set()
    for combo in priority_combinations:
        key = tuple(sorted(combo.items()))
        if key not in seen:
            seen.add(key)
            unique_combinations.append(combo)
    
    return unique_combinations[:20]  # Возвращаем топ-20


def save_results(results, output_dir):
    """Сохранить результаты в CSV и JSON."""
    
    # Сортируем по accuracy
    sorted_results = sorted(results, key=lambda x: x['val_acc'], reverse=True)
    
    # Сохраняем в CSV
    csv_path = output_dir / 'grid_search_results.csv'
    with open(csv_path, 'w', newline='') as f:
        if sorted_results:
            writer = csv.DictWriter(f, fieldnames=sorted_results[0].keys())
            writer.writeheader()
            writer.writerows(sorted_results)
    
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Сохраняем в JSON для программного доступа
    json_path = output_dir / 'grid_search_results.json'
    with open(json_path, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    
    print(f"✓ Results saved to: {json_path}")
    
    return sorted_results


def print_summary(results):
    """Вывести краткую сводку результатов."""
    
    successful_runs = [r for r in results if r['success']]
    
    if not successful_runs:
        print("\n⚠️  No successful runs!")
        return
    
    print("\n" + "="*80)
    print("GRID SEARCH SUMMARY")
    print("="*80)
    
    print(f"\nTotal runs: {len(results)}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(results) - len(successful_runs)}")
    
    print("\n" + "-"*80)
    print("TOP 5 RESULTS")
    print("-"*80)
    
    sorted_results = sorted(successful_runs, key=lambda x: x['val_acc'], reverse=True)
    
    header = f"{'Rank':<6} {'Val Acc':<10} {'LR':<10} {'Epochs':<8} {'Patience':<10} {'Batch':<8} {'Time (s)':<10}"
    print(header)
    print("-"*80)
    
    for i, result in enumerate(sorted_results[:5], 1):
        row = (f"{i:<6} "
               f"{result['val_acc']:<10.4f} "
               f"{result['lr']:<10.6f} "
               f"{result['stopped_at']:<8} "
               f"{result['patience']:<10} "
               f"{result['batch_size']:<8} "
               f"{result['elapsed_time']:<10.1f}")
        print(row)
    
    print("\n" + "-"*80)
    print("BEST CONFIGURATION")
    print("-"*80)
    
    best = sorted_results[0]
    print(f"  Validation Accuracy: {best['val_acc']:.4f}")
    print(f"  Learning Rate:       {best['lr']}")
    print(f"  Epochs (stopped):    {best['stopped_at']}/{best['epochs']}")
    print(f"  Warmup Epochs:       {best['warmup_epochs']}")
    print(f"  Patience:            {best['patience']}")
    print(f"  Batch Size:          {best['batch_size']}")
    print(f"  Training Time:       {best['elapsed_time']:.1f}s")
    print(f"  Model Path:          {best['output_path']}")
    
    print("\n" + "="*80)


def main():
    # Создаем директорию для результатов
    output_dir = Path('../results/grid_search')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GRID SEARCH: Hyperparameter Tuning for Baseline Model")
    print("="*80)
    
    # Генерируем комбинации параметров
    param_combinations = generate_param_combinations()
    
    print(f"\nGenerated {len(param_combinations)} parameter combinations")
    print(f"Estimated time: ~{len(param_combinations) * 2} minutes (assuming ~2 min per run)")
    
    input("\nPress Enter to start grid search...")
    
    start_time = time.time()
    results = []
    
    # Запускаем эксперименты
    for i, params in enumerate(param_combinations, 1):
        result = run_training(params, i)
        results.append(result)
        
        if result['success']:
            print(f"✓ Run {i}/{len(param_combinations)}: Val Acc = {result['val_acc']:.4f}")
        else:
            print(f"✗ Run {i}/{len(param_combinations)}: Failed")
        
        # Промежуточное сохранение каждые 5 запусков
        if i % 5 == 0:
            save_results(results, output_dir)
    
    total_time = time.time() - start_time
    
    # Финальное сохранение
    sorted_results = save_results(results, output_dir)
    
    # Вывод сводки
    print_summary(sorted_results)
    
    print(f"\nTotal grid search time: {total_time/60:.1f} minutes")
    print(f"\n✓ Grid search complete!")


if __name__ == '__main__':
    main()
