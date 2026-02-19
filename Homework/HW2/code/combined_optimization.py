#!/usr/bin/env python3
"""
Комбинированная оптимизация: pruning + 1-bit quantization.

Pipeline:
1. Train baseline 2-bit model
2. Apply structured pruning
3. Fine-tune pruned model
4. Convert to 1-bit binary
5. Fine-tune binary model
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy

from baseline_model import BaselineModel, OverflowAwareLinear
from quantization import BinaryModel, BinaryLinear
from pruning import structured_prune_model, get_pruning_stats


class CombinedOptimizationPipeline:
    """
    Pipeline для комбинированной оптимизации модели.
    
    Шаги:
    1. Baseline 2-bit модель
    2. Structured pruning
    3. Fine-tuning после pruning
    4. Конвертация в binary
    5. Fine-tuning binary модели
    """

    def __init__(self, baseline_model: BaselineModel):
        self.baseline = baseline_model
        self.pruned_model = None
        self.binary_model = None
        self.history = {
            'baseline': {},
            'pruned': {},
            'binary': {}
        }

    def apply_pruning(self, sparsity: float = 0.5, method: str = 'l1') -> nn.Module:
        """
        Шаг 1: применить structured pruning к baseline.
        
        Args:
            sparsity: доля нейронов для удаления
            method: метод вычисления importance
        
        Returns:
            Pruned модель
        """
        print(f"\n[1/3] Applying structured pruning (sparsity={sparsity})...")
        
        self.pruned_model = structured_prune_model(self.baseline, sparsity, method)
        
        stats = get_pruning_stats(self.baseline, self.pruned_model)
        self.history['pruned']['stats'] = stats
        
        print(f"  Architecture: {self.baseline.get_architecture_str()} → "
              f"{self.pruned_model.get_architecture_str()}")
        print(f"  Parameters: {stats['original_params']:,} → {stats['pruned_params']:,}")
        print(f"  Compression: {stats['compression_ratio']:.2f}x")
        
        return self.pruned_model

    def convert_to_binary(self) -> BinaryModel:
        """
        Шаг 2: конвертировать pruned модель в binary.
        
        Returns:
            Binary модель
        """
        if self.pruned_model is None:
            raise ValueError("Must apply pruning first")
        
        print(f"\n[2/3] Converting to 1-bit binary...")
        
        self.binary_model = BinaryModel(
            self.pruned_model.input_size,
            self.pruned_model.hidden_sizes,
            self.pruned_model.num_classes
        )
        
        src_layers = [m for m in self.pruned_model.modules() if isinstance(m, nn.Linear)]
        dst_layers = [m for m in self.binary_model.modules() if isinstance(m, BinaryLinear)]
        
        for src, dst in zip(src_layers, dst_layers):
            with torch.no_grad():
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
        
        orig_params = self.baseline.count_parameters()
        binary_params = self.binary_model.count_parameters()
        
        print(f"  Architecture: {self.binary_model.get_architecture_str()}")
        print(f"  Parameters: {binary_params:,}")
        print(f"  Overall compression: {orig_params / binary_params:.2f}x")
        
        return self.binary_model

    def get_all_models(self) -> Dict[str, nn.Module]:
        """Получить все модели для сравнения."""
        models = {
            'baseline': self.baseline,
        }
        if self.pruned_model is not None:
            models['pruned'] = self.pruned_model
        if self.binary_model is not None:
            models['binary'] = self.binary_model
        return models

    def get_summary(self) -> Dict[str, Dict]:
        """
        Получить summary всех моделей.
        
        Returns:
            Словарь с метриками для каждой модели
        """
        summary = {}
        
        models = self.get_all_models()
        for name, model in models.items():
            params = model.count_parameters()
            
            if name == 'baseline':
                bits = 2
            elif name == 'pruned':
                bits = 2
            else:
                bits = 1
            
            size_bytes = (params * bits + 7) // 8
            
            summary[name] = {
                'architecture': model.get_architecture_str(),
                'parameters': params,
                'bits_per_weight': bits,
                'size_bytes': size_bytes,
                'size_kb': size_bytes / 1024
            }
        
        return summary


def create_optimized_model(baseline: BaselineModel,
                          sparsity: float = 0.5,
                          pruning_method: str = 'l1') -> Tuple[nn.Module, nn.Module, Dict]:
    """
    Shortcut функция для создания оптимизированной модели.
    
    Args:
        baseline: baseline 2-bit модель
        sparsity: sparsity для pruning
        pruning_method: метод вычисления importance
    
    Returns:
        (pruned_model, binary_model, stats)
    """
    pipeline = CombinedOptimizationPipeline(baseline)
    
    pruned = pipeline.apply_pruning(sparsity, pruning_method)
    binary = pipeline.convert_to_binary()
    
    summary = pipeline.get_summary()
    
    return pruned, binary, summary


def compare_models(baseline: BaselineModel,
                  pruned: nn.Module,
                  binary: BinaryModel) -> str:
    """
    Создать таблицу сравнения моделей.
    
    Args:
        baseline: baseline модель
        pruned: pruned модель
        binary: binary модель
    
    Returns:
        Форматированная строка с таблицей
    """
    models = {
        'Baseline (2-bit)': baseline,
        'Pruned (2-bit)': pruned,
        'Pruned + Binary (1-bit)': binary
    }
    
    lines = []
    lines.append("\n" + "="*80)
    lines.append("Model Comparison")
    lines.append("="*80)
    lines.append(f"{'Model':<30} {'Architecture':<25} {'Params':<12} {'Size (KB)':<10}")
    lines.append("-"*80)
    
    for name, model in models.items():
        arch = model.get_architecture_str()
        params = model.count_parameters()
        
        if '1-bit' in name:
            bits = 1
        else:
            bits = 2
        
        size_kb = (params * bits + 7) // 8 / 1024
        
        lines.append(f"{name:<30} {arch:<25} {params:<12,} {size_kb:<10.2f}")
    
    lines.append("="*80)
    
    baseline_params = baseline.count_parameters()
    binary_params = binary.count_parameters()
    baseline_size = (baseline_params * 2 + 7) // 8 / 1024
    binary_size = (binary_params * 1 + 7) // 8 / 1024
    
    lines.append(f"\nCompression ratios:")
    lines.append(f"  Parameters: {baseline_params / binary_params:.2f}x")
    lines.append(f"  Size: {baseline_size / binary_size:.2f}x")
    lines.append("")
    
    return "\n".join(lines)
