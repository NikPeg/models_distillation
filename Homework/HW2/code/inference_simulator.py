#!/usr/bin/env python3
"""
Arduino inference simulator для оценки производительности.

Эмулирует ограничения Arduino UNO:
- Flash: 32 KB
- SRAM: 2 KB
- CPU: 16 MHz (AVR)

Оценивает:
- Размер модели (Flash usage)
- Runtime SRAM usage
- Latency (на основе cycle count)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from baseline_model import OverflowAwareLinear
from quantization import BinaryLinear


class ArduinoConstraints:
    """Константы для Arduino UNO."""
    FLASH_SIZE = 32 * 1024
    SRAM_SIZE = 2 * 1024
    CPU_MHZ = 16
    
    CYCLES_2BIT_UNPACK = 12
    CYCLES_1BIT_UNPACK = 4
    CYCLES_MAC_2BIT = 50
    CYCLES_MAC_1BIT = 15
    CYCLES_RELU = 20
    CYCLES_BIAS = 10


class ArduinoInferenceSimulator:
    """
    Симулятор inference на Arduino UNO.
    
    Оценивает Flash usage, SRAM usage и latency.
    """

    def __init__(self):
        self.constraints = ArduinoConstraints()

    def count_weights(self, model: nn.Module, bits_per_weight: int) -> int:
        """
        Подсчитать размер весов в байтах (Flash usage).
        
        Args:
            model: PyTorch модель
            bits_per_weight: бит на вес (1 или 2)
        
        Returns:
            Размер в байтах
        """
        total_weights = 0
        for module in model.modules():
            if isinstance(module, (nn.Linear, OverflowAwareLinear, BinaryLinear)):
                total_weights += module.weight.numel()
                if module.bias is not None:
                    total_weights += module.bias.numel()
        
        total_bits = total_weights * bits_per_weight
        if module.bias is not None:
            total_bits += sum(m.bias.numel() * 16 for m in model.modules() 
                            if isinstance(m, (nn.Linear, OverflowAwareLinear, BinaryLinear)))
        
        return (total_bits + 7) // 8

    def count_activations(self, model: nn.Module, input_size: int) -> int:
        """
        Подсчитать SRAM usage для activations.
        
        Args:
            model: PyTorch модель
            input_size: размер входа
        
        Returns:
            SRAM usage в байтах
        """
        max_activation_size = input_size
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, OverflowAwareLinear, BinaryLinear)):
                max_activation_size = max(max_activation_size, module.out_features)
        
        bytes_per_activation = 2
        input_buffer = input_size * bytes_per_activation
        output_buffer = max_activation_size * bytes_per_activation
        
        return input_buffer + output_buffer

    def estimate_cycles(self, model: nn.Module, bits_per_weight: int) -> int:
        """
        Оценить cycle count для одного inference.
        
        Args:
            model: PyTorch модель
            bits_per_weight: бит на вес (1 или 2)
        
        Returns:
            Число циклов
        """
        total_cycles = 0
        
        if bits_per_weight == 2:
            cycles_unpack = self.constraints.CYCLES_2BIT_UNPACK
            cycles_mac = self.constraints.CYCLES_MAC_2BIT
        else:
            cycles_unpack = self.constraints.CYCLES_1BIT_UNPACK
            cycles_mac = self.constraints.CYCLES_MAC_1BIT
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, OverflowAwareLinear, BinaryLinear)):
                num_weights = module.weight.numel()
                num_outputs = module.out_features
                
                total_cycles += num_weights * cycles_unpack
                total_cycles += num_weights * cycles_mac
                total_cycles += num_outputs * self.constraints.CYCLES_BIAS
            
            elif isinstance(module, nn.ReLU):
                pass
        
        return total_cycles

    def measure_inference(self, model: nn.Module, input_size: int, 
                         bits_per_weight: int = 2) -> Dict[str, float]:
        """
        Измерить все метрики для модели.
        
        Args:
            model: PyTorch модель
            input_size: размер входа
            bits_per_weight: бит на вес
        
        Returns:
            Словарь с метриками
        """
        flash_used = self.count_weights(model, bits_per_weight)
        sram_used = self.count_activations(model, input_size)
        cycles = self.estimate_cycles(model, bits_per_weight)
        
        latency_ms = cycles / (self.constraints.CPU_MHZ * 1000)
        
        flash_ok = flash_used <= self.constraints.FLASH_SIZE
        sram_ok = sram_used <= self.constraints.SRAM_SIZE
        
        return {
            'flash_bytes': flash_used,
            'flash_kb': flash_used / 1024,
            'flash_usage_percent': (flash_used / self.constraints.FLASH_SIZE) * 100,
            'flash_ok': flash_ok,
            'sram_bytes': sram_used,
            'sram_kb': sram_used / 1024,
            'sram_usage_percent': (sram_used / self.constraints.SRAM_SIZE) * 100,
            'sram_ok': sram_ok,
            'cycles': cycles,
            'latency_ms': latency_ms,
            'throughput_per_sec': 1000 / latency_ms if latency_ms > 0 else 0,
            'fits_on_arduino': flash_ok and sram_ok
        }

    def print_report(self, model: nn.Module, input_size: int, 
                    bits_per_weight: int = 2, model_name: str = "Model"):
        """
        Вывести отчёт об inference на Arduino.
        
        Args:
            model: PyTorch модель
            input_size: размер входа
            bits_per_weight: бит на вес
            model_name: имя модели для отчёта
        """
        metrics = self.measure_inference(model, input_size, bits_per_weight)
        
        print(f"\n{'='*70}")
        print(f"Arduino Inference Report: {model_name}")
        print(f"{'='*70}")
        
        print(f"\nFlash Usage (Model Weights):")
        print(f"  Size: {metrics['flash_kb']:.2f} KB ({metrics['flash_bytes']:,} bytes)")
        print(f"  Usage: {metrics['flash_usage_percent']:.1f}% of 32 KB")
        print(f"  Status: {'✓ OK' if metrics['flash_ok'] else '✗ TOO LARGE'}")
        
        print(f"\nSRAM Usage (Runtime):")
        print(f"  Size: {metrics['sram_kb']:.2f} KB ({metrics['sram_bytes']:,} bytes)")
        print(f"  Usage: {metrics['sram_usage_percent']:.1f}% of 2 KB")
        print(f"  Status: {'✓ OK' if metrics['sram_ok'] else '✗ TOO LARGE'}")
        
        print(f"\nPerformance:")
        print(f"  Cycles: {metrics['cycles']:,}")
        print(f"  Latency: {metrics['latency_ms']:.2f} ms @ 16 MHz")
        print(f"  Throughput: {metrics['throughput_per_sec']:.1f} inferences/sec")
        
        print(f"\n{'='*70}")
        print(f"Fits on Arduino UNO: {'✓ YES' if metrics['fits_on_arduino'] else '✗ NO'}")
        print(f"{'='*70}\n")


def compare_models_on_arduino(models: Dict[str, Tuple[nn.Module, int, int]]) -> str:
    """
    Сравнить несколько моделей на Arduino.
    
    Args:
        models: словарь {name: (model, input_size, bits_per_weight)}
    
    Returns:
        Форматированная таблица сравнения
    """
    simulator = ArduinoInferenceSimulator()
    
    results = {}
    for name, (model, input_size, bits) in models.items():
        results[name] = simulator.measure_inference(model, input_size, bits)
    
    lines = []
    lines.append("\n" + "="*100)
    lines.append("Arduino Inference Comparison")
    lines.append("="*100)
    lines.append(f"{'Model':<25} {'Flash (KB)':<12} {'SRAM (B)':<12} {'Latency (ms)':<15} {'Fits?':<8}")
    lines.append("-"*100)
    
    for name, metrics in results.items():
        fits = '✓' if metrics['fits_on_arduino'] else '✗'
        lines.append(f"{name:<25} {metrics['flash_kb']:<12.2f} {metrics['sram_bytes']:<12} "
                    f"{metrics['latency_ms']:<15.2f} {fits:<8}")
    
    lines.append("="*100)
    lines.append("")
    
    return "\n".join(lines)


def create_comparison_table(baseline_metrics: Dict, 
                           pruned_metrics: Dict,
                           binary_metrics: Dict) -> str:
    """
    Создать таблицу сравнения из готовых метрик.
    
    Args:
        baseline_metrics: метрики baseline модели
        pruned_metrics: метрики pruned модели
        binary_metrics: метрики binary модели
    
    Returns:
        Форматированная таблица
    """
    models = {
        'Baseline (2-bit)': baseline_metrics,
        'Pruned (2-bit)': pruned_metrics,
        'Binary (1-bit)': binary_metrics
    }
    
    lines = []
    lines.append("\n" + "="*100)
    lines.append(f"{'Model':<25} {'Flash (KB)':<12} {'SRAM (B)':<12} {'Latency (ms)':<15} {'Speedup':<10}")
    lines.append("-"*100)
    
    baseline_latency = baseline_metrics['latency_ms']
    
    for name, metrics in models.items():
        speedup = baseline_latency / metrics['latency_ms'] if metrics['latency_ms'] > 0 else 0
        lines.append(f"{name:<25} {metrics['flash_kb']:<12.2f} {metrics['sram_bytes']:<12} "
                    f"{metrics['latency_ms']:<15.2f} {speedup:<10.2f}x")
    
    lines.append("="*100)
    
    return "\n".join(lines)
