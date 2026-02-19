#!/usr/bin/env python3
"""
Baseline модель с 2-bit QAT из HW1.

Адаптировано из z80ai/libqat.py для использования в экспериментах HW2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


MAX_ACCUM = 32767
MIN_ACCUM = -32768
ACTIVATION_SCALE = 32


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator для недифференцируемых операций."""

    @staticmethod
    def forward(ctx, x, x_quantized):
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize_weights_2bit(w: torch.Tensor, hard: bool = True, temperature: float = 1.0) -> torch.Tensor:
    """
    Квантование весов в 2-bit: {-2, -1, 0, +1}.
    
    Args:
        w: веса
        hard: использовать STE для градиентов
        temperature: 0.0 = float, 1.0 = полностью квантованные
    """
    if temperature <= 0:
        return w

    scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
    w_scaled = w / scale
    w_quant = torch.clamp(torch.round(w_scaled), -2, 1) * scale

    if temperature >= 1.0:
        if hard:
            return StraightThroughEstimator.apply(w, w_quant)
        else:
            return w_quant
    else:
        w_blend = (1 - temperature) * w + temperature * w_quant
        if hard:
            return StraightThroughEstimator.apply(w, w_blend)
        else:
            return w_blend


def quantization_friendly_loss(w: torch.Tensor) -> torch.Tensor:
    """Loss для поощрения весов быть близкими к квантовой сетке."""
    scale = torch.quantile(w.abs().flatten(), 0.95).clamp(min=1e-6)
    w_scaled = w / scale
    w_rounded = torch.clamp(torch.round(w_scaled), -2, 1)
    distance = (w_scaled - w_rounded).abs()
    return distance.mean()


class OverflowAwareLinear(nn.Module):
    """Linear слой с 2-bit квантованием и контролем overflow."""

    def __init__(self, in_features: int, out_features: int,
                 simulate_overflow: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.simulate_overflow = simulate_overflow

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 
                                   np.sqrt(2.0 / (in_features + out_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.register_buffer('max_accum_seen', torch.tensor(0.0))

    def forward(self, x: torch.Tensor, quant_temp: float = 1.0) -> torch.Tensor:
        w_quant = quantize_weights_2bit(self.weight, hard=True, temperature=quant_temp)
        out = F.linear(x, w_quant, self.bias)

        if self.training and self.simulate_overflow:
            with torch.no_grad():
                w_hard = quantize_weights_2bit(self.weight, hard=False, temperature=1.0)
                worst_case = (w_hard.abs() @ x.abs().T).max()
                self.max_accum_seen = max(self.max_accum_seen, worst_case)

        return out

    def get_quantization_loss(self) -> torch.Tensor:
        return quantization_friendly_loss(self.weight)

    def get_overflow_risk(self) -> float:
        return (self.max_accum_seen / MAX_ACCUM).item()


class BaselineModel(nn.Module):
    """
    Baseline модель с 2-bit квантованием из HW1.
    
    Архитектура: 256→256→192→128→40
    """

    def __init__(self, input_size: int = 256, 
                 hidden_sizes: List[int] = [256, 192, 128],
                 num_classes: int = 40,
                 simulate_overflow: bool = True):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.simulate_overflow = simulate_overflow

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(OverflowAwareLinear(prev_size, hidden_size, simulate_overflow))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(OverflowAwareLinear(prev_size, num_classes, simulate_overflow))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, quant_temp: float = 1.0) -> torch.Tensor:
        """
        Forward pass с опциональным temperature для progressive quantization.
        
        Args:
            x: входные данные [batch, input_size]
            quant_temp: температура квантования (0.0 = float, 1.0 = 2-bit)
        """
        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                x = module(x, quant_temp)
            else:
                x = module(x)
        return x

    def get_total_quantization_loss(self) -> torch.Tensor:
        """Суммарный quantization loss по всем слоям."""
        total = 0.0
        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                total = total + module.get_quantization_loss()
        return total

    def get_overflow_stats(self) -> Dict[str, float]:
        """Статистика overflow risk по всем слоям."""
        stats = {}
        layer_idx = 0
        for module in self.network:
            if isinstance(module, OverflowAwareLinear):
                layer_idx += 1
                stats[f'layer{layer_idx}'] = module.get_overflow_risk()
        return stats

    def get_architecture_str(self) -> str:
        """Строковое представление архитектуры."""
        sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]
        return "→".join(map(str, sizes))

    def count_parameters(self) -> int:
        """Подсчёт общего числа параметров."""
        return sum(p.numel() for p in self.parameters())


def create_baseline_model(input_size: int = 256,
                          hidden_sizes: List[int] = [256, 192, 128],
                          num_classes: int = 40) -> BaselineModel:
    """Создать baseline модель из HW1."""
    return BaselineModel(input_size, hidden_sizes, num_classes, simulate_overflow=True)


def create_micro_model(input_size: int = 64,
                       hidden_sizes: List[int] = [32, 16],
                       num_classes: int = 10) -> BaselineModel:
    """Создать micro модель для Arduino (из плана HW1)."""
    return BaselineModel(input_size, hidden_sizes, num_classes, simulate_overflow=True)
