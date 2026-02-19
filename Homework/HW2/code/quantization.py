#!/usr/bin/env python3
"""
1-bit binary quantization с Straight-Through Estimator.

Реализует Binary Neural Networks где веса квантуются в {-1, +1}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from baseline_model import StraightThroughEstimator


def quantize_weights_1bit(w: torch.Tensor, hard: bool = True) -> torch.Tensor:
    """
    Квантование весов в 1-bit: {-1, +1} используя sign function.
    
    Args:
        w: веса
        hard: использовать STE для градиентов
    
    Returns:
        Бинарные веса {-1, +1}
    """
    w_binary = torch.sign(w)
    w_binary = torch.where(w_binary == 0, torch.ones_like(w_binary), w_binary)
    
    if hard:
        return StraightThroughEstimator.apply(w, w_binary)
    else:
        return w_binary


def binarization_loss(w: torch.Tensor) -> torch.Tensor:
    """
    Loss для поощрения весов быть близкими к {-1, +1}.
    
    Минимизирует расстояние до ближайшего бинарного значения.
    """
    w_sign = torch.sign(w)
    w_sign = torch.where(w_sign == 0, torch.ones_like(w_sign), w_sign)
    distance = (w - w_sign).abs()
    return distance.mean()


class BinaryLinear(nn.Module):
    """
    Linear слой с 1-bit весами {-1, +1}.
    
    Во время forward pass веса бинаризуются, но градиенты 
    проходят через STE как если бы веса были real-valued.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 
                                   np.sqrt(2.0 / (in_features + out_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_binary = quantize_weights_1bit(self.weight, hard=True)
        return F.linear(x, w_binary, self.bias)

    def get_binarization_loss(self) -> torch.Tensor:
        return binarization_loss(self.weight)

    def get_weight_stats(self) -> Dict[str, float]:
        """Статистика весов для мониторинга обучения."""
        with torch.no_grad():
            w = self.weight.detach()
            w_binary = quantize_weights_1bit(w, hard=False)
            
            return {
                'mean': w.mean().item(),
                'std': w.std().item(),
                'min': w.min().item(),
                'max': w.max().item(),
                'binary_match': (w_binary == torch.sign(w)).float().mean().item()
            }


class BinaryModel(nn.Module):
    """
    Модель с 1-bit весами.
    
    Использует BinaryLinear слои для экстремального сжатия.
    """

    def __init__(self, input_size: int,
                 hidden_sizes: List[int],
                 num_classes: int):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(BinaryLinear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(BinaryLinear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_total_binarization_loss(self) -> torch.Tensor:
        """Суммарный binarization loss по всем слоям."""
        total = 0.0
        for module in self.network:
            if isinstance(module, BinaryLinear):
                total = total + module.get_binarization_loss()
        return total

    def get_weight_statistics(self) -> Dict[str, Dict[str, float]]:
        """Статистика весов по всем слоям."""
        stats = {}
        layer_idx = 0
        for module in self.network:
            if isinstance(module, BinaryLinear):
                layer_idx += 1
                stats[f'layer{layer_idx}'] = module.get_weight_stats()
        return stats

    def get_architecture_str(self) -> str:
        """Строковое представление архитектуры."""
        sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]
        return "→".join(map(str, sizes))

    def count_parameters(self) -> int:
        """Подсчёт общего числа параметров."""
        return sum(p.numel() for p in self.parameters())


def convert_to_binary(model: nn.Module) -> BinaryModel:
    """
    Конвертировать существующую модель в бинарную.
    
    Копирует архитектуру и инициализирует веса из исходной модели.
    
    Args:
        model: исходная модель (например, BaselineModel)
    
    Returns:
        BinaryModel с той же архитектурой
    """
    if hasattr(model, 'input_size') and hasattr(model, 'hidden_sizes'):
        binary_model = BinaryModel(
            model.input_size,
            model.hidden_sizes,
            model.num_classes
        )
        
        src_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        dst_layers = [m for m in binary_model.modules() if isinstance(m, BinaryLinear)]
        
        for src, dst in zip(src_layers, dst_layers):
            with torch.no_grad():
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
        
        return binary_model
    else:
        raise ValueError("Model must have input_size, hidden_sizes, and num_classes attributes")


def create_binary_model(input_size: int,
                        hidden_sizes: List[int],
                        num_classes: int) -> BinaryModel:
    """Создать новую бинарную модель."""
    return BinaryModel(input_size, hidden_sizes, num_classes)
