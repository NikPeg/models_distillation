#!/usr/bin/env python3
"""
Общие утилиты для HW2.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def count_parameters(model: torch.nn.Module) -> int:
    """Подсчёт числа параметров в модели."""
    return sum(p.numel() for p in model.parameters())


def calculate_model_size(model: torch.nn.Module, bits_per_param: int = 2) -> int:
    """
    Оценка размера модели в байтах.
    
    Args:
        model: PyTorch модель
        bits_per_param: бит на параметр (2 для 2-bit, 1 для binary)
    
    Returns:
        Размер в байтах
    """
    num_params = count_parameters(model)
    return (num_params * bits_per_param + 7) // 8


def save_model_npz(model: torch.nn.Module, path: str, bits: int = 2):
    """Сохранить веса модели в .npz формате."""
    state = {}
    layer_idx = 1
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else np.zeros(module.out_features)
            
            state[f'fc{layer_idx}_weight'] = weight
            state[f'fc{layer_idx}_bias'] = bias
            layer_idx += 1
    
    np.savez(path, **state)
    print(f"✓ Модель сохранена в {path}")


def load_model_npz(path: str) -> Dict[str, np.ndarray]:
    """Загрузить веса модели из .npz файла."""
    return dict(np.load(path))


def format_size(size_bytes: int) -> str:
    """Форматирование размера в читаемый вид."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Вычисление accuracy."""
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


class EarlyStopping:
    """Early stopping для предотвращения overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop
