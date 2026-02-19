#!/usr/bin/env python3
"""
Structured pruning для уменьшения размера модели.

Удаляет целые нейроны на основе importance scores.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import copy


def compute_neuron_importance(layer: nn.Linear, method: str = 'l1') -> torch.Tensor:
    """
    Вычислить importance score для каждого нейрона.
    
    Args:
        layer: Linear слой
        method: метод вычисления ('l1', 'l2', 'variance')
    
    Returns:
        Tensor [out_features] с importance scores
    """
    with torch.no_grad():
        if method == 'l1':
            importance = layer.weight.abs().sum(dim=1)
        elif method == 'l2':
            importance = (layer.weight ** 2).sum(dim=1).sqrt()
        elif method == 'variance':
            importance = layer.weight.var(dim=1)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    return importance


def prune_layer(layer: nn.Linear, keep_indices: torch.Tensor) -> nn.Linear:
    """
    Создать новый слой, оставив только указанные нейроны.
    
    Args:
        layer: исходный слой
        keep_indices: индексы нейронов которые надо сохранить
    
    Returns:
        Новый pruned слой
    """
    new_out_features = len(keep_indices)
    new_layer = nn.Linear(layer.in_features, new_out_features, bias=(layer.bias is not None))
    
    with torch.no_grad():
        new_layer.weight.copy_(layer.weight[keep_indices])
        if layer.bias is not None:
            new_layer.bias.copy_(layer.bias[keep_indices])
    
    return new_layer


def prune_next_layer_inputs(layer: nn.Linear, keep_indices: torch.Tensor) -> nn.Linear:
    """
    Обновить следующий слой, удалив входы соответствующие pruned нейронам.
    
    Args:
        layer: слой идущий после pruned слоя
        keep_indices: индексы нейронов которые были сохранены в предыдущем слое
    
    Returns:
        Новый слой с уменьшенным количеством входов
    """
    new_in_features = len(keep_indices)
    new_layer = nn.Linear(new_in_features, layer.out_features, bias=(layer.bias is not None))
    
    with torch.no_grad():
        new_layer.weight.copy_(layer.weight[:, keep_indices])
        if layer.bias is not None:
            new_layer.bias.copy_(layer.bias)
    
    return new_layer


def prune_neurons(layer: nn.Linear, sparsity: float, method: str = 'l1') -> Tuple[nn.Linear, torch.Tensor]:
    """
    Удалить neurons с наименьшим importance.
    
    Args:
        layer: слой для pruning
        sparsity: доля нейронов для удаления (0.5 = удалить 50%)
        method: метод вычисления importance
    
    Returns:
        (pruned_layer, keep_indices)
    """
    importance = compute_neuron_importance(layer, method)
    k = int(layer.out_features * (1 - sparsity))
    k = max(1, k)
    
    keep_indices = torch.topk(importance, k).indices
    keep_indices = keep_indices.sort().values
    
    pruned_layer = prune_layer(layer, keep_indices)
    
    return pruned_layer, keep_indices


class PrunedModel(nn.Module):
    """
    Модель после structured pruning.
    
    Хранит информацию о том какие нейроны были удалены.
    """

    def __init__(self, input_size: int,
                 hidden_sizes: List[int],
                 num_classes: int,
                 layer_type: type = nn.Linear):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.layer_type = layer_type

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(layer_type(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(layer_type(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_architecture_str(self) -> str:
        sizes = [self.input_size] + self.hidden_sizes + [self.num_classes]
        return "→".join(map(str, sizes))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def structured_prune_model(model: nn.Module, 
                          sparsity: float = 0.5,
                          method: str = 'l1') -> PrunedModel:
    """
    Применить structured pruning к модели.
    
    Удаляет нейроны из hidden layers на основе importance.
    
    Args:
        model: исходная модель с атрибутами input_size, hidden_sizes, num_classes
        sparsity: доля нейронов для удаления (0.5 = удалить 50%)
        method: метод вычисления importance
    
    Returns:
        PrunedModel с уменьшенным числом параметров
    """
    if not (hasattr(model, 'input_size') and hasattr(model, 'hidden_sizes')):
        raise ValueError("Model must have input_size and hidden_sizes attributes")
    
    from baseline_model import OverflowAwareLinear
    
    linear_layers = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, OverflowAwareLinear)):
            linear_layers.append(module)
    
    if len(linear_layers) < 2:
        raise ValueError("Model must have at least 2 linear layers")
    
    new_hidden_sizes = []
    pruned_layers = []
    keep_indices_list = []
    
    for i, layer in enumerate(linear_layers[:-1]):
        if i == 0:
            pruned_layer, keep_indices = prune_neurons(layer, sparsity, method)
            pruned_layers.append(pruned_layer)
            keep_indices_list.append(keep_indices)
            new_hidden_sizes.append(pruned_layer.out_features)
        else:
            layer_updated = prune_next_layer_inputs(layer, keep_indices_list[-1])
            pruned_layer, keep_indices = prune_neurons(layer_updated, sparsity, method)
            pruned_layers.append(pruned_layer)
            keep_indices_list.append(keep_indices)
            new_hidden_sizes.append(pruned_layer.out_features)
    
    output_layer = prune_next_layer_inputs(linear_layers[-1], keep_indices_list[-1])
    pruned_layers.append(output_layer)
    
    layer_type = type(linear_layers[0])
    pruned_model = PrunedModel(
        model.input_size,
        new_hidden_sizes,
        model.num_classes,
        layer_type=layer_type
    )
    
    from baseline_model import OverflowAwareLinear
    
    layer_idx = 0
    for module in pruned_model.network:
        if isinstance(module, (nn.Linear, OverflowAwareLinear)):
            with torch.no_grad():
                module.weight.copy_(pruned_layers[layer_idx].weight)
                if module.bias is not None:
                    module.bias.copy_(pruned_layers[layer_idx].bias)
            layer_idx += 1
    
    return pruned_model


def get_pruning_stats(original_model: nn.Module, pruned_model: nn.Module) -> Dict[str, float]:
    """
    Статистика pruning: сколько параметров удалено.
    
    Args:
        original_model: исходная модель
        pruned_model: pruned модель
    
    Returns:
        Словарь со статистикой
    """
    orig_params = sum(p.numel() for p in original_model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    return {
        'original_params': orig_params,
        'pruned_params': pruned_params,
        'removed_params': orig_params - pruned_params,
        'compression_ratio': orig_params / pruned_params if pruned_params > 0 else 0,
        'sparsity': (orig_params - pruned_params) / orig_params if orig_params > 0 else 0
    }


def progressive_prune(model: nn.Module,
                     target_sparsity: float = 0.5,
                     num_steps: int = 5,
                     method: str = 'l1') -> List[PrunedModel]:
    """
    Progressive pruning: постепенное удаление нейронов.
    
    Вместо удаления 50% сразу, удаляем по 10% за шаг.
    Это может помочь модели лучше адаптироваться.
    
    Args:
        model: исходная модель
        target_sparsity: финальная sparsity
        num_steps: количество шагов
        method: метод вычисления importance
    
    Returns:
        Список моделей после каждого шага pruning
    """
    models = []
    current_model = model
    sparsity_per_step = target_sparsity / num_steps
    
    for step in range(num_steps):
        pruned = structured_prune_model(current_model, sparsity_per_step, method)
        models.append(pruned)
        current_model = pruned
    
    return models
