#!/usr/bin/env python3
"""
Обучение оптимизированных моделей (pruned + binary).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from baseline_model import BaselineModel
from quantization import BinaryModel
from pruning import structured_prune_model
from combined_optimization import CombinedOptimizationPipeline
from utils import accuracy, save_model_npz, EarlyStopping
from train_baseline import generate_synthetic_data, evaluate


def train_binary_epoch(model, dataloader, criterion, optimizer, device):
    """Один epoch обучения binary модели."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        task_loss = criterion(outputs, batch_y)
        
        binary_loss = model.get_total_binarization_loss()
        loss = task_loss + 0.01 * binary_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += accuracy(outputs, batch_y)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate_binary(model, dataloader, criterion, device):
    """Оценка binary модели."""
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


def train_optimized(args):
    """Обучение оптимизированных моделей."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n[1] Loading baseline model...")
    baseline = BaselineModel(
        input_size=args.input_size,
        hidden_sizes=args.hidden_sizes,
        num_classes=args.num_classes
    )
    baseline.load_state_dict(torch.load(args.baseline_path))
    baseline = baseline.to(device)
    print(f"  Loaded from: {args.baseline_path}")
    
    print("\n[2] Generating data...")
    X, y = generate_synthetic_data(
        num_samples=args.num_samples,
        input_size=args.input_size,
        num_classes=args.num_classes,
        seed=args.seed
    )
    
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\n[3] Creating optimized models...")
    pipeline = CombinedOptimizationPipeline(baseline)
    
    pruned = pipeline.apply_pruning(sparsity=args.sparsity, method='l1')
    pruned = pruned.to(device)
    
    print("\n[4] Fine-tuning pruned model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pruned.parameters(), lr=args.lr)
    
    best_val_acc = 0.0
    for epoch in range(args.finetune_epochs):
        pruned.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = pruned(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        val_loss, val_acc = evaluate_binary(pruned, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(pruned.state_dict(), args.pruned_output)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.finetune_epochs}: "
                  f"val_acc={val_acc:.4f}")
    
    print(f"  Best pruned accuracy: {best_val_acc:.4f}")
    
    print("\n[5] Converting to binary...")
    binary = pipeline.convert_to_binary()
    binary = binary.to(device)
    
    print("\n[6] Fine-tuning binary model...")
    optimizer = optim.Adam(binary.parameters(), lr=args.lr * 0.5)
    
    best_val_acc = 0.0
    for epoch in range(args.finetune_epochs):
        train_loss, train_acc = train_binary_epoch(
            binary, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate_binary(
            binary, val_loader, criterion, device
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(binary.state_dict(), args.binary_output)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.finetune_epochs}: "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    print(f"  Best binary accuracy: {best_val_acc:.4f}")
    
    print("\n[7] Summary:")
    summary = pipeline.get_summary()
    for name, stats in summary.items():
        print(f"\n{name}:")
        print(f"  Architecture: {stats['architecture']}")
        print(f"  Parameters: {stats['parameters']:,}")
        print(f"  Size: {stats['size_kb']:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Train optimized models')
    
    parser.add_argument('--baseline-path', type=str, required=True,
                       help='Path to baseline model')
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 192, 128])
    parser.add_argument('--num-classes', type=int, default=40)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--finetune-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--sparsity', type=float, default=0.5,
                       help='Pruning sparsity (0.5 = remove 50%)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pruned-output', type=str, default='../results/pruned_model.pt')
    parser.add_argument('--binary-output', type=str, default='../results/binary_model.pt')
    
    args = parser.parse_args()
    
    train_optimized(args)


if __name__ == '__main__':
    main()
