#!/usr/bin/env python3
"""
Обучение baseline модели с 2-bit QAT.
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

from baseline_model import create_baseline_model, create_micro_model
from utils import accuracy, save_model_npz, EarlyStopping


def generate_synthetic_data(num_samples: int = 1000, 
                           input_size: int = 256,
                           num_classes: int = 40,
                           seed: int = 42):
    """
    Генерация синтетических данных для экспериментов.
    
    В реальной работе здесь должна быть загрузка из simple_chat.txt,
    но для демонстрации используем синтетику.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


def train_epoch(model, dataloader, criterion, optimizer, device, quant_temp=1.0):
    """Один epoch обучения."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_x, quant_temp=quant_temp)
        
        task_loss = criterion(outputs, batch_y)
        
        quant_loss = model.get_total_quantization_loss()
        
        loss = task_loss + 0.01 * quant_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += accuracy(outputs, batch_y)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, criterion, device, quant_temp=1.0):
    """Оценка на валидации."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x, quant_temp=quant_temp)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            total_acc += accuracy(outputs, batch_y)
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def train_baseline(args):
    """Основная функция обучения baseline модели."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n[1] Generating synthetic data...")
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
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    
    print("\n[2] Creating model...")
    if args.model_type == 'baseline':
        model = create_baseline_model(
            input_size=args.input_size,
            hidden_sizes=args.hidden_sizes,
            num_classes=args.num_classes
        )
    else:
        model = create_micro_model(
            input_size=args.input_size,
            hidden_sizes=args.hidden_sizes,
            num_classes=args.num_classes
        )
    
    model = model.to(device)
    print(f"  Architecture: {model.get_architecture_str()}")
    print(f"  Parameters: {model.count_parameters():,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    early_stopping = EarlyStopping(patience=args.patience)
    
    print(f"\n[3] Training for {args.epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        quant_temp = min(1.0, epoch / args.warmup_epochs)
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, quant_temp
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, quant_temp
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                  f"quant_temp={quant_temp:.2f}")
        
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n[4] Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {args.output}")
    
    model.load_state_dict(torch.load(args.output))
    
    npz_path = args.output.replace('.pt', '.npz')
    save_model_npz(model, npz_path)
    
    overflow_stats = model.get_overflow_stats()
    print(f"\n[5] Overflow statistics:")
    for layer, risk in overflow_stats.items():
        print(f"  {layer}: {risk:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train baseline 2-bit QAT model')
    
    parser.add_argument('--model-type', type=str, default='baseline', 
                       choices=['baseline', 'micro'],
                       help='Model type to train')
    parser.add_argument('--input-size', type=int, default=256,
                       help='Input size')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 192, 128],
                       help='Hidden layer sizes')
    parser.add_argument('--num-classes', type=int, default=40,
                       help='Number of output classes')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=20,
                       help='Epochs for quantization warmup')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='../results/baseline_model.pt',
                       help='Output model path')
    
    args = parser.parse_args()
    
    train_baseline(args)


if __name__ == '__main__':
    main()
