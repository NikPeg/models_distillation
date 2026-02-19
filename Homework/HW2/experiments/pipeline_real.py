#!/usr/bin/env python3
"""
Полный цикл оптимизации: baseline → pruned → binary
на реальных данных из z80ai.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from baseline_model import BaselineModel
from data_loader import create_dataloaders
from utils import accuracy, save_model_npz, EarlyStopping
from pruning import structured_prune_model
from quantization import convert_to_binary


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Один epoch обучения."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += accuracy(outputs, batch_y)
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def evaluate(model, dataloader, criterion, device):
    """Оценка на валидации."""
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


def fine_tune(model, train_loader, val_loader, criterion, device, 
              epochs=50, lr=0.0001, patience=10, desc="Model"):
    """Fine-tune модели после модификации."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)
    
    best_val_acc = 0.0
    
    print(f"\n  Fine-tuning {desc} for {epochs} epochs...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        if early_stopping(val_loss):
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Full optimization pipeline on real data')
    
    parser.add_argument('--baseline', type=str, default='../results/baseline_real.pt',
                       help='Path to baseline model')
    parser.add_argument('--data', type=str,
                       default='../../../z80ai/examples/tinychat/training-data.txt.gz')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--fine-tune-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='../results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n[1] Loading data...")
    train_loader, val_loader, charset = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        train_split=0.8,
        seed=args.seed
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n[2] Loading baseline model...")
    checkpoint = torch.load(args.baseline)
    
    baseline = BaselineModel(
        input_size=checkpoint['architecture']['input_size'],
        hidden_sizes=checkpoint['architecture']['hidden_sizes'],
        num_classes=checkpoint['architecture']['num_classes']
    )
    baseline.load_state_dict(checkpoint['model_state_dict'])
    baseline = baseline.to(device)
    
    baseline_acc = evaluate(baseline, val_loader, criterion, device)[1]
    baseline_params = baseline.count_parameters()
    
    print(f"  Baseline accuracy: {baseline_acc:.4f}")
    print(f"  Baseline parameters: {baseline_params:,}")
    
    print("\n[3] Applying structured pruning (50%)...")
    pruned = structured_prune_model(
        baseline,
        sparsity=0.5,
        method='l1'
    )
    pruned = pruned.to(device)
    
    pruned_acc_before = evaluate(pruned, val_loader, criterion, device)[1]
    print(f"  Pruned accuracy (before fine-tune): {pruned_acc_before:.4f}")
    
    pruned_acc = fine_tune(
        pruned, train_loader, val_loader, criterion, device,
        epochs=args.fine_tune_epochs,
        lr=args.lr,
        patience=args.patience,
        desc="Pruned"
    )
    
    pruned_params = pruned.count_parameters()
    print(f"  Pruned accuracy (after fine-tune): {pruned_acc:.4f}")
    print(f"  Pruned parameters: {pruned_params:,} ({pruned_params/baseline_params*100:.1f}%)")
    
    torch.save({
        'model_state_dict': pruned.state_dict(),
        'architecture': {
            'input_size': pruned.input_size,
            'hidden_sizes': pruned.hidden_sizes,
            'num_classes': pruned.num_classes
        },
        'charset': charset,
        'val_acc': pruned_acc
    }, os.path.join(args.output_dir, 'pruned_real.pt'))
    
    print("\n[4] Converting to binary (1-bit)...")
    binary = convert_to_binary(pruned)
    binary = binary.to(device)
    
    binary_acc_before = evaluate(binary, val_loader, criterion, device)[1]
    print(f"  Binary accuracy (before fine-tune): {binary_acc_before:.4f}")
    
    binary_acc = fine_tune(
        binary, train_loader, val_loader, criterion, device,
        epochs=args.fine_tune_epochs,
        lr=args.lr,
        patience=args.patience,
        desc="Binary"
    )
    
    print(f"  Binary accuracy (after fine-tune): {binary_acc:.4f}")
    
    torch.save({
        'model_state_dict': binary.state_dict(),
        'architecture': {
            'input_size': binary.input_size,
            'hidden_sizes': binary.hidden_sizes,
            'num_classes': binary.num_classes
        },
        'charset': charset,
        'val_acc': binary_acc
    }, os.path.join(args.output_dir, 'binary_real.pt'))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline:  {baseline_acc:.4f} acc, {baseline_params:,} params")
    print(f"Pruned:    {pruned_acc:.4f} acc, {pruned_params:,} params ({pruned_params/baseline_params*100:.1f}%)")
    print(f"Binary:    {binary_acc:.4f} acc, {pruned_params:,} params (1-bit)")
    print(f"\nAccuracy loss:")
    print(f"  Pruning: {(baseline_acc - pruned_acc)*100:.2f}%")
    print(f"  Binary:  {(baseline_acc - binary_acc)*100:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
