#!/usr/bin/env python3
"""
Обучение baseline модели с нуля на реальных данных.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from baseline_model import create_baseline_model
from data_loader import create_dataloaders
from utils import accuracy, save_model_npz, EarlyStopping


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


def test_inference(model, charset, device):
    """Протестировать inference."""
    from data_loader import TrigramEncoder, ContextEncoder
    import numpy as np
    
    query_encoder = TrigramEncoder(num_buckets=128)
    context_encoder = ContextEncoder(num_buckets=128)
    
    test_queries = [
        "hello",
        "how are you",
        "are you a robot",
        "bye"
    ]
    
    idx_to_char = {i: c for i, c in enumerate(charset)}
    
    print("\n=== Test Inference ===")
    model.eval()
    
    with torch.no_grad():
        for query in test_queries:
            query_vec = query_encoder.encode(query)
            context_vec = np.zeros(128, dtype=np.float32)
            input_vec = np.concatenate([query_vec, context_vec])
            
            x = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x, quant_temp=1.0)
            probs = torch.softmax(logits, dim=-1)
            
            top3_probs, top3_idx = torch.topk(probs[0], k=3)
            
            predictions = [(idx_to_char[idx.item()], prob.item()) 
                          for prob, idx in zip(top3_probs, top3_idx)]
            
            print(f"  '{query}' → {predictions[0][0]} ({predictions[0][1]:.3f})")


def main():
    parser = argparse.ArgumentParser(description='Train baseline model on real data')
    
    parser.add_argument('--data', type=str,
                       default='../../../z80ai/examples/tinychat/training-data.txt.gz',
                       help='Path to training data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup-epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='../results/baseline_real.pt')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n[1] Loading real data...")
    train_loader, val_loader, charset = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        train_split=0.8,
        seed=args.seed
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Charset: {charset}")
    
    print("\n[2] Creating model...")
    model = create_baseline_model(
        input_size=256,
        hidden_sizes=[256, 192, 128],
        num_classes=len(charset)
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'architecture': {
                    'input_size': model.input_size,
                    'hidden_sizes': model.hidden_sizes,
                    'num_classes': model.num_classes
                },
                'charset': charset,
                'epoch': epoch,
                'val_acc': val_acc
            }, args.output)
        
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
    
    checkpoint = torch.load(args.output)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_inference(model, charset, device)
    
    npz_path = args.output.replace('.pt', '.npz')
    save_model_npz(model, npz_path)
    
    overflow_stats = model.get_overflow_stats()
    print(f"\n[5] Overflow statistics:")
    for layer, risk in overflow_stats.items():
        print(f"  {layer}: {risk:.4f}")


if __name__ == '__main__':
    main()
