#!/usr/bin/env python3
"""
Загрузка baseline модели из z80ai/examples/tinychat/model.npz
и проверка её работы на реальных данных.
"""

import torch
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from baseline_model import BaselineModel
from data_loader import create_dataloaders, load_pretrained_weights
from utils import accuracy


def load_baseline_from_npz(model_path: str) -> BaselineModel:
    """
    Загрузить baseline модель из .npz файла.
    
    ВНИМАНИЕ: веса в .npz уже квантованные в int8 (2-bit).
    Нужно конвертировать их обратно в float для PyTorch.
    """
    params, arch, charset = load_pretrained_weights(model_path)
    
    print(f"Architecture: {arch}")
    print(f"Charset ({len(charset)}): {charset}")
    
    model = BaselineModel(
        input_size=arch['input_size'],
        hidden_sizes=arch['hidden_sizes'],
        num_classes=arch['num_classes']
    )
    
    linear_layers = [m for m in model.network if hasattr(m, 'weight')]
    
    for i, layer in enumerate(linear_layers):
        weight_key = f'fc{i+1}_weight'
        bias_key = f'fc{i+1}_bias'
        
        if weight_key in params and bias_key in params:
            w = params[weight_key].astype(np.float32)
            b = params[bias_key].astype(np.float32)
            
            layer.weight.data = torch.from_numpy(w)
            layer.bias.data = torch.from_numpy(b)
            
            print(f"Loaded {weight_key}: {w.shape}, range=[{w.min():.2f}, {w.max():.2f}]")
    
    return model, charset


def evaluate_model(model, dataloader, device):
    """Оценка модели на данных."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x, quant_temp=1.0)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            total_acc += accuracy(outputs, batch_y)
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def test_inference(model, charset, device):
    """Протестировать inference на примерах."""
    from data_loader import TrigramEncoder, ContextEncoder
    
    query_encoder = TrigramEncoder(num_buckets=128)
    context_encoder = ContextEncoder(num_buckets=128)
    
    test_queries = [
        "hello",
        "how are you",
        "are you a robot",
        "what is your name",
        "bye"
    ]
    
    char_to_idx = {c: i for i, c in enumerate(charset)}
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
            
            top5_probs, top5_idx = torch.topk(probs[0], k=5)
            
            predictions = [(idx_to_char[idx.item()], prob.item()) 
                          for prob, idx in zip(top5_probs, top5_idx)]
            
            print(f"\n  Query: '{query}'")
            print(f"  Top-5: {predictions}")


def main():
    parser = argparse.ArgumentParser(description='Load baseline model from z80ai')
    parser.add_argument('--model', type=str, 
                       default='../../z80ai/examples/tinychat/model.npz',
                       help='Path to model.npz')
    parser.add_argument('--data', type=str,
                       default='../../z80ai/examples/tinychat/training-data.txt.gz',
                       help='Path to training data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=str, default='../results/baseline_loaded.pt',
                       help='Save converted model')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("[1] Loading baseline model from z80ai...")
    model, npz_charset = load_baseline_from_npz(args.model)
    model = model.to(device)
    
    print(f"\n[2] Loading data...")
    train_loader, val_loader, data_charset = create_dataloaders(
        args.data, 
        batch_size=args.batch_size,
        train_split=0.8
    )
    
    if npz_charset != data_charset:
        print(f"\nWARNING: charset mismatch!")
        print(f"  Model: {npz_charset}")
        print(f"  Data:  {data_charset}")
    
    print(f"\n[3] Evaluating on validation set...")
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    print(f"  Val loss: {val_loss:.4f}")
    print(f"  Val accuracy: {val_acc:.4f}")
    
    print(f"\n[4] Testing inference...")
    test_inference(model, npz_charset, device)
    
    print(f"\n[5] Saving converted model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': {
            'input_size': model.input_size,
            'hidden_sizes': model.hidden_sizes,
            'num_classes': model.num_classes
        },
        'charset': npz_charset
    }, args.output)
    print(f"  Saved to: {args.output}")
    
    print("\n✓ Baseline model loaded successfully!")
    print(f"  Architecture: {model.get_architecture_str()}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Validation accuracy: {val_acc:.4f}")


if __name__ == '__main__':
    main()
