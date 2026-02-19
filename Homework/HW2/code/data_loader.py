#!/usr/bin/env python3
"""
Загрузка и preprocessing реальных данных из z80ai/examples/tinychat.

Данные в формате: query|response
Входы: trigram encoding + context encoding = 256 dimensions
Выходы: next character index (40 classes)
"""

import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import gzip


class TrigramEncoder:
    """Encode text into trigram hash buckets (копия из z80ai/feedme.py)."""

    def __init__(self, num_buckets: int = 128):
        self.num_buckets = num_buckets

    def _hash_trigram(self, trigram: str) -> int:
        """Hash a trigram to a bucket index."""
        h = 0
        for c in trigram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, text: str) -> np.ndarray:
        """Encode text into bucket counts (raw counts, Z80-compatible)."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)
        text = text.lower()
        text = ' ' + text + ' '

        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            bucket = self._hash_trigram(trigram)
            vec[bucket] += 1.0

        return vec


class ContextEncoder:
    """Encode recent output context (копия из z80ai/feedme.py)."""

    def __init__(self, num_buckets: int = 128, context_len: int = 8):
        self.num_buckets = num_buckets
        self.context_len = context_len

    def _hash_trigram(self, trigram: str) -> int:
        h = 0
        for c in trigram:
            h = (h * 31 + ord(c)) & 0xFFFF
        return h % self.num_buckets

    def encode(self, context: str) -> np.ndarray:
        """Encode context string."""
        vec = np.zeros(self.num_buckets, dtype=np.float32)
        context = context[-self.context_len:].lower()
        context = ' ' + context + ' '

        for i in range(len(context) - 2):
            trigram = context[i:i+3]
            bucket = self._hash_trigram(trigram)
            vec[bucket] += 1.0

        return vec


def load_training_pairs(filepath: str) -> List[Tuple[str, str]]:
    """
    Загрузить пары (query, response) из файла.
    
    Формат: query|response
    Пример: hey|HI
    """
    pairs = []
    
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) == 2:
                    query, response = parts
                    pairs.append((query.strip(), response.strip()))
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) == 2:
                    query, response = parts
                    pairs.append((query.strip(), response.strip()))
    
    return pairs


def build_charset(pairs: List[Tuple[str, str]]) -> str:
    """
    Построить charset из данных.
    
    Returns:
        charset: строка с уникальными символами
    """
    chars = set()
    for query, response in pairs:
        chars.update(response.upper())
    
    letters = sorted(c for c in chars if c.isalpha())
    digits = sorted(c for c in chars if c.isdigit())
    space = [' '] if ' ' in chars else []
    punct = sorted(c for c in chars if not c.isalnum() and c != ' ')
    
    return ''.join(space + letters + digits + punct)


class ChatDataset(Dataset):
    """
    Dataset для обучения модели на реальных данных.
    
    Каждый семпл:
    - X: [query_trigrams (128), context_trigrams (128)] = 256 dims
    - y: индекс первого символа ответа
    """
    
    def __init__(self, pairs: List[Tuple[str, str]], charset: str):
        self.pairs = pairs
        self.charset = charset
        self.char_to_idx = {c: i for i, c in enumerate(charset)}
        
        self.query_encoder = TrigramEncoder(num_buckets=128)
        self.context_encoder = ContextEncoder(num_buckets=128, context_len=8)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        query, response = self.pairs[idx]
        
        query_vec = self.query_encoder.encode(query)
        
        context_vec = np.zeros(128, dtype=np.float32)
        
        input_vec = np.concatenate([query_vec, context_vec])
        
        target_char = response[0].upper() if response else ' '
        target_idx = self.char_to_idx.get(target_char, 0)
        
        return torch.tensor(input_vec, dtype=torch.float32), torch.tensor(target_idx, dtype=torch.long)


def create_dataloaders(data_path: str, 
                       batch_size: int = 32,
                       train_split: float = 0.8,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, str]:
    """
    Создать train и val dataloaders из файла.
    
    Returns:
        train_loader, val_loader, charset
    """
    pairs = load_training_pairs(data_path)
    print(f"Loaded {len(pairs)} training pairs")
    
    charset = build_charset(pairs)
    print(f"Charset ({len(charset)} chars): {charset}")
    
    np.random.seed(seed)
    indices = np.random.permutation(len(pairs))
    split_idx = int(len(pairs) * train_split)
    
    train_pairs = [pairs[i] for i in indices[:split_idx]]
    val_pairs = [pairs[i] for i in indices[split_idx:]]
    
    train_dataset = ChatDataset(train_pairs, charset)
    val_dataset = ChatDataset(val_pairs, charset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, charset


def load_pretrained_weights(model_path: str):
    """
    Загрузить веса из model.npz.
    
    Returns:
        params: словарь с весами
        architecture: dict с архитектурой
        charset: строка charset
    """
    import json
    
    data = np.load(model_path)
    
    arch = json.loads(bytes(data['_architecture']).decode('utf-8'))
    charset = bytes(data['_charset']).decode('utf-8')
    
    params = {k: data[k] for k in data.files if not k.startswith('_')}
    
    return params, arch, charset
