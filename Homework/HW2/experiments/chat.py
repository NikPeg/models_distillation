#!/usr/bin/env python3
"""
Интерактивный чат с обученной моделью tinychat.

Использование:
    python3 chat.py --model ../results/baseline_real.pt
    python3 chat.py --model ../results/quick_grid/run_04.pt
"""

import torch
import numpy as np
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from baseline_model import BaselineModel
from data_loader import TrigramEncoder, ContextEncoder


class ChatBot:
    """Чат-бот на основе обученной модели."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Загрузка модели
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = BaselineModel(
            input_size=checkpoint['architecture']['input_size'],
            hidden_sizes=checkpoint['architecture']['hidden_sizes'],
            num_classes=checkpoint['architecture']['num_classes']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Charset и encoders
        self.charset = checkpoint['charset']
        self.char_to_idx = {c: i for i, c in enumerate(self.charset)}
        self.idx_to_char = {i: c for i, c in enumerate(self.charset)}
        
        self.query_encoder = TrigramEncoder(num_buckets=128)
        self.context_encoder = ContextEncoder(num_buckets=128, context_len=8)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Architecture: {checkpoint['architecture']}")
        print(f"  Charset: {self.charset}")
        if 'val_acc' in checkpoint:
            print(f"  Validation accuracy: {checkpoint['val_acc']:.2%}")
        print()
    
    def predict_next_char(self, query: str, context: str = "") -> tuple[str, float]:
        """
        Предсказать следующий символ.
        
        Args:
            query: входной запрос пользователя
            context: контекст (уже сгенерированные символы ответа)
        
        Returns:
            (predicted_char, confidence)
        """
        # Encode input
        query_vec = self.query_encoder.encode(query)
        context_vec = self.context_encoder.encode(context)
        input_vec = np.concatenate([query_vec, context_vec])
        
        # Predict
        with torch.no_grad():
            x = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x, quant_temp=1.0)
            probs = torch.softmax(logits, dim=-1)
            
            top_prob, top_idx = torch.max(probs[0], dim=0)
            predicted_char = self.idx_to_char[top_idx.item()]
            confidence = top_prob.item()
        
        return predicted_char, confidence
    
    def generate_response(self, query: str, max_length: int = 20, 
                         temperature: float = 1.0, top_k: int = 5) -> str:
        """
        Сгенерировать полный ответ (autoregressive generation).
        
        Args:
            query: запрос пользователя
            max_length: максимальная длина ответа
            temperature: температура сэмплирования (1.0 = без изменений)
            top_k: брать топ-k предсказаний для разнообразия
        
        Returns:
            Сгенерированный ответ
        """
        response = ""
        context = ""
        
        for _ in range(max_length):
            # Encode input
            query_vec = self.query_encoder.encode(query)
            context_vec = self.context_encoder.encode(context)
            input_vec = np.concatenate([query_vec, context_vec])
            
            # Predict with temperature
            with torch.no_grad():
                x = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(x, quant_temp=1.0)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                probs = torch.softmax(logits, dim=-1)
                
                # Top-k sampling для разнообразия
                if top_k > 1:
                    top_probs, top_indices = torch.topk(probs[0], k=top_k)
                    top_probs = top_probs / top_probs.sum()  # Renormalize
                    
                    # Sample from top-k
                    sampled_idx = torch.multinomial(top_probs, 1).item()
                    char_idx = top_indices[sampled_idx].item()
                else:
                    # Greedy (argmax)
                    char_idx = torch.argmax(probs[0]).item()
                
                predicted_char = self.idx_to_char[char_idx]
            
            # Остановка на EOS или пробеле в конце
            if predicted_char == '\x00':  # EOS
                break
            
            response += predicted_char
            context = response[-8:]  # Последние 8 символов для context
            
            # Остановка если нашли естественное завершение
            if len(response) > 3 and response[-1] in [' ', '!', '?']:
                # Проверяем, что это не начало ответа
                if len(response) > 5:
                    break
        
        return response.strip()
    
    def chat(self):
        """Запустить интерактивный чат."""
        print("="*70)
        print("TINYCHAT - Interactive Chat with Neural Network")
        print("="*70)
        print()
        print("Commands:")
        print("  /help    - Show this help")
        print("  /quit    - Exit chat")
        print("  /stats   - Show model statistics")
        print("  Ctrl+C   - Exit chat")
        print()
        print("Type your message and press Enter to chat!")
        print("-"*70)
        print()
        
        conversation_history = []
        
        try:
            while True:
                # Получить ввод пользователя
                try:
                    user_input = input("You: ").strip()
                except EOFError:
                    print("\n\nGoodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                # Обработка команд
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("\nGoodbye! 👋")
                        break
                    elif user_input == '/help':
                        print("\nCommands:")
                        print("  /help    - Show this help")
                        print("  /quit    - Exit chat")
                        print("  /stats   - Show conversation stats")
                        print()
                        continue
                    elif user_input == '/stats':
                        print(f"\nConversation statistics:")
                        print(f"  Messages: {len(conversation_history)}")
                        print(f"  Model: {self.model.get_architecture_str()}")
                        print(f"  Parameters: {self.model.count_parameters():,}")
                        print()
                        continue
                    else:
                        print(f"Unknown command: {user_input}")
                        print("Type /help for available commands")
                        print()
                        continue
                
                # Генерация ответа
                response = self.generate_response(
                    user_input,
                    max_length=15,
                    temperature=0.8,  # Немного разнообразия
                    top_k=3           # Top-3 sampling
                )
                
                # Если ответ пустой, попробуем greedy
                if not response:
                    response = self.generate_response(
                        user_input,
                        max_length=10,
                        temperature=1.0,
                        top_k=1
                    )
                
                # Если всё равно пустой, берём просто первый символ
                if not response:
                    char, conf = self.predict_next_char(user_input)
                    response = char
                
                print(f"Bot: {response}")
                print()
                
                # Сохранить в историю
                conversation_history.append({
                    'user': user_input,
                    'bot': response
                })
        
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye! 👋")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Interactive chat with tinychat model')
    
    parser.add_argument('--model', type=str, default='../results/quick_grid/run_04.pt',
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run on')
    
    args = parser.parse_args()
    
    # Проверка существования модели
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print(f"\nAvailable models:")
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        if os.path.exists(results_dir):
            for root, dirs, files in os.walk(results_dir):
                for f in files:
                    if f.endswith('.pt'):
                        rel_path = os.path.relpath(os.path.join(root, f), 
                                                  os.path.dirname(__file__))
                        print(f"  {rel_path}")
        sys.exit(1)
    
    # Создать чат-бот
    bot = ChatBot(args.model, device=args.device)
    
    # Запустить чат
    bot.chat()


if __name__ == '__main__':
    main()
