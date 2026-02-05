#!/usr/bin/env python3
"""
Подготовка упрощённых training data для micro-модели Arduino.

Вместо 40 символов используем 10 команд с короткими ответами.
"""

SIMPLE_PAIRS = [
    # Приветствия
    "hi|HI",
    "hello|HI",
    "hey|HI",
    "hi there|HI",
    
    # Yes/No
    "yes|YES",
    "yeah|YES",
    "yep|YES",
    "sure|YES",
    "ok|OK",
    "okay|OK",
    "no|NO",
    "nope|NO",
    "nah|NO",
    
    # Вопросы
    "why|WHY",
    "what|WHAT",
    "who|WHO",
    "how|HELP",
    "help|HELP",
    "help me|HELP",
    
    # Состояния
    "maybe|MAYBE",
    "unsure|MAYBE",
    "not sure|MAYBE",
    "dunno|MAYBE",
    
    # Прощания
    "bye|BYE",
    "goodbye|BYE",
    "exit|BYE",
    "quit|BYE",
    
    # Проверки
    "are you real|MAYBE",
    "are you human|NO",
    "are you robot|YES",
    "are you ai|YES",
    "do you think|MAYBE",
    "do you know|MAYBE",
    
    # Простые ответы
    "good|OK",
    "bad|NO",
    "cool|YES",
    "nice|YES",
]

def generate_variations(pairs, output_file):
    """Генерирует больше вариаций для каждой пары."""
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(pair + '\n')
            
            # Добавляем вариации с пунктуацией
            query, response = pair.split('|')
            f.write(f"{query}?|{response}\n")
            f.write(f"{query}.|{response}\n")
            f.write(f"{query}!|{response}\n")
            
            # Вариации с "I"
            if not query.startswith("are "):
                f.write(f"i {query}|{response}\n")
                
    print(f"Сгенерировано {len(pairs) * 5} пар в {output_file}")

if __name__ == '__main__':
    output = 'micro_training_data.txt'
    generate_variations(SIMPLE_PAIRS, output)
    
    # Статистика по ответам
    responses = {}
    for pair in SIMPLE_PAIRS:
        _, resp = pair.split('|')
        responses[resp] = responses.get(resp, 0) + 1
    
    print("\nРаспределение ответов:")
    for resp, count in sorted(responses.items(), key=lambda x: -x[1]):
        print(f"  {resp}: {count}")
    
    unique = len(responses)
    print(f"\nУникальных команд: {unique}")
    print("✓ Подходит для output layer размером 10-15")
