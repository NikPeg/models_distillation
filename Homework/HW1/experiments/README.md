# Эксперименты: портирование Z80-μLM на Arduino

## Файлы

- `plan.md` — детальный план экспериментов
- `prepare_micro_data.py` — генерация упрощённых training data
- `export_to_arduino.py` — экспорт модели в C arrays
- `arduino_template.ino` — template кода для Arduino

## Быстрый старт

### 1. Подготовка данных

```bash
python3 prepare_micro_data.py
# Создаст micro_training_data.txt
```

### 2. Обучение micro-модели

```bash
cd ../../z80ai

# Модифицировать feedme.py:
# - hidden_sizes = [32, 16]
# - query_encoder = TrigramEncoder(num_buckets=32)
# - context_encoder = ContextEncoder(num_buckets=32, context_len=4)

cat ../Homework/HW1/experiments/micro_training_data.txt | \
    python3 feedme.py --epochs 300 --chat

# Экспортировать
python3 exportmodel.py --model model.pt --output micro_model.npz
```

### 3. Экспорт для Arduino

```bash
cd ../Homework/HW1/experiments
python3 export_to_arduino.py \
    --model ../../../z80ai/micro_model.npz \
    --output weights.h
```

### 4. Загрузка на Arduino

1. Скопировать `arduino_template.ino` в папку проекта Arduino
2. Скопировать `weights.h` туда же
3. Открыть в Arduino IDE
4. Залить на плату (UNO или LilyPad)
5. Открыть Serial Monitor (9600 baud)

## Изменения в z80ai/feedme.py

Для обучения micro-модели нужно изменить конфигурацию:

```python
# Около строки 380
# Было:
# hidden_sizes = [256, 192, 128]
# query_encoder = TrigramEncoder(num_buckets=128)
# context_encoder = ContextEncoder(num_buckets=128, context_len=8)

# Нужно:
hidden_sizes = [32, 16]
query_encoder = TrigramEncoder(num_buckets=32)
context_encoder = ContextEncoder(num_buckets=32, context_len=4)
```

## Ожидаемые результаты

### Модель
- Параметры: ~3K (vs 144K original)
- Flash: ~800 bytes (weights + biases)
- SRAM: ~600 bytes (runtime)

### Производительность
- Latency: 10-20ms @ 16MHz (UNO)
- Latency: 20-40ms @ 8MHz (LilyPad)
- Accuracy: 50-70% (на упрощённом test set)

## Troubleshooting

**Проблема**: модель не влезает в SRAM
- Уменьшить `hidden_sizes`: [16, 8]
- Уменьшить `num_buckets`: 16

**Проблема**: слишком долгий inference
- Включить compiler optimizations: `-O3`
- Loop unrolling для малых слоёв
- Уменьшить модель

**Проблема**: низкая accuracy
- Больше training data
- Увеличить epochs: 500-1000
- Проверить class balance
