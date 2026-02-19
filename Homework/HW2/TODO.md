# TODO: Переделка экспериментов с реальной моделью

## Проблема

Текущая реализация использует **синтетические данные** и **маленькую модель**, обученную с нуля:
- Архитектура: 64→32→16→10 (2.7K параметров)
- Данные: `generate_synthetic_data()` - случайный шум
- Результат: бессмысленные метрики accuracy (0.14-0.20 на шуме)

**Это неправильно**, потому что:
1. В репозитории есть **готовая обученная модель**: `z80ai/examples/tinychat/model.npz` (145 KB)
2. Есть **реальные данные**: `z80ai/examples/tinychat/training-data.txt.gz`
3. В HW1 я анализировал именно эту модель: 256→256→192→128→40 (144K параметров)
4. Цель HW2 — ускорить **именно её**, а не toy-модель

## Что нужно переделать

### 1. Загрузить baseline модель из z80ai

**Файл**: `z80ai/examples/tinychat/model.npz`

**Что делать**:
```python
# Загрузить веса
data = np.load('z80ai/examples/tinychat/model.npz')

# Создать BaselineModel с правильной архитектурой
baseline = BaselineModel(
    input_size=256,          # 128 query + 128 context buckets
    hidden_sizes=[256, 192, 128],
    num_classes=40           # charset
)

# Перенести веса из .npz в модель
for i, layer in enumerate(linear_layers):
    layer.weight = data[f'fc{i+1}_weight']
    layer.bias = data[f'fc{i+1}_bias']
```

**Проблема**: веса в .npz уже квантованные в 2-bit. Нужно понять формат и правильно их загрузить.

### 2. Подготовить реальные данные

**Файл**: `z80ai/examples/tinychat/training-data.txt.gz`

**Что делать**:
```bash
# Распаковать
gunzip -c z80ai/examples/tinychat/training-data.txt.gz > data/training-data.txt

# Изучить формат
head -20 data/training-data.txt
```

**Формат данных**: судя по `genpairs.py`, это пары (query → response) для chatbot.

**Нужно**:
- Понять формат входа (trigram encoding из HW1)
- Реализовать загрузку и preprocessing
- Создать правильный DataLoader

### 3. Использовать правильный input encoding

**Из HW1/report.md**:
- Input: 256 dimensions = 128 trigram buckets (query) + 128 context buckets
- Trigram hash encoding: текст → триграммы → hash в buckets
- Нужен encoder из `z80ai/`

**Что делать**:
```python
# Найти в z80ai код для encoding
# Скорее всего в feedme.py или libz80.py

from z80ai.feedme import TrigramEncoder, ContextEncoder

query_encoder = TrigramEncoder(num_buckets=128)
context_encoder = ContextEncoder(num_buckets=128, context_len=8)
```

### 4. Переобучить с правильными данными

**Pipeline**:
1. Загрузить baseline из .npz
2. (Опционально) Fine-tune на части данных для warmup
3. Apply pruning → архитектура 256→128→96→64→40
4. Fine-tune pruned model
5. Convert to binary (1-bit)
6. Fine-tune binary model

### 5. Измерить на реальных метриках

**Quality metrics**:
- Character prediction accuracy
- Perplexity
- Примеры: "hello" → "HI", "are you a robot" → "YES"

**Performance metrics** (те же, что сейчас):
- Flash/SRAM usage
- Latency simulation

## План действий

### Фаза 1: Исследование (1-2 часа)

- [ ] Изучить `z80ai/feedme.py` — как обучалась модель
- [ ] Изучить `z80ai/libz80.py` — формат экспорта
- [ ] Изучить `z80ai/examples/tinychat/genpairs.py` — формат данных
- [ ] Понять структуру model.npz

### Фаза 2: Адаптация кода (2-3 часа)

- [ ] Добавить в `code/utils.py`: загрузку .npz модели
- [ ] Добавить в `code/data_loader.py`: preprocessing реальных данных
- [ ] Адаптировать trigram encoding из z80ai
- [ ] Создать `experiments/load_baseline.py` для загрузки готовой модели

### Фаза 3: Переобучение (1-2 часа)

- [ ] Запустить baseline на test set (без дообучения)
- [ ] Apply pruning к baseline
- [ ] Fine-tune pruned (50 epochs)
- [ ] Convert to binary
- [ ] Fine-tune binary (50 epochs)

### Фаза 4: Benchmark (30 мин)

- [ ] Собрать метрики на реальных данных
- [ ] Примеры inference: "hello" → ?
- [ ] Сравнить baseline vs pruned vs binary

### Фаза 5: Отчёт (1 час)

- [ ] Переписать `report.md` с реальными результатами
- [ ] Добавить примеры работы модели
- [ ] Честный анализ quality loss

## Альтернативный подход (если сложно)

Если загрузка готовой модели окажется слишком сложной:

### Вариант А: Обучить с нуля на реальных данных
- Использовать `training-data.txt.gz`
- Обучить baseline 256→256→192→128→40
- Применить методы оптимизации
- **Минус**: долго обучать (300+ epochs)

### Вариант Б: Micro-модель на реальных данных
- Оставить маленькую архитектуру 64→32→16→10
- Но использовать **реальные данные** из z80ai
- Показать методы на упрощённой версии задачи
- **Плюс**: быстрее, проще
- **Минус**: не совсем baseline из HW1

## Что оставить как есть

**Код методов оптимизации** — всё работает правильно:
- ✅ `quantization.py` — 1-bit binary quantization
- ✅ `pruning.py` — structured pruning
- ✅ `combined_optimization.py` — pipeline
- ✅ `inference_simulator.py` — Arduino simulation

Нужно только **заменить данные** и **baseline модель**.

## Оценка времени

- **Вариант с загрузкой готовой модели**: 4-6 часов
- **Вариант с обучением с нуля**: 8-10 часов (включая время обучения)
- **Вариант Б (micro + реальные данные)**: 3-4 часа

## Текущий статус

✅ **Сделано**:
- Реализация всех методов (pruning, quantization, combined)
- Arduino inference simulator
- Pipeline обучения и benchmark
- Визуализация

❌ **Нужно переделать**:
- Использовать baseline из `z80ai/examples/tinychat/model.npz`
- Загрузить реальные данные из `training-data.txt.gz`
- Переобучить с правильными входами
- Обновить отчёт с реальными метриками

## Решение

**Рекомендую**: начать с исследования z80ai кода и попытки загрузить model.npz. Если окажется слишком сложно — переключиться на Вариант Б (micro-модель на реальных данных).
