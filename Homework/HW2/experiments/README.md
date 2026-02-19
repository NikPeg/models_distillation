# Эксперименты HW2: Model Optimization

Эксперименты по оптимизации модели z80ai tinychat для Arduino UNO.

## Быстрый старт

### 1. Обучить baseline модель на реальных данных

```bash
python3 train_real.py --epochs 50 --output ../results/baseline_real.pt
```

**Результат**: baseline модель 256→256→192→128→39 (~145K параметров, accuracy ~30-40%)

### 2. Полный pipeline оптимизации

```bash
python3 pipeline_real.py --fine-tune-epochs 30
```

**Результат**:
- `baseline_real.pt` - исходная модель
- `pruned_real.pt` - после structured pruning (50%)
- `binary_real.pt` - после binary quantization (1-bit)

### 3. Benchmark всех моделей

```bash
python3 benchmark_real.py
```

**Результат**: `../results/benchmark_real.json` с метриками качества и производительности

## Скрипты

### Обучение

#### `train_real.py` - Обучение baseline с нуля

Обучает baseline модель на реальных данных из z80ai/tinychat.

```bash
python3 train_real.py \
    --data ../../../z80ai/examples/tinychat/training-data.txt.gz \
    --epochs 100 \
    --warmup-epochs 20 \
    --batch-size 32 \
    --lr 0.001 \
    --output ../results/baseline_real.pt
```

**Параметры**:
- `--data` - путь к training-data.txt.gz
- `--epochs` - число эпох обучения
- `--warmup-epochs` - эпохи для progressive quantization
- `--batch-size` - размер батча
- `--lr` - learning rate
- `--output` - путь для сохранения модели

**Выход**:
- `.pt` файл с моделью и метаданными
- `.npz` файл для экспорта в C

**Время**: ~40 секунд на 30 эпох (CPU)

#### `load_baseline.py` - Загрузка готовой модели

Загружает предобученную модель из z80ai/tinychat/model.npz.

```bash
python3 load_baseline.py \
    --model ../../../z80ai/examples/tinychat/model.npz \
    --output ../results/baseline_loaded.pt
```

**Внимание**: модель в .npz уже квантована в 2-bit, поэтому accuracy низкий (~8%). Для экспериментов лучше обучить с нуля через `train_real.py`.

### Оптимизация

#### `pipeline_real.py` - Полный цикл оптимизации

Применяет методы оптимизации последовательно:
1. Baseline модель
2. Structured pruning (50% нейронов)
3. Binary quantization (1-bit)

```bash
python3 pipeline_real.py \
    --baseline ../results/baseline_real.pt \
    --fine-tune-epochs 50 \
    --lr 0.0001 \
    --patience 10
```

**Параметры**:
- `--baseline` - путь к baseline модели
- `--fine-tune-epochs` - эпохи fine-tuning после каждой модификации
- `--lr` - learning rate для fine-tuning
- `--patience` - early stopping patience

**Выход**:
- `pruned_real.pt` - модель после pruning
- `binary_real.pt` - модель после binary quantization

**Время**: ~20-30 секунд (зависит от числа эпох)

### Оценка

#### `benchmark_real.py` - Сравнение моделей

Измеряет метрики качества и производительности для всех моделей.

```bash
python3 benchmark_real.py \
    --baseline ../results/baseline_real.pt \
    --pruned ../results/pruned_real.pt \
    --binary ../results/binary_real.pt \
    --output ../results/benchmark_real.json
```

**Метрики качества**:
- Accuracy на validation set
- Loss
- Примеры inference

**Метрики производительности**:
- Flash usage (KB)
- SRAM usage (bytes)
- Inference latency (ms)
- Влезает ли в Arduino UNO (32 KB Flash, 2 KB SRAM)

**Выход**: JSON с полными результатами + таблица в stdout

## Результаты

### Текущие метрики (30 эпох обучения)

| Model | Accuracy | Parameters | Flash (KB) | Latency (ms) | Fits Arduino? |
|-------|----------|------------|------------|--------------|---------------|
| Baseline | 28.66% | 144,871 | 36.57 | 559.38 | ❌ |
| Pruned | 35.57% | 54,023 | 13.83 | 208.28 | ✅ |
| Binary | 15.55% | 54,023 | 13.83 | 208.28 | ✅ |

**Выводы**:
- Baseline не влезает в Arduino (>32 KB Flash)
- Pruned модель лучше baseline по accuracy (+6.91%) и в 2.7× быстрее
- Binary модель теряет 13% accuracy, но работает на Arduino

## Данные

### Формат

Данные в формате `query|response`:
```
hello|HI
how are you|OK
are you a robot|YES
```

### Preprocessing

1. **Query encoding**: trigram hash buckets (128 dimensions)
   - Текст разбивается на триграммы (3 символа)
   - Каждая триграмма хешируется в bucket [0, 127]
   - Подсчитываются частоты

2. **Context encoding**: последние 8 символов ответа (128 dimensions)

3. **Input**: concatenate(query_vec, context_vec) = 256 dimensions

4. **Output**: индекс первого символа ответа (39 классов)

### Charset

39 символов: ` ABCDEFGHIJKLMNOPQRSTUVWXYZ012345678!/?`

## Архитектура

### Baseline (из HW1)
```
Input: 256 (128 query + 128 context)
  ↓
FC1: 256 → 256, ReLU
  ↓
FC2: 256 → 192, ReLU
  ↓
FC3: 192 → 128, ReLU
  ↓
FC4: 128 → 39 (output)
```

**Параметры**: 144,871
- FC1: 256×256 + 256 = 65,792
- FC2: 256×192 + 192 = 49,344
- FC3: 192×128 + 128 = 24,704
- FC4: 128×39 + 39 = 5,031

### Pruned (50% нейронов)
```
Input: 256
  ↓
FC1: 256 → 128, ReLU
  ↓
FC2: 128 → 96, ReLU
  ↓
FC3: 96 → 64, ReLU
  ↓
FC4: 64 → 39 (output)
```

**Параметры**: 54,023 (37.3% of baseline)

### Binary (1-bit веса)
Та же архитектура, что Pruned, но веса квантованы в {-1, +1}.

## Устаревшие скрипты (синтетика)

⚠️ Следующие скрипты используют синтетические данные и не должны использоваться:
- `train_baseline.py` - toy модель на случайном шуме
- `train_optimized.py` - оптимизация toy модели
- `benchmark.py` - benchmark на синтетике
- `visualize_results.py` - визуализация синтетических результатов

Используйте скрипты с суффиксом `_real` для работы с реальной моделью.

## Troubleshooting

### FileNotFoundError: model.npz

Проверьте путь к данным:
```bash
ls ../../../z80ai/examples/tinychat/
```

Если данных нет, скачайте z80ai репозиторий.

### Low accuracy after loading from .npz

Модель в .npz уже квантована, используйте `train_real.py` для обучения с нуля.

### RuntimeError: state_dict keys mismatch

Binary модель использует BinaryLinear вместо OverflowAwareLinear. Benchmark загружает с `strict=False`.

## Следующие шаги

Для улучшения результатов:

1. **Обучить дольше**: 100-200 эпох для лучшего baseline
2. **Подобрать prune ratio**: попробовать 30%, 40%, 60%
3. **Улучшить binary**: больше эпох fine-tuning, другой LR
4. **Добавить 2-bit**: промежуточный вариант между 1-bit и float
