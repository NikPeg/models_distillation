# Grid Search для подбора гиперпараметров

Автоматический подбор оптимальных гиперпараметров для baseline модели.

## Быстрый запуск (рекомендуется)

**15 запусков за ~10-15 минут** (50 эпох каждый):

```bash
cd Homework/HW2/experiments
python3 quick_grid_search.py
```

Результаты сохраняются в:
- `results/quick_grid/quick_grid_results.csv` - таблица результатов
- `results/quick_grid/quick_grid_results.json` - JSON для программного доступа
- `results/quick_grid/run_XX.pt` - сохраненные модели

## Полный grid search

**20 запусков за ~40-60 минут** (150 эпох каждый):

```bash
python3 grid_search.py
```

Результаты в `results/grid_search/`.

## Что тестируется

### quick_grid_search.py (15 комбинаций)

| Параметр | Значения |
|----------|----------|
| Learning Rate | 0.001, 0.0005, 0.0002, 0.0001, 0.00005 |
| Patience | 10, 15, 20, 25, 30 |
| Warmup Epochs | 5, 10, 15, 20 |
| Batch Size | 16, 32, 64 |
| Epochs | 50 (фиксировано для скорости) |

### grid_search.py (20 комбинаций)

| Параметр | Значения |
|----------|----------|
| Learning Rate | 0.001, 0.0005, 0.0002, 0.0001 |
| Patience | 20, 30, 50 |
| Warmup Epochs | 20, 30, 40 |
| Batch Size | 32, 64 |
| Epochs | 100, 150, 200 |

## Формат результатов

### CSV таблица

```csv
run_id,success,val_acc,stopped_at,elapsed_time,epochs,warmup_epochs,lr,patience,batch_size,output_path
1,True,0.4123,45,89.3,50,10,0.0002,15,32,../results/quick_grid/run_01.pt
2,True,0.3987,38,82.1,50,10,0.001,15,32,../results/quick_grid/run_02.pt
...
```

### Вывод в консоль

```
================================================================================
QUICK GRID SEARCH RESULTS
================================================================================

Successful runs: 15/15

TOP 5:
--------------------------------------------------------------------------------
#    Val Acc    LR           Patience   Batch    Time    
--------------------------------------------------------------------------------
1    0.4123     0.000200     15         32       89.3s
2    0.4087     0.000100     20         32       91.2s
3    0.4045     0.000500     15         32       85.7s
4    0.3998     0.000200     20         32       93.1s
5    0.3976     0.000200     10         64       88.4s

================================================================================
BEST CONFIG:
  Val Accuracy:  0.4123
  Learning Rate: 0.0002
  Patience:      15
  Batch Size:    32
  Warmup:        10
  Stopped at:    45/50
================================================================================
```

## Использование лучшей модели

После завершения grid search:

```bash
# Посмотри результаты
cat ../results/quick_grid/quick_grid_results.csv | head -6

# Запусти pipeline с лучшей моделью
python3 pipeline_real.py --baseline ../results/quick_grid/run_01.pt --fine-tune-epochs 50

# Обнови benchmark
python3 benchmark_real.py --baseline ../results/quick_grid/run_01.pt
```

## Продолжить обучение лучшей конфигурации

Найди лучшие параметры из таблицы и переобучи с большим числом эпох:

```bash
# Пример: лучшие параметры lr=0.0002, patience=15
python3 train_real.py \
    --epochs 200 \
    --lr 0.0002 \
    --patience 15 \
    --warmup-epochs 10 \
    --batch-size 32 \
    --output ../results/best_from_grid.pt
```

## Мониторинг процесса

Скрипты выводят прогресс в реальном времени:

```
[1/15] Running: lr=0.001, patience=15, batch=32
  ✓ Val Acc: 0.3987 (stopped at epoch 38)

[2/15] Running: lr=0.0005, patience=15, batch=32
  ✓ Val Acc: 0.4045 (stopped at epoch 42)
...
```

## Промежуточное сохранение

Результаты автоматически сохраняются каждые 5 запусков, так что можно остановить (Ctrl+C) и посмотреть промежуточные результаты.

## Tips

1. **Начни с quick_grid_search.py** — быстрая оценка за 10-15 минут
2. **Используй лучшие параметры** из quick для полного обучения (150-200 эпох)
3. **Сравни результаты** — таблица покажет, какие параметры влияют больше всего
4. **Переобучение**: если stopped_at << epochs, нужно больше данных или регуляризация

## Что дальше

После grid search:

1. Возьми топ-3 конфигурации
2. Переобучи каждую с 150-200 эпохами
3. Запусти pipeline (pruning + binary) на лучшей
4. Обнови REPORT.md с новыми результатами
