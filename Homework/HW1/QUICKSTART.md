# Quick Start: Домашнее задание №1

## TL;DR

Анализ Z80-μLM и попытка портировать на Arduino. Три этапа экспериментов с проверкой гипотез.

## Этап 1: Baseline на Z80 (30 мин)

```bash
# Установить эмулятор
brew install iz-cpm  # macOS

# Запустить tinychat
cd z80ai/examples/tinychat
bash run.sh  # или: iz-cpm chat.com

# Протестировать
> hello
HI
> are you a robot
YES
```

**Измерить**: latency на генерацию ответа, размер бинарника.

## Этап 2: Обучить micro-модель (1-2 часа)

```bash
cd Homework/HW1/experiments

# 1. Подготовить данные
python3 prepare_micro_data.py

# 2. Модифицировать z80ai/feedme.py
# Изменить hidden_sizes, num_buckets (см. experiments/README.md)

# 3. Обучить
cd ../../../z80ai
cat ../Homework/HW1/experiments/micro_training_data.txt | \
    python3 feedme.py --epochs 300

# 4. Экспортировать
python3 exportmodel.py --model model.pt --output micro_model.npz
```

## Этап 3: Запустить на Arduino (1 час)

```bash
cd ../Homework/HW1/experiments

# Экспорт в C arrays
python3 export_to_arduino.py \
    --model ../../../z80ai/micro_model.npz \
    --output weights.h

# Загрузить на Arduino:
# 1. Открыть arduino_template.ino в Arduino IDE
# 2. Скопировать weights.h в папку проекта
# 3. Upload на UNO
# 4. Serial Monitor (9600 baud)
```

## Что измерить

| Метрика | Z80 | Arduino UNO | Arduino LilyPad |
|---------|-----|-------------|-----------------|
| CPU | 4 MHz | 16 MHz | 8 MHz |
| RAM | 64 KB | 2 KB | 2 KB |
| Latency | ? ms | ? ms | ? ms |
| Params | 144K | 3K | 3K |
| Flash used | 40 KB | ? KB | ? KB |
| SRAM used | - | ? bytes | ? bytes |

## Критерии успеха

- ✅ Baseline работает на Z80
- ✅ Micro-модель влезает в Arduino (SRAM <1.5 KB)
- ✅ Latency <100ms на UNO
- ✅ Хотя бы 50% accuracy на test set

## Если что-то не работает

**Модель не влезает**:
- Уменьшить `hidden_sizes` до [16, 8]
- Уменьшить `num_buckets` до 16

**Слишком медленно**:
- Compiler optimization: `-O3` в Arduino IDE
- Уменьшить размер модели

**Низкая accuracy**:
- Больше epochs (500-1000)
- Проверить class balance в данных
- Добавить вариации в training data

## Документация

- `report.md` — полный анализ с расчётами
- `experiments/plan.md` — детальный план экспериментов
- `experiments/README.md` — инструкции по коду

## За рамками основных экспериментов

В report.md есть секция "Альтернативный подход" про offloading на SD/Flash для улучшения качества за счёт жертвы latency (1 token/min). Это не входит в основной plan, но интересно как теоретический анализ trade-offs.
