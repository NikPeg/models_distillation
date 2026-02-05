# Анализ и портирование Z80-μLM

## 1. Модель и контекст применения

**Модель**: Z80-μLM — conversational AI с 2-bit квантизацией для 8-bit процессора.

**Задача**: генерация коротких текстовых ответов (1-2 слова) на пользовательский ввод в реальном времени.

**Сценарий**: инференс в интерактивном режиме на экстремально ограниченном железе. Offline режим, модель зашита в бинарник. Latency <100ms на символ критична для UX.

**Источник**: https://github.com/HarryR/z80ai

---

## 2. Целевые платформы

### Baseline: Z80 @ 4MHz (CP/M)
- RAM: 64 KB
- CPU: 8-bit, нет FPU
- Частота: 4 MHz
- Регистры: 16-bit пары (HL, DE, BC)

### Эксперимент 1: Arduino UNO
- MCU: ATmega328P @ 16 MHz
- SRAM: 2 KB
- Flash: 32 KB
- EEPROM: 1 KB

### Эксперимент 2: Arduino LilyPad
- MCU: ATmega328P @ 8 MHz
- SRAM: 2 KB
- Flash: 32 KB
- Батарейное питание

**План**: проверить, возможно ли портировать модель на платформы с 32x меньше RAM.

---

## 3. Архитектура модели

### Кодирование входа
**Trigram hash encoding**: текст хешируется в 128 buckets через триграммы. Инвариантно к порядку слов, толерантно к опечаткам.

**Context encoding**: последние 8 символов output тоже хешируются в 128 buckets для autoregressive generation.

**Total input**: 256 dimensions (128 query + 128 context).

### Сеть
```
Input: 256
  ↓
FC1: 256 → 256 + ReLU
  ↓
FC2: 256 → 192 + ReLU
  ↓
FC3: 192 → 128 + ReLU
  ↓
FC4: 128 → 40 (charset)
```

**Параметры**: ~144K weights, 2-bit quantized → ~36 KB packed.

### Ресурсоёмкие части
- **FC1**: 256×256 = 65K weights → 16 KB
- **Matmul loops**: ~100K multiply-accumulate на символ
- **Weight unpacking**: 4 weights на байт, распаковка в цикле

---

## 4. Вычислительные затраты и узкие места

### Операции на Z80

**Weight unpacking** (на каждый вес):
```z80
ld a, (PACKED)      ; Загрузить упакованный байт
and 03h             ; Маска нижних 2 бит
sub 2               ; Map 0,1,2,3 → -2,-1,0,+1
```
**Cost**: ~12 cycles на вес.

**Multiply-accumulate**:
```z80
MULADD:
    or a
    jr z, DONE       ; weight=0: skip (4 cycles)
    jp m, NEG        ; weight<0: subtract
    ; weight=+1
    ld hl, (ACC)     ; 16 cycles
    add hl, de       ; 11 cycles
    ld (ACC), hl     ; 16 cycles
    ret              ; 10 cycles
```
**Cost per non-zero weight**: ~50 cycles.

**Total на символ**:
- Unpack: 144K × 12 = 1.7M cycles
- MAC: ~70K non-zero × 50 = 3.5M cycles
- **Total**: ~5M cycles ≈ **1.25 sec** @ 4MHz

### Bottlenecks на Z80

1. **CPU-bound**: нет параллелизма, sequential execution
2. **Memory bandwidth**: sequential read из packed weights
3. **Register pressure**: 8-bit registers, нужны 16-bit операции
4. **Отсутствие SIMD**: каждый weight обрабатывается отдельно

---

## 5. Системные ограничения

### Z80 (baseline)
- **Память**: 40 KB для модели + code влезает в 64 KB
- **Latency**: 1-1.5 sec на символ (приемлемо для chatbot)
- **Throughput**: не критично (single user)

### Arduino UNO/LilyPad
- **Критично: SRAM 2 KB**
  - Оригинальная модель: 36 KB weights
  - Runtime buffers: 1 KB
  - **Не влезает даже во Flash целиком!**

- **Latency бюджет**: <100ms на символ (UX)
  - Потенциально быстрее: 16 MHz vs 4 MHz

- **Энергопотребление** (для LilyPad):
  - Батарейное питание
  - Нужна minimal inference cost

---

## 6. Гипотезы по ускорению/адаптации

### Гипотеза 1: PROGMEM для весов (UNO/LilyPad)

**Идея**: хранить веса во Flash, читать через `pgm_read_byte`.

**Pros**:
- Освобождает SRAM для runtime
- Flash 32 KB достаточно для ~30 KB модели

**Cons**:
- Медленнее чтение из Flash (~3 cycles/byte overhead)
- Все равно нужно 1+ KB SRAM для activations

**Применимо**: да, но модель слишком большая.

---

### Гипотеза 2: Радикальное уменьшение архитектуры

**Идея**: урезать модель до 64→32→16→10.

**Параметры**:
- Layer 1: 64×32 = 2K weights → 512 bytes
- Layer 2: 32×16 = 512 weights → 128 bytes
- Layer 3: 16×10 = 160 weights → 40 bytes
- **Total**: ~680 bytes (влезает!)

**Runtime в SRAM**:
- Input buffer: 64×2 = 128 bytes
- Activations: 32+16+10 = 58 values × 2 = 116 bytes
- Trigram state: ~200 bytes
- **Total**: ~650 bytes ✅

**Cons**:
- **48x меньше параметров** (144K → 3K)
- Меньше buckets → хуже различение входов
- Меньше output классов (10 вместо 40)
- Потеря качества

**Применимо**: технически да, но модель станет примитивной.

---

### Гипотеза 3: Оптимизация inference кода

**Идея**: использовать преимущества AVR.

**Методы**:
- **32-bit accumulator**: AVR gcc поддерживает int32
- **Loop unrolling**: для малых моделей
- **Eliminate zero weights**: pruning при экспорте
- **Fixed-point optimization**: использовать AVR asm для MAC

**Выигрыш**:
- UNO @ 16 MHz → 4x faster than Z80
- Оптимизированный код → еще 2x
- **Potential**: 10-20ms на символ (micro model)

**Применимо**: да, для уменьшенной модели.

---

### Гипотеза 4: Квантизация ниже 2-bit

**Идея**: 1-bit weights (binary net) или ternary.

**Pros**:
- 1-bit: {-1, +1} → 8 weights/byte
- Проще MAC: только add/subtract

**Cons**:
- Еще больше потеря качества
- Training сложнее (уже на грани)

**Применимо**: теоретически, но Z80-μLM уже на пределе с 2-bit.

---

### Альтернативный подход: качество через жертву latency

**Важно**: это не основной путь для экспериментов, а возможное развитие проекта.

**Идея**: если снять требование latency и допустить "минутное мышление" (1 token/min), можно улучшить качество модели.

#### Что даёт отказ от speed constraints

**1. Offloading весов на SD/SPI-Flash**
- Веса хранятся на внешней памяти (SD карта, SPI Flash)
- Читаются маленькими блоками в runtime
- Позволяет поднять модель с 3K до десятков/сотен тысяч параметров
- Ограничение — размер Flash/SD, а не SRAM

**Механика**: читаем блок весов → делаем MAC → выгружаем, берём следующий блок.

**2. Более сложная архитектура**
- Упрощённый Transformer вместо flat MLP
- Tiny attention с малой размерностью
- Более адекватный embedding
- Несколько мини-блоков вместо одного прохода

**3. Алгоритмические улучшения**
- Несколько проходов: статистический N-gram predictor + neural corrector
- Coarse-to-fine sampling
- Beam search для выбора токена
- Перегенерация "плохих" токенов

#### Реальные примеры

**ESP32-S3 + PSRAM** (не UNO, но показательно):
- 260K параметров tiny-LLM через llama2.c
- ~20 tokens/sec
- Качество: "очень слабое", но на порядок лучше 3K-модели
- TensorFlow Lite Micro с offloading на SD

#### Где проходит потолок на UNO

**RAM — главное ограничение**:
- Веса можно стримить с SD
- Но промежуточные activations и контекст — нет
- Это ограничивает размерность слоёв и глубину

**2 KB SRAM никуда не делись**:
- Можно убить скорость
- Можно усложнить архитектуру
- Но нельзя обойти RAM constraint

**Вывод**: "минутное мышление" превращает 3K demo-toy в "чуть менее игрушечную" модель, но не в "нормальную" LM. Для реального улучшения нужен переход на ESP32-S3/RP2040 + PSRAM.

#### Применимо для проекта?

**Не в основном русле**, потому что:
- Нужна внешняя SD карта (усложнение setup)
- Latency ~минуты (плохой UX для демо)
- Качество всё равно слабое

**Но может быть интересно как:**
- Теоретический анализ trade-off: speed vs capacity
- Пример extreme offloading техник
- Обоснование перехода на другое железо

---

## 7. Инженерные компромиссы

### Сценарий: портирование на Arduino UNO

**Изменения**:
- Архитектура: 256→256→192→128 ⇒ 64→32→16→10
- Параметры: 144K ⇒ 3K (в 48 раз меньше)
- Input buckets: 256 ⇒ 64
- Output: 40 символов ⇒ 10 команд
- Хранение: RAM ⇒ Flash (PROGMEM)

**Что выигрываем**:
- ✅ Работает на 2 KB SRAM
- ✅ Быстрее инференс (~10-20ms vs 1250ms)
- ✅ Дешевая платформа ($3 vs $100+ vintage)
- ✅ GPIO/сенсоры доступны
- ✅ Низкое энергопотребление (для LilyPad)

**Чем платим**:
- ❌ Качество: тупая модель, только basic patterns
- ❌ Словарь: 10 команд вместо 40 символов
- ❌ Контекст: 64 buckets хуже различают входы
- ❌ Персональность: почти исчезнет
- ❌ Нужно переобучение с нуля

### Memory breakdown (UNO после оптимизации)

**Flash (read-only)**:
- Weights: 680 bytes
- Biases: 116 bytes
- Inference code: ~2 KB
- I/O code: ~1 KB
- **Total**: ~4 KB из 32 KB ✅

**SRAM (runtime)**:
- Activations: 244 bytes
- Trigram buffers: 200 bytes
- Stack: 150 bytes
- **Total**: ~600 bytes из 2 KB ✅

---

## Выводы

1. **Z80-μLM технически не портируется на Arduino напрямую** из-за 32x меньше памяти.

2. **Возможна радикальная адаптация** с потерей 95%+ качества — модель превратится в простейший pattern matcher.

3. **Подход (2-bit QAT, integer inference) переносим**, но архитектура требует полного пересмотра под 2 KB SRAM.

4. **Arduino выигрывает в скорости** (16 MHz vs 4 MHz), но проигрывает в capacity. Это trade-off: latency vs quality.

5. **Практический вывод**: для embedded AI на ATmega328P нужны модели на порядок проще, чем Z80-μLM. Либо переходить на ARM Cortex-M (32 KB+ SRAM).

---

## План экспериментов

### Этап 1: Baseline на Z80 эмуляторе
- Запустить tinychat в iz-cpm
- Измерить latency на символ
- Проверить размер модели

### Этап 2: Proof-of-concept на Arduino UNO
- Обучить micro-модель 64→32→16→10
- Реализовать inference на C++ с PROGMEM
- Измерить latency, память

### Этап 3: Оптимизация для LilyPad @ 8MHz
- Профилировать узкие места
- Применить loop unrolling
- Измерить энергопотребление

На каждом этапе проверяется гипотеза: возможно ли вообще работать на данной платформе.
