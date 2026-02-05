# План экспериментов

## Этап 1: Z80 baseline

### Цель
Измерить характеристики оригинальной модели на целевой платформе.

### Шаги
1. Установить CP/M эмулятор iz-cpm
2. Запустить tinychat: `iz-cpm z80ai/examples/tinychat/chat.com`
3. Замерить latency на генерацию ответа
4. Проверить размер бинарника
5. Проанализировать код inference (libz80.py, buildz80com.py)

### Ожидаемые результаты
- Latency: ~1-1.5 sec на символ @ 4MHz
- Размер: ~40 KB
- Baseline для сравнения

---

## Этап 2: Arduino UNO proof-of-concept

### Цель
Проверить гипотезу: возможно ли запустить micro-модель на 2 KB SRAM.

### Подготовка модели

1. **Изменить архитектуру в feedme.py:**
```python
hidden_sizes = [32, 16]  # вместо [256, 192, 128]
query_encoder = TrigramEncoder(num_buckets=32)  # вместо 128
context_encoder = ContextEncoder(num_buckets=32, context_len=4)  # вместо 128, 8
```

2. **Упростить training data:**
   - Только 10 команд вместо 40 символов
   - Например: OK, YES, NO, MAYBE, HI, BYE, HELP, WHY, WHO, WHAT

3. **Обучить модель:**
```bash
cat data/simple.txt | python3 feedme.py --epochs 300
python3 exportmodel.py --model model.pt --output micro_model.npz
```

4. **Экспортировать веса для Arduino:**
   - Написать скрипт конвертации .npz → C arrays
   - Использовать PROGMEM для Flash storage

### Arduino код

**Структура:**
```
arduino_inference/
├── inference.ino         # Main Arduino sketch
├── weights.h             # PROGMEM arrays с весами
├── trigram.h             # Trigram encoding
└── README.md
```

**Ключевые функции:**
- `unpack_weight()` — распаковка 2-bit весов из Flash
- `matmul_layer()` — multiply-accumulate с PROGMEM
- `relu()` — activation function
- `inference()` — full forward pass

### Измерения

1. Memory usage (compile-time):
   - Flash usage
   - SRAM usage

2. Runtime (с Serial output):
   - Latency на inference
   - Throughput

3. Correctness:
   - Тестовые входы vs ожидаемые выходы

### Критерии успеха
- ✅ Модель влезает: Flash <30 KB, SRAM <1.5 KB
- ✅ Latency <100ms на inference
- ✅ Хотя бы 50% правильных ответов на test set

---

## Этап 3: LilyPad оптимизация

### Цель
Адаптировать под 8 MHz CPU и батарейное питание.

### Оптимизации

1. **Снизить частоту Serial** (меньше энергопотребление):
   - 9600 baud вместо 115200

2. **Sleep mode между запросами**:
```cpp
#include <avr/sleep.h>
set_sleep_mode(SLEEP_MODE_IDLE);
sleep_mode();
```

3. **Loop unrolling для малых слоёв**:
   - Развернуть последний слой 16→10

4. **Fixed-point optimization**:
   - Проверить AVR asm для критичных loops

### Измерения

1. Latency @ 8 MHz (должно быть ~2x медленнее UNO)
2. Power consumption:
   - Active inference
   - Idle mode
   - Батарейная жизнь (оценка)

### Критерии успеха
- ✅ Latency <200ms @ 8MHz
- ✅ <50 mA active current
- ✅ Работает от батарейки >8 часов (при 1 запрос/мин)

---

## Метрики для сравнения

| Платформа | CPU | RAM | Latency | Params | Accuracy |
|-----------|-----|-----|---------|--------|----------|
| Z80       | 4MHz | 64KB | 1250ms | 144K | baseline |
| UNO       | 16MHz | 2KB | ?ms | 3K | ?% |
| LilyPad   | 8MHz | 2KB | ?ms | 3K | ?% |

## Гипотезы для проверки

- [ ] H1: PROGMEM достаточно быстр для inference
- [ ] H2: 3K параметров дают >50% accuracy
- [ ] H3: UNO @ 16MHz быстрее Z80 @ 4MHz для micro-модели
- [ ] H4: LilyPad может работать от батарейки >8 часов

---

## Возможные направления развития

### (Не для основных экспериментов)

#### Альтернатива: качество через жертву скорости

**Концепция**: если допустить latency ~1 token/min, можно улучшить качество.

**Методы**:
1. **SD/SPI-Flash offloading**
   - Хранить модель на SD карте
   - Читать веса блоками в runtime
   - Поднять модель до 50-200K параметров

2. **Более сложная архитектура**
   - Tiny transformer вместо flat MLP
   - Несколько проходов по контексту

3. **Beam search / re-sampling**
   - Улучшить selection без роста модели

**Ограничение**: 2 KB SRAM всё равно ограничивает activation size.

**Почему не делаем**:
- Нужна SD карта (усложнение)
- Плохой UX (минуты ожидания)
- Качество всё равно слабое

**Но интересно для**:
- Теоретического анализа trade-offs
- Обоснования перехода на ESP32-S3/STM32

#### Переход на другое железо

Если нужно реально улучшить качество:

**ESP32-S3 + PSRAM**:
- PSRAM: 2-8 MB (vs 2 KB SRAM)
- CPU: 240 MHz (vs 16 MHz)
- Демонстрация: 260K params tiny-LLM, ~20 tok/s
- Качество: всё ещё слабое, но на порядок лучше

**STM32 + external RAM**:
- ARM Cortex-M4/M7
- Несколько МБ RAM
- TensorFlow Lite Micro friendly

**Trade-off**: теряем дешевизну ($3 → $10-15), выигрываем capacity.
