# Результаты экспериментов

## Baseline: Z80 @ 4MHz

### Конфигурация модели
- Архитектура: 256→256→192→128→40
- Параметры: 144,000
- Размер бинарника: ~40 KB

### Измерения
- [ ] Latency на символ: ___ ms
- [ ] Total time для ответа "HI": ___ ms
- [ ] Размер .COM файла: ___ bytes

### Тестовые запросы
| Input | Expected | Actual | OK? |
|-------|----------|--------|-----|
| hello | HI | | |
| are you a robot | YES | | |
| do you dream | MAYBE | | |

---

## Micro-модель: Arduino UNO @ 16MHz

### Конфигурация модели
- Архитектура: 64→32→16→10
- Параметры: ___
- Training epochs: ___

### Memory usage
- [ ] Flash (weights): ___ bytes
- [ ] Flash (code): ___ bytes  
- [ ] Flash (total): ___ bytes / 32 KB
- [ ] SRAM (runtime): ___ bytes / 2 KB
- [ ] Free SRAM: ___ bytes

### Производительность
- [ ] Latency (inference): ___ ms
- [ ] Throughput: ___ inferences/sec

### Accuracy
Training set size: ___ examples

Test results (первые 20):
| Input | Expected | Predicted | OK? |
|-------|----------|-----------|-----|
| hi | HI | | |
| hello | HI | | |
| yes | YES | | |
| no | NO | | |
| maybe | MAYBE | | |
| bye | BYE | | |
| help | HELP | | |
| why | WHY | | |
| who | WHO | | |
| what | WHAT | | |

**Accuracy**: ___% (___ / ___)

---

## Micro-модель: LilyPad @ 8MHz

### Memory usage
(те же значения что UNO)

### Производительность
- [ ] Latency (inference): ___ ms
- [ ] Throughput: ___ inferences/sec

### Power consumption
- [ ] Active current: ___ mA @ 5V
- [ ] Idle current: ___ mA
- [ ] Estimated battery life (CR2032): ___ hours

---

## Сравнение платформ

| Метрика | Z80 | UNO | LilyPad | Ratio UNO/Z80 |
|---------|-----|-----|---------|---------------|
| CPU | 4 MHz | 16 MHz | 8 MHz | 4.0x |
| RAM | 64 KB | 2 KB | 2 KB | 0.03x |
| Params | 144K | ~3K | ~3K | 0.02x |
| Latency | ___ ms | ___ ms | ___ ms | ___x |
| Flash/Binary | 40 KB | ___ KB | ___ KB | ___x |

---

## Выводы

### Гипотезы
- [ ] H1: PROGMEM достаточно быстр для inference
  - Результат: ___
  - Overhead: ___ cycles/byte

- [ ] H2: 3K параметров дают >50% accuracy
  - Результат: ___% accuracy
  - Достаточно? ___

- [ ] H3: UNO @ 16MHz быстрее Z80 @ 4MHz
  - Результат: UNO ___ ms vs Z80 ___ ms
  - Speedup: ___x

- [ ] H4: LilyPad может работать >8 часов от батарейки
  - Ток: ___ mA
  - Расчётное время: ___ hours
  - Достаточно? ___

### Инженерные компромиссы

**Что выиграли при портировании на Arduino**:
- ___
- ___

**Чем заплатили**:
- ___
- ___

**Стоило ли оно того?**: ___
