/*
 * Z80-μLM Micro Model для Arduino UNO
 * 
 * Архитектура: 64 → 32 → 16 → 10
 * Weights: 2-bit quantized, хранятся в PROGMEM (Flash)
 * 
 * Inference: ~10-20ms @ 16MHz
 * Memory: <1.5 KB SRAM
 */

#include <avr/pgmspace.h>

// === Конфигурация модели ===
#define INPUT_SIZE 64
#define LAYER1_SIZE 32
#define LAYER2_SIZE 16
#define OUTPUT_SIZE 10

#define SCALE_FACTOR 32

// === Веса в Flash (будет сгенерировано из Python) ===
// TODO: Заполнить из экспортированной модели
const uint8_t PROGMEM weights_layer1[(INPUT_SIZE * LAYER1_SIZE) / 4] = {
    // Packed 2-bit weights
};

const int16_t PROGMEM biases_layer1[LAYER1_SIZE] = {
    // Scaled by SCALE_FACTOR
};

// Аналогично для layer2, layer3...

// === Output mapping ===
const char* const OUTPUT_LABELS[OUTPUT_SIZE] PROGMEM = {
    "OK", "YES", "NO", "MAYBE", "HI", "BYE", "HELP", "WHY", "WHO", "WHAT"
};

// === Runtime buffers (SRAM) ===
int16_t input_buffer[INPUT_SIZE];
int16_t layer1_buffer[LAYER1_SIZE];
int16_t layer2_buffer[LAYER2_SIZE];
int16_t output_buffer[OUTPUT_SIZE];

// === Функции ===

/**
 * Распаковка 2-bit веса из Flash
 * 
 * @param addr Адрес в массиве packed weights
 * @param idx Индекс веса (0-3 в байте)
 * @return Вес: -2, -1, 0, или +1
 */
int8_t unpack_weight(const uint8_t* addr, uint16_t idx) {
    uint8_t packed = pgm_read_byte(addr + idx / 4);
    uint8_t shift = (idx % 4) * 2;
    uint8_t val = (packed >> shift) & 0x03;
    return (int8_t)(val - 2);  // Map 0,1,2,3 → -2,-1,0,+1
}

/**
 * Matrix multiply: output[i] = weights[i,:] · input + bias[i]
 * 
 * @param input Input activations
 * @param input_size Input dimension
 * @param output Output activations
 * @param output_size Output dimension
 * @param weights_addr Адрес весов в PROGMEM
 * @param biases_addr Адрес biases в PROGMEM
 */
void matmul_layer(const int16_t* input, uint16_t input_size,
                  int16_t* output, uint16_t output_size,
                  const uint8_t* weights_addr,
                  const int16_t* biases_addr) {
    for (uint16_t i = 0; i < output_size; i++) {
        int32_t acc = 0;
        
        // Multiply-accumulate
        for (uint16_t j = 0; j < input_size; j++) {
            int8_t w = unpack_weight(weights_addr, i * input_size + j);
            if (w != 0) {
                acc += (int32_t)input[j] * w;
            }
        }
        
        // Add bias
        int16_t bias = pgm_read_word(&biases_addr[i]);
        acc += bias;
        
        // Scale down (fixed-point division)
        output[i] = (int16_t)(acc >> 2);  // Divide by 4
    }
}

/**
 * ReLU activation: max(0, x)
 */
void relu(int16_t* buffer, uint16_t size) {
    for (uint16_t i = 0; i < size; i++) {
        if (buffer[i] < 0) {
            buffer[i] = 0;
        }
    }
}

/**
 * Trigram encoding: хеширование строки в buckets
 * 
 * Упрощённая версия для Arduino
 */
void encode_trigrams(const char* text, int16_t* output, uint16_t num_buckets) {
    // Clear buffer
    for (uint16_t i = 0; i < num_buckets; i++) {
        output[i] = 0;
    }
    
    // Hash trigrams
    size_t len = strlen(text);
    for (size_t i = 0; i < len - 2; i++) {
        // Simple hash
        uint16_t hash = ((uint16_t)text[i] * 7 + 
                        (uint16_t)text[i+1] * 31 + 
                        (uint16_t)text[i+2]) % num_buckets;
        output[hash] += SCALE_FACTOR;
    }
}

/**
 * Полный inference pass
 * 
 * @param text Входной текст
 * @return Индекс predicted класса
 */
uint8_t inference(const char* text) {
    // Encode input
    encode_trigrams(text, input_buffer, INPUT_SIZE);
    
    // Layer 1: 64 → 32
    matmul_layer(input_buffer, INPUT_SIZE, 
                 layer1_buffer, LAYER1_SIZE,
                 weights_layer1, biases_layer1);
    relu(layer1_buffer, LAYER1_SIZE);
    
    // Layer 2: 32 → 16
    // TODO: аналогично
    
    // Layer 3: 16 → 10
    // TODO: аналогично
    
    // Find argmax
    int16_t max_val = output_buffer[0];
    uint8_t max_idx = 0;
    for (uint8_t i = 1; i < OUTPUT_SIZE; i++) {
        if (output_buffer[i] > max_val) {
            max_val = output_buffer[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// === Arduino Setup & Loop ===

void setup() {
    Serial.begin(9600);
    Serial.println(F("Z80-uLM Micro Model Ready"));
    Serial.print(F("Free SRAM: "));
    Serial.println(freeRam());
    Serial.println(F("Type your query:"));
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        input.toLowerCase();
        
        if (input.length() == 0) return;
        
        // Measure latency
        unsigned long start = micros();
        uint8_t pred = inference(input.c_str());
        unsigned long elapsed = micros() - start;
        
        // Output result
        char label[8];
        strcpy_P(label, (char*)pgm_read_word(&(OUTPUT_LABELS[pred])));
        
        Serial.print(F("> "));
        Serial.println(label);
        Serial.print(F("Latency: "));
        Serial.print(elapsed / 1000.0);
        Serial.println(F(" ms"));
    }
}

/**
 * Helper: измерить свободную SRAM
 * Источник: https://www.arduino.cc/en/Tutorial/Foundations/Memory
 */
int freeRam() {
    extern int __heap_start, *__brkval;
    int v;
    return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
}
