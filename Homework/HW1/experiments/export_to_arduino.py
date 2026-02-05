#!/usr/bin/env python3
"""
Экспорт обученной модели в C arrays для Arduino.

Читает model.npz и генерирует weights.h с PROGMEM декларациями.
"""

import numpy as np
import argparse

def pack_2bit_weights(weights):
    """
    Упаковка весов в 2-bit: {-2, -1, 0, +1} → {0, 1, 2, 3}
    4 веса на байт.
    """
    # Map -2,-1,0,+1 → 0,1,2,3
    weights_mapped = (weights + 2).astype(np.uint8)
    
    # Pack 4 weights per byte
    packed = []
    for i in range(0, len(weights_mapped), 4):
        chunk = weights_mapped[i:i+4]
        if len(chunk) < 4:
            chunk = np.pad(chunk, (0, 4 - len(chunk)), constant_values=2)
        
        byte = (chunk[0] | (chunk[1] << 2) | (chunk[2] << 4) | (chunk[3] << 6))
        packed.append(byte)
    
    return np.array(packed, dtype=np.uint8)

def format_array(arr, name, progmem=True, line_width=12):
    """Форматирование массива для C кода."""
    lines = []
    
    # Header
    type_str = "uint8_t" if arr.dtype == np.uint8 else "int16_t"
    progmem_str = " PROGMEM" if progmem else ""
    lines.append(f"const {type_str}{progmem_str} {name}[{len(arr)}] = {{")
    
    # Data
    for i in range(0, len(arr), line_width):
        chunk = arr[i:i+line_width]
        formatted = ", ".join(f"{x:5d}" if arr.dtype == np.int16 else f"0x{x:02X}" for x in chunk)
        lines.append(f"    {formatted},")
    
    lines.append("};")
    return "\n".join(lines)

def export_model(npz_path, output_path):
    """Экспорт модели в Arduino .h файл."""
    print(f"Загружаем модель из {npz_path}...")
    data = np.load(npz_path)
    
    # Извлекаем слои
    layers = []
    layer_idx = 1
    while f'fc{layer_idx}_weight' in data:
        w = data[f'fc{layer_idx}_weight']
        b = data[f'fc{layer_idx}_bias']
        layers.append((w, b))
        layer_idx += 1
    
    print(f"Найдено {len(layers)} слоёв")
    
    # Генерируем C код
    with open(output_path, 'w') as f:
        f.write("/*\n")
        f.write(" * Auto-generated weights for Arduino\n")
        f.write(f" * Source: {npz_path}\n")
        f.write(" * Model architecture:\n")
        
        for i, (w, b) in enumerate(layers):
            f.write(f" *   Layer {i+1}: {w.shape[1]} → {w.shape[0]}\n")
        
        f.write(" */\n\n")
        f.write("#ifndef WEIGHTS_H\n")
        f.write("#define WEIGHTS_H\n\n")
        f.write("#include <avr/pgmspace.h>\n\n")
        
        # Экспортируем каждый слой
        for i, (weights, biases) in enumerate(layers):
            layer_name = f"layer{i+1}"
            
            # Flatten и pack weights
            w_flat = weights.flatten()
            w_packed = pack_2bit_weights(w_flat)
            
            f.write(f"// Layer {i+1}: {weights.shape[1]} → {weights.shape[0]}\n")
            f.write(format_array(w_packed, f"weights_{layer_name}", progmem=True))
            f.write("\n\n")
            f.write(format_array(biases, f"biases_{layer_name}", progmem=True))
            f.write("\n\n")
            
            print(f"Layer {i+1}: weights {w_packed.shape}, biases {biases.shape}")
        
        # Размеры для удобства
        f.write("// Layer dimensions\n")
        for i, (w, b) in enumerate(layers):
            f.write(f"#define LAYER{i+1}_INPUT_SIZE {w.shape[1]}\n")
            f.write(f"#define LAYER{i+1}_OUTPUT_SIZE {w.shape[0]}\n")
        
        f.write("\n#endif // WEIGHTS_H\n")
    
    print(f"✓ Экспортировано в {output_path}")
    
    # Статистика по памяти
    total_flash = sum(len(pack_2bit_weights(w.flatten())) + len(b) * 2 
                     for w, b in layers)
    print(f"\nИспользование Flash: ~{total_flash} bytes")

def main():
    parser = argparse.ArgumentParser(description='Export model to Arduino C arrays')
    parser.add_argument('--model', '-m', default='micro_model.npz',
                        help='Input model file (.npz)')
    parser.add_argument('--output', '-o', default='weights.h',
                        help='Output header file (.h)')
    args = parser.parse_args()
    
    export_model(args.model, args.output)

if __name__ == '__main__':
    main()
