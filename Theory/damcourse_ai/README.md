# Материалы курса "Дистилляция и ускорение моделей"

В данном репозитории находятся примеры, рассмотренные на семинарах курса.

Для запуска требуется Python3, а также дополнительные библиотеки, которые устанавливаются непосредственно по тексту каждого примера.

Крайне рекомендуется создать отдельное окружение для выполнения всех заданий курса. Сделать это можно следующим образом.

`python3 -m venv .venv`

### Вводный семинар
Профилирование моделей
[profiler.ipynb](profiler.ipynb)

### Дистилляция знаний
Единый пример с демонстраций разных техник дистилляции знаний
[knowledge_distillation.ipynb](knowledge_distillation.ipynb)

### Квантование моделей
Пример динамического квантования модели BERT
[quantization_dynamic_bert.ipynb](quantization_dynamic_bert.ipynb)

Простой пример статического квантования модели
[quantization_static.ipynb](quantization_static.ipynb)

Продвинутый пример статического квантования модели
[quantization_static.ipynb](quantization_static_ext.ipynb)

Продвинутый пример квантования модели при обучении
[quantization_qat_ext.ipynb](quantization_qat_ext.ipynb)

### Усечение (прореживание) моделей 
Простой пример использования разных методов усечения моделей.
[pruning.ipynb](pruning.ipynb)

Продвинутый пример усечения моделей с дообучением.
- Вначале запускается предварительное обучение.
[pruning_ext_pretrain.ipynb](pruning_ext_pretrain.ipynb)
- А затем усечение с дообучением.
[pruning_ext_prune.ipynb](pruning_ext_prune.ipynb)
