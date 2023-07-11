# irene_plugin_fuzzy_sklearn
Плагин для Ирины для нечеткого распознавания команд на основе машинного обучения.

Используется нейросетевая модель: https://huggingface.co/inkoziev/sbert_synonymy

В отличие от https://github.com/janvarev/irene_plugin_fuzzy_ai_sentence толерантнее относится к дополнительным словам, которые были сказаны кроме конкретного названия команды.

(Требуется версия Ирины 8.0+)

## Установка

1. Скопировать `plugin_boltalka2_openai.py` в plugins
2. Установить пакеты из requirements.txt (`pip install [пакеты]`)
3. Запустить Ирину
