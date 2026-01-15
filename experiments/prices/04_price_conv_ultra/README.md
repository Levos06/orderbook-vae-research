# Эксперимент: Ultra-Conv VAE для цен

## Назначение
Сравнение более глубокой Conv1D архитектуры с предыдущими моделями.

## Входные данные
- `data/orderbook_data.npy`
- Модели из `prices/02_price_detrended_vae` и `prices/03_price_conv`

## Скрипты
- `train_price_conv_ultra.py`
- `compare_price_ultra.py`

## Графики
- `plots/price_ultra_fidelity.png`
