# Эксперимент: Conv1D VAE для цен

## Назначение
Сравнение Conv1D VAE и MLP на детрендированных ценах.

## Входные данные
- `data/orderbook_data.npy`
- Детрендированная модель из `prices/02_price_detrended_vae`

## Скрипты
- `train_price_conv.py`
- `compare_price_conv.py`

## Графики
- `plots/price_conv_comparison.png`
