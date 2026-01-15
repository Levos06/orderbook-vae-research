# Эксперимент: бенчмарк ценовых моделей

## Назначение
Единая таблица метрик на OOS-выборке для ценовых моделей.

## Входные данные
- `data/orderbook_data.npy`
- Модели из `prices/01_price_mlp`, `prices/02_price_detrended_vae`, `prices/03_price_conv`, `prices/04_price_conv_ultra`

## Скрипты
- `benchmark_price_models.py`

## Результаты
- `results/price_benchmark_table.md`
