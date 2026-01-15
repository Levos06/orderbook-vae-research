# Эксперимент: бенчмарк всех моделей объема

## Назначение
Единая таблица метрик для набора моделей на OOS-выборке.

## Входные данные
- `data/orderbook_data.npy`
- Модели из `02_log_cumsum_vae`, `03_log_cumsum_derivative_loss`, `04_conv1d_vae`, `05_conv1d_complex`, `06_autoencoder`, `07_hybrid`, `08_hybrid_complex`, `09_meta_ensemble`

## Скрипты
- `benchmark_all_models.py`

## Результаты
- `results/benchmark_table.md`
