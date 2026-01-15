# Эксперимент: мета-ансамбль

## Назначение
Сбор предсказаний нескольких моделей и обучение мета-MLP для финальной реконструкции.

## Входные данные
- `data/orderbook_data.npy`
- Модели из `03_log_cumsum_derivative_loss`, `04_conv1d_vae`, `05_conv1d_complex`, `07_hybrid`, `06_autoencoder`

## Скрипты
- `prepare_ensemble_data.py` — подготовка входов ансамбля.
- `train_meta_model.py` — обучение мета-MLP.
- `visualize_meta_results.py` — визуализация результата.

## Выходные артефакты
- `data/ensemble_inputs.npy`, `data/ensemble_targets.npy`
- Модели и скейлеры: `models/`
- Графики: `plots/volume_meta_comparison.png`
