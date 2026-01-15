# Эксперимент: гибрид AE + VAE (simple)

## Назначение
Сначала AE аппроксимирует остаток, затем VAE моделирует остаток ошибки AE.

## Входные данные
- `data/orderbook_data.npy`
- Предобученный AE из `experiments/volumes/06_autoencoder/models/`

## Скрипты
- `train_volume_hybrid.py`

## Выходные артефакты
- Модели и скейлеры: `models/`
