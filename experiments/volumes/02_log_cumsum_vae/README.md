# Эксперимент: VAE на лог-кумулятивных объемах

## Назначение
Обучение VAE на остатках лог-кумулятивной кривой объемов (OLS детрендинг).

## Входные данные
- `data/orderbook_data.npy`

## Скрипты
- `train_volume.py` — обучение модели и генерация базовых реконструкций.
- `visualize_volume_expanded.py` — расширенная визуализация реконструкций.

## Выходные артефакты
- Модели и скейлеры: `models/`
- Графики: `plots/`

## Графики
- `plots/volume_vae_reconstructions.png`
- `plots/volume_vae_reconstructions_expanded.png`
