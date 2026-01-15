# Эксперимент: базовый VAE на полном состоянии стакана

## Назначение
Базовая модель на векторизованном состоянии стакана без специализированной обработки.

## Входные данные
- `data/orderbook_data.npy`

## Скрипты
- `train.py` — обучение базового VAE и сохранение артефактов.
- `experiment.py` — перебор размерности латентного пространства.
- `visualize_comparison.py` — визуализация реконструкций для разных размерностей.

## Выходные артефакты
- Модели и скейлеры: `models/`
- Графики: `plots/`
- Логи/таблицы: `results/`

## Графики
- `plots/training_curves.png`
- `plots/reconstructions.png`
- `plots/latent_dim_comparison.png`
- `plots/comparison_grid.png`
