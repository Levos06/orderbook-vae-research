# Эксперимент: сравнение точности (fidelity)

## Назначение
Визуальное сравнение нескольких моделей по реконструкции лог-объемов.

## Входные данные
- `data/orderbook_data.npy`
- Модели из `02_log_cumsum_vae`, `03_log_cumsum_derivative_loss`, `04_conv1d_vae`

## Скрипты
- `compare_volume_fidelity.py`
- `visualize_fidelity_head_to_head.py`

## Графики
- `plots/volume_fidelity_comparison.png`
- `plots/volume_fidelity_comparison_detailed.png`
