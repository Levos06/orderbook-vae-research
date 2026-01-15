# VAE исследования ордербука (объемы и цены)

Коротко: репозиторий содержит набор экспериментов по реконструкции состояний стакана (объемы) и ценовых линий с использованием VAE/ConvVAE и гибридных схем. Все эксперименты разложены по папкам, данные лежат в `data/`, общие модули — в `src/`.

## Структура
- `data/` — исходные и подготовленные датасеты.
- `experiments/volumes/` — эксперименты по объемам (лог-кумулятивные кривые, детрендинг, гибриды, ансамбли, сравнения).
- `experiments/prices/` — эксперименты по ценовым линиям (детрендинг, Conv1D, сравнения).
- `src/` — общий код (datasets, models, utils).
- `REPORT.md` — полный отчет с метриками и таблицами.

## Примеры визуализаций

### Объемы
![volume_advanced](experiments/volumes/11_advanced_visualizations/plots/volume_advanced_comparison.png)
![volume_fidelity](experiments/volumes/10_fidelity_comparisons/plots/volume_fidelity_comparison.png)
![volume_meta](experiments/volumes/09_meta_ensemble/plots/volume_meta_comparison.png)

### Цены
![price_reconstructions](experiments/prices/01_price_mlp/plots/price_reconstructions.png)
![price_conv_comparison](experiments/prices/03_price_conv/plots/price_conv_comparison.png)
![price_ultra_fidelity](experiments/prices/04_price_conv_ultra/plots/price_ultra_fidelity.png)

## Полный отчет
См. `REPORT.md` — содержит все метрики, таблицы и описания экспериментов.
