# Проект: Геомаркетинговый анализ аптек

Этот проект был реструктурирован из одного скрипта в модульное приложение.

## Структура

- `main.py`: Основной файл запуска. Управляет всем процессом.
- `config.py`: Конфигурация (параметры ROI, теги OSM, настройки модели).
- `utils.py`: Вспомогательные функции (проверка зависимостей).
- `data_loader.py`: Загрузка и сохранение данных из OpenStreetMap.
- `features.py`: Генерация сетки H3 и расчет пространственных признаков.
- `modeling.py`: Обучение моделей машинного обучения и кластеризация.
- `analysis.py`: Расчет потенциала и формирование рекомендаций.
- `visualization.py`: Создание карт и графиков.

## Запуск

Убедитесь, что вы находитесь в корневой папке проекта (на уровень выше `app_pharmacy`).

1. Установите зависимости:
   ```bash
   uv pip install -r app_pharmacy/requirements.txt
   ```

2. Запустите анализ:
   ```bash
   python3 -m app_pharmacy.main
   ```

## Использование в Jupyter Notebook

Если вы хотите использовать этот код в Notebook, вы можете импортировать модули:

```python
from app_pharmacy import data_loader, features, modeling, analysis, visualization, config

# Пример вызова функций
roi_geometry, roi_params, _ = data_loader.get_roi_geometry()
...
```

