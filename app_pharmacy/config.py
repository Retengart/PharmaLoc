"""
Единая конфигурация проекта геомаркетингового анализа.
Все параметры собраны в одном месте для удобства настройки и воспроизводимости.
"""
import os

# =============================================================================
# ОБЛАСТЬ ИССЛЕДОВАНИЯ
# =============================================================================
ROI_PLACE_NAME = "Северо-Восточный административный округ, Москва, Россия"
BACKUP_COORDS = {
    'north': 55.9500, 'south': 55.7500,
    'east': 37.8000, 'west': 37.5000,
    'center_lat': 55.8500, 'center_lon': 37.6500
}

# Округ для out-of-domain валидации
VALIDATION_PLACE_NAME = "Западный административный округ, Москва, Россия"
VALIDATION_COORDS = {
    'north': 55.8000, 'south': 55.6500,
    'east': 37.5000, 'west': 37.3000,
    'center_lat': 55.7250, 'center_lon': 37.4000
}

# =============================================================================
# ГЕОМЕТРИЧЕСКИЕ КОНСТАНТЫ
# =============================================================================
GEOM_CONFIG = {
    'deg_to_meters': 111000,           # Конвертация градусов в метры (на широте Москвы)
    'deg_to_km': 111,                  # Конвертация градусов в километры
    'deg2_to_km2': 12321,              # Квадратные градусы в км² (111*111)
    'earth_radius_km': 6371,           # Радиус Земли
}

# =============================================================================
# ПАРАМЕТРЫ H3 СЕТКИ
# =============================================================================
H3_RESOLUTION = 9  # ~174м между центрами гексов

# =============================================================================
# РАДИУСЫ АНАЛИЗА (метры)
# =============================================================================
RADII = [200, 500, 1000]
MIN_DISTANCE_TO_COMPETITOR = 300

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
FEATURE_CONFIG = {
    # Заполнение пропусков
    'default_distance_fillna': 10000,  # Метры для отсутствующих расстояний
    'default_count_fillna': 0,
    'default_density_fillna': 0,
    
    # Медицинская синергия
    'medical_synergy_radius': 300,     # Радиус для расчёта синергии (м)
    'medical_synergy_decay': 'linear', # 'linear', 'exponential', 'gaussian'
    
    # Целевая переменная
    'target_buffer_radius': 100,       # Буфер для определения наличия аптеки (м)
    
    # Порог корреляции для удаления признаков
    'correlation_threshold': 0.95,
    'vif_threshold': 10.0,             # Variance Inflation Factor
}

# Известные аптечные сети (для анализа конкурентов)
KNOWN_PHARMACY_CHAINS = [
    'ригла', '36.6', 'горздрав', 'планета здоровья', 'апрель', 
    'вита', 'неофарм', 'столички', 'самсон-фарма', 'доктор столетов',
    'аптека.ру', 'здоров.ру', 'озерки', 'радуга', 'фармленд',
    'максавит', 'ладушка', 'фармакопейка', 'мелодия здоровья'
]

# =============================================================================
# OSM ТЕГИ ДЛЯ СБОРА ДАННЫХ
# =============================================================================
OSM_TAGS = {
    'pharmacies': {'amenity': 'pharmacy'},
    'transport_subway': {
        'station': 'subway',
        'railway': 'station'
    },
    'transport_ground': {
        'highway': 'bus_stop',
        'public_transport': ['stop_position', 'platform']
    },
    'residential': {
        'building': ['residential', 'apartments', 'house'],
        'landuse': 'residential'
    },
    'medical': {
        'amenity': ['hospital', 'clinic', 'doctors'],
        'healthcare': True
    },
    'offices': {
        'building': 'office',
        'landuse': 'commercial',
        'office': True
    },
    'retail': {
        'shop': True,
        'amenity': ['marketplace', 'cafe', 'restaurant', 'fast_food']
    },
    'parking': {
        'amenity': 'parking'
    },
    'pedestrian': {
        'highway': ['pedestrian', 'footway'],
        'area:highway': 'pedestrian'
    }
}

# =============================================================================
# ВЕСА ДЛЯ РАСЧЕТА ПОТЕНЦИАЛА
# =============================================================================
POTENTIAL_WEIGHTS = {
    'medical_synergy': 0.25,
    'transport_subway_accessibility': 0.15,
    'transport_ground_accessibility': 0.05,
    'residential_density': 0.15,
    'commercial_activity': 0.10,
    'office_density': 0.05,
    'parking_availability': 0.10,
    'pedestrian_accessibility': 0.10,
    'competition_penalty': -0.25
}

# Веса для смешивания rule-based и ML скоров
POTENTIAL_BLEND = {
    'rule_weight': 0.6,
    'ml_weight': 0.4,
}

# =============================================================================
# МАШИННОЕ ОБУЧЕНИЕ
# =============================================================================
ML_CONFIG = {
    # Разбиение данных
    'test_size': 0.3,
    'random_state': 42,
    
    # Optuna
    'optuna_trials': 50,
    'optuna_cv_folds': 5,
    'optuna_timeout': None,
    
    # Random Forest
    'rf_n_estimators': 1000,
    'rf_max_depth_range': (3, 20),
    'rf_min_samples_split_range': (2, 20),
    'rf_min_samples_leaf_range': (1, 10),
    
    # CatBoost
    'cb_iterations': 1000,
    'cb_depth_range': (3, 10),
    'cb_learning_rate_range': (0.01, 0.3),
    'cb_l2_leaf_reg_range': (0.1, 10),
    
    # Class Imbalance
    'smote_k_neighbors': 3,
    'use_tomek_links': True,            # Удаление пограничных примеров
    'class_weight': 'balanced',
    
    # Калибровка вероятностей
    'calibration_method': 'isotonic',   # 'sigmoid' или 'isotonic'
    'calibration_cv': 5,
    
    # Порог переобучения
    'overfit_threshold': 0.99,
    
    # Spatial CV
    'spatial_cv_n_splits': 5,
    'spatial_cv_buffer_km': 0.5,        # Буферная зона между фолдами
}

# Признаки с утечкой данных
LEAKAGE_FEATURES = [
    'pharmacy_nearest_distance',
    'pharmacy_count_200m', 'pharmacy_count_500m', 'pharmacy_count_1000m',
    'pharmacy_density_200m', 'pharmacy_density_500m', 'pharmacy_density_1000m',
    'competitor_chain_nearest_distance',
    'competitor_chain_count_500m', 'competitor_chain_density_500m'
]

# =============================================================================
# КЛАСТЕРИЗАЦИЯ
# =============================================================================
CLUSTER_CONFIG = {
    'max_clusters': 10,
    'min_clusters': 2,
    'n_init': 10,
    'random_state': 42,
}

# =============================================================================
# БИЗНЕС-МЕТРИКИ
# =============================================================================
BUSINESS_CONFIG = {
    'precision_at_k': [5, 10, 20],      # K для Precision@K
    'bootstrap_n_iterations': 1000,     # Итерации для bootstrap CI
    'bootstrap_ci': 0.95,               # Уровень доверия
    
    # Оценка потенциальной выручки (упрощённая модель)
    'avg_pharmacy_revenue_monthly': 3_000_000,  # Рублей/месяц (средняя)
    'location_quality_impact': 0.3,     # Влияние качества локации на выручку (±30%)
}

# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================
VIZ_CONFIG = {
    'figure_dpi': 100,
    'figure_size_small': (8, 6),
    'figure_size_medium': (10, 8),
    'figure_size_large': (14, 10),
    'figure_size_wide': (16, 6),
    'cmap_correlation': 'RdBu_r',
    'cmap_heatmap': 'YlOrRd',
    'cmap_clusters': 'Set2',
    'cmap_potential': 'RdYlGn',
    'map_tiles': 'cartodbpositron',
    'map_zoom_start': 12,
    'top_n_features': 20,
    'tsne_perplexity': 30,
    'tsne_random_state': 42,
}

# =============================================================================
# ВЫВОД РЕЗУЛЬТАТОВ
# =============================================================================
OUTPUT_CONFIG = {
    'top_n_recommendations': 10,
    'top_n_features_display': 7,
    'decimal_places': 4,
}

# =============================================================================
# ПУТИ К ФАЙЛАМ
# =============================================================================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FILES = {
    # Данные
    'roi_params': os.path.join(DATA_DIR, 'roi_parameters.json'),
    'h3_grid_features': os.path.join(DATA_DIR, 'h3_grid_with_features.geojson'),
    'h3_grid_final': os.path.join(DATA_DIR, 'h3_grid_final_results.geojson'),
    'feature_list': os.path.join(DATA_DIR, 'feature_list.json'),
    
    # Модель
    'model': os.path.join(DATA_DIR, 'final_trained_model.pkl'),
    'modeling_results': os.path.join(DATA_DIR, 'comprehensive_modeling_results.json'),
    
    # Рекомендации
    'recommendations': os.path.join(DATA_DIR, 'final_business_recommendations.json'),
    'top_10_summary': os.path.join(DATA_DIR, 'top_10_locations_summary.csv'),
    'top_10_geojson': os.path.join(DATA_DIR, 'top_10_recommended_locations.geojson'),
    
    # Визуализации
    'correlation_matrix': os.path.join(DATA_DIR, 'correlation_matrix.png'),
    'feature_importance': os.path.join(DATA_DIR, 'feature_importance.png'),
    'elbow_plot': os.path.join(DATA_DIR, 'clustering_elbow_method.png'),
    'silhouette_plot': os.path.join(DATA_DIR, 'clustering_silhouette_method.png'),
    'models_comparison': os.path.join(DATA_DIR, 'models_comparison.png'),
    'potential_map': os.path.join(DATA_DIR, 'potential_map.html'),
    'vif_analysis': os.path.join(DATA_DIR, 'vif_analysis.png'),
    'business_metrics': os.path.join(DATA_DIR, 'business_metrics.png'),
    
    # Кластерный анализ
    'cluster_profiles': os.path.join(DATA_DIR, 'cluster_profiles.csv'),
    'cluster_report': os.path.join(DATA_DIR, 'cluster_analysis_report.json'),
}

# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================
LOGGING_CONFIG = {
    'verbose': True,
    'show_warnings': True,
    'progress_bar': True,
}

# =============================================================================
# DATA.MOS.RU ИНТЕГРАЦИЯ
# =============================================================================
DATA_MOS_CONFIG = {
    'enabled': True,
    'api_key': 'f9f69a50-7835-4047-8d2f-de17160137c7',
    'cache_enabled': True,
    'timeout': 30,
    'datasets': {
        'polyclinics_adult': 503,
        'polyclinics_child': 505,
        'hospitals_adult': 517,
        'hospitals_child': 502,
        'dental_clinics': 518,
        'transport_hubs': 1047,
        'population_working': 2084,
        'population_young': 2085,
        'salaries': 2087,
    }
}
