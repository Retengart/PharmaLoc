import os

# Параметры области интереса
ROI_PLACE_NAME = "Отрадное, Москва, Россия"
BACKUP_COORDS = {
    'north': 55.8700, 'south': 55.8400,
    'east': 37.6000, 'west': 37.5700,
    'center_lat': 55.862060, 'center_lon': 37.600262
}

# Параметры сетки H3
H3_RESOLUTION = 9

# Радиусы для анализа (в метрах)
RADII = [200, 500, 1000]

# Теги OSM для сбора данных
OSM_TAGS = {
    'pharmacies': {'amenity': 'pharmacy'},
    'transport': {
        'highway': 'bus_stop',
        'public_transport': ['stop_position', 'platform'],
        'railway': 'station'
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
    }
}

# Веса для расчета потенциала
POTENTIAL_WEIGHTS = {
    'medical_synergy': 0.30,
    'transport_accessibility': 0.25,
    'residential_density': 0.20,
    'commercial_activity': 0.15,
    'office_density': 0.10,
    'competition_penalty': -0.30
}

# Пути к файлам (можно изменить на абсолютные при необходимости)
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

FILES = {
    'roi_params': os.path.join(DATA_DIR, 'roi_parameters.json'),
    'h3_grid_features': os.path.join(DATA_DIR, 'h3_grid_with_features.geojson'),
    'h3_grid_final': os.path.join(DATA_DIR, 'h3_grid_final_results.geojson'),
    'feature_list': os.path.join(DATA_DIR, 'feature_list.json'),
    'model': os.path.join(DATA_DIR, 'final_trained_model.pkl'),
    'modeling_results': os.path.join(DATA_DIR, 'comprehensive_modeling_results.json'),
    'recommendations': os.path.join(DATA_DIR, 'final_business_recommendations.json'),
    'top_10_summary': os.path.join(DATA_DIR, 'top_10_locations_summary.csv'),
    'top_10_geojson': os.path.join(DATA_DIR, 'top_10_recommended_locations.geojson')
}

