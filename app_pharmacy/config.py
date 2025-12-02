"""
Единая конфигурация проекта геомаркетингового анализа.
"""
import os

# Основной округ для обучения
ROI_PLACE_NAME = "Северо-Восточный административный округ, Москва, Россия"
BACKUP_COORDS = {
    'north': 55.9500, 'south': 55.7500,
    'east': 37.8000, 'west': 37.5000,
    'center_lat': 55.8500, 'center_lon': 37.6500
}

# Округ для валидации (out-of-domain validation)
VALIDATION_PLACE_NAME = "Западный административный округ, Москва, Россия"
VALIDATION_COORDS = {
    'north': 55.8000, 'south': 55.6500,
    'east': 37.5000, 'west': 37.3000,
    'center_lat': 55.7250, 'center_lon': 37.4000
}

H3_RESOLUTION = 9

RADII = [200, 500, 1000]
MIN_DISTANCE_TO_COMPETITOR = 300
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

ML_CONFIG = {
    'test_size': 0.3,
    'random_state': 42,
    'optuna_trials': 50,
    'optuna_cv_folds': 5,
    'optuna_timeout': None,
    'rf_n_estimators': 1000,
    'rf_max_depth_range': (3, 20),
    'rf_min_samples_split_range': (2, 20),
    'rf_min_samples_leaf_range': (1, 10),
    'cb_iterations': 1000,
    'cb_depth_range': (3, 10),
    'cb_learning_rate_range': (0.01, 0.3),
    'cb_l2_leaf_reg_range': (0.1, 10),
    'smote_k_neighbors': 3,
    'overfit_threshold': 0.99,
}

LEAKAGE_FEATURES = [
    'pharmacy_nearest_distance',
    'pharmacy_count_200m', 'pharmacy_count_500m', 'pharmacy_count_1000m',
    'pharmacy_density_200m', 'pharmacy_density_500m', 'pharmacy_density_1000m',
    'competitor_chain_nearest_distance',
    'competitor_chain_count_500m', 'competitor_chain_density_500m'
]

CLUSTER_CONFIG = {
    'max_clusters': 10,
    'min_clusters': 2,
    'n_init': 10,
    'random_state': 42,
}

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

OUTPUT_CONFIG = {
    'top_n_recommendations': 10,
    'top_n_features_display': 7,
    'decimal_places': 4,
}

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
    'top_10_geojson': os.path.join(DATA_DIR, 'top_10_recommended_locations.geojson'),
    'correlation_matrix': os.path.join(DATA_DIR, 'correlation_matrix.png'),
    'feature_importance': os.path.join(DATA_DIR, 'feature_importance.png'),
    'elbow_plot': os.path.join(DATA_DIR, 'clustering_elbow_method.png'),
    'silhouette_plot': os.path.join(DATA_DIR, 'clustering_silhouette_method.png'),
    'models_comparison': os.path.join(DATA_DIR, 'models_comparison.png'),
    'potential_map': os.path.join(DATA_DIR, 'potential_map.html'),
    'cluster_profiles': os.path.join(DATA_DIR, 'cluster_profiles.csv'),
    'cluster_report': os.path.join(DATA_DIR, 'cluster_analysis_report.json'),
}

LOGGING_CONFIG = {
    'verbose': True,
    'show_warnings': True,
    'progress_bar': True,
}
