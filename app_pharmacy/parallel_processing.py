"""
Модуль параллельной обработки данных для геомаркетингового анализа.

Использует:
- ThreadPoolExecutor для I/O-bound задач (загрузка OSM данных)
- ProcessPoolExecutor для CPU-bound задач (расчет признаков)
- scipy.spatial.cKDTree для сверхбыстрых пространственных запросов
- joblib для параллелизации вычислений в sklearn
"""

import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic
from scipy.spatial import cKDTree
from . import config

# Определяем количество ядер CPU
N_JOBS = max(1, mp.cpu_count() - 1)  # Оставляем одно ядро свободным
print(f"🔧 Параллельная обработка: используется {N_JOBS} ядер CPU")

def parallel_load_osm_data(roi_geometry):
    """
    Параллельная загрузка данных OSM для разных типов объектов.
    Использует ThreadPoolExecutor, т.к. это I/O-bound операции.
    """
    from . import data_loader
    
    def load_single_type(key_tags):
        key, tags = key_tags
        return key, data_loader.safe_get_osm_data(tags, roi_geometry, key)
    
    # Подготовка задач
    tasks = list(config.OSM_TAGS.items())
    
    osm_data = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), N_JOBS)) as executor:
        future_to_key = {executor.submit(load_single_type, task): task[0] for task in tasks}
        
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result_key, result_data = future.result()
                osm_data[result_key] = result_data
            except Exception as e:
                print(f"✗ Ошибка при параллельной загрузке {key}: {e}")
                osm_data[key] = gpd.GeoDataFrame()
    
    return osm_data

def _project_to_meters(gdf):
    """
    Проецирует GeoDataFrame в метрическую систему координат (UTM) для корректной работы KDTree.
    """
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    
    # Автоматически определяем UTM зону через estimate_utm_crs (доступно в новых версиях geopandas)
    # Или используем Web Mercator (EPSG:3857) как универсальный вариант для расчета расстояний на небольших расстояниях
    return gdf.to_crs(epsg=3857)

def scipy_kdtree_features(h3_grid, poi_data, feature_name, radii=config.RADII):
    """
    Расчет признаков с использованием SciPy cKDTree (O(log N)).
    Самый производительный метод для пространственных запросов.
    """
    if poi_data.empty:
        for radius in radii:
            h3_grid[f'{feature_name}_count_{radius}m'] = 0
            h3_grid[f'{feature_name}_density_{radius}m'] = 0
        h3_grid[f'{feature_name}_nearest_distance'] = float('inf')
        return h3_grid

    # Prepare POI points
    if 'Point' not in poi_data.geometry.geom_type.values:
        poi_points = poi_data.copy()
        poi_points.geometry = poi_points.geometry.centroid
    else:
        poi_points = poi_data[poi_data.geometry.geom_type == 'Point'].copy()

    # Проецируем в метры
    poi_points_proj = _project_to_meters(poi_points)
    h3_grid_proj = _project_to_meters(h3_grid)
    
    # Извлекаем координаты (x, y)
    # Для POI (точки) берем координаты напрямую
    poi_coords = np.column_stack((poi_points_proj.geometry.x, poi_points_proj.geometry.y))
    # Используем centroid, так как h3_grid_proj содержит полигоны
    grid_centroids = h3_grid_proj.geometry.centroid
    grid_coords = np.column_stack((grid_centroids.x, grid_centroids.y))
    
    # Строим KDTree
    tree = cKDTree(poi_coords)
    
    # 1. Ближайшее расстояние (query)
    # workers=-1 использует все ядра для параллельного поиска
    distances, _ = tree.query(grid_coords, k=1, workers=N_JOBS)
    h3_grid[f'{feature_name}_nearest_distance'] = distances
    
    # 2. Подсчет соседей в радиусе (query_ball_point)
    # Внимание: query_ball_point возвращает индексы, считаем их количество
    for radius in radii:
        # query_ball_point также поддерживает workers в новых версиях SciPy
        indices_list = tree.query_ball_point(grid_coords, r=radius, workers=N_JOBS)
        counts = np.array([len(x) for x in indices_list])
        
        h3_grid[f'{feature_name}_count_{radius}m'] = counts
        area_km2 = np.pi * (radius/1000)**2
        h3_grid[f'{feature_name}_density_{radius}m'] = counts / area_km2
        
    return h3_grid

def vectorized_distance_features(h3_grid, poi_data, feature_name, radii=config.RADII):
    """Legacy wrapper, now points to optimized scipy implementation"""
    return scipy_kdtree_features(h3_grid, poi_data, feature_name, radii)

def parallel_calculate_multiple_features(h3_grid, osm_data, feature_configs):
    """
    Параллельный расчет нескольких типов признаков одновременно.
    """
    from . import features
    
    def calculate_single_feature(config_item):
        feature_name, data_key, feature_type = config_item
        poi_data = osm_data.get(data_key, gpd.GeoDataFrame())
        
        if feature_type == 'distance':
            # Используем SciPy версию
            return scipy_kdtree_features(h3_grid.copy(), poi_data, feature_name)
        elif feature_type == 'area':
            return features.calculate_area_based_features(h3_grid.copy(), poi_data, feature_name)
        elif feature_type == 'road':
            return features.calculate_road_features(h3_grid.copy(), poi_data)
        else:
            return h3_grid.copy()
    
    # Параллельная обработка
    with ThreadPoolExecutor(max_workers=min(len(feature_configs), N_JOBS)) as executor:
        futures = {executor.submit(calculate_single_feature, config): config[0] 
                  for config in feature_configs}
        
        results = {}
        for future in as_completed(futures):
            feature_name = futures[future]
            try:
                result_gdf = future.result()
                # Объединяем результаты
                for col in result_gdf.columns:
                    if col not in ['h3_cell', 'geometry', 'center_lat', 'center_lon']:
                        if col not in h3_grid.columns:
                            h3_grid[col] = result_gdf[col]
                        else:
                            # Обновляем только если значение не пустое
                            mask = ~result_gdf[col].isna()
                            h3_grid.loc[mask, col] = result_gdf.loc[mask, col]
            except Exception as e:
                print(f"⚠️ Ошибка расчета признака {feature_name}: {e}")
    
    return h3_grid
