"""
Модуль параллельной обработки данных.

Использует Pandarallel для параллелизации apply-операций,
SciPy cKDTree для пространственных запросов,
ThreadPoolExecutor для I/O операций.
"""

import time
import multiprocessing as mp
import numpy as np
import geopandas as gpd
import h3
from scipy.spatial import cKDTree
from shapely import STRtree
from concurrent.futures import ThreadPoolExecutor, as_completed
from pandarallel import pandarallel
from . import config

N_JOBS = max(1, mp.cpu_count() - 1)
print(f"🔧 Инициализация параллельного движка (Ядер: {N_JOBS})")

try:
    pandarallel.initialize(nb_workers=N_JOBS, progress_bar=True, verbose=1)
except Exception as e:
    print(f"⚠️ Не удалось инициализировать pandarallel: {e}")
    pandarallel.initialize(nb_workers=N_JOBS, progress_bar=True, use_memory_fs=False, verbose=1)

def parallel_load_osm_data(roi_geometry):
    """Параллельная загрузка данных OSM."""
    from . import data_loader
    
    def load_single_type(key_tags):
        key, tags = key_tags
        return key, data_loader.safe_get_osm_data(tags, roi_geometry, key)
    
    tasks = list(config.OSM_TAGS.items())
    osm_data = {}
    
    with ThreadPoolExecutor(max_workers=min(len(tasks), N_JOBS)) as executor:
        future_to_key = {executor.submit(load_single_type, task): task[0] for task in tasks}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                r_key, r_data = future.result()
                osm_data[r_key] = r_data
            except Exception:
                osm_data[key] = gpd.GeoDataFrame()
    
    return osm_data

def _project_to_meters(gdf):
    """Проекция в метрическую систему (Web Mercator) для расчетов"""
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(epsg=3857)

def scipy_kdtree_features(h3_grid, poi_data, feature_name, radii=config.RADII):
    """Расчет дистанционных признаков через KDTree."""
    start_time = time.time()
    print(f"  📊 {feature_name}: начало обработки SciPy KDTree...")
    
    if poi_data.empty:
        for r in radii:
            h3_grid[f'{feature_name}_count_{r}m'] = 0
            h3_grid[f'{feature_name}_density_{r}m'] = 0
        h3_grid[f'{feature_name}_nearest_distance'] = 10000
        return h3_grid

    poi_points = poi_data.copy()
    geom_types = poi_points.geometry.geom_type.unique()
    if not (len(geom_types) == 1 and geom_types[0] == 'Point'):
        poi_points.geometry = poi_points.geometry.centroid
        
    poi_proj = _project_to_meters(poi_points)
    grid_proj = _project_to_meters(h3_grid)
    
    poi_centroids = poi_proj.geometry.centroid
    poi_coords = np.column_stack((poi_centroids.x, poi_centroids.y))
    
    grid_centroids = grid_proj.geometry.centroid
    grid_coords = np.column_stack((grid_centroids.x, grid_centroids.y))
    
    tree = cKDTree(poi_coords)
    
    distances, _ = tree.query(grid_coords, k=1, workers=N_JOBS)
    h3_grid[f'{feature_name}_nearest_distance'] = distances
    
    for r in radii:
        indices = tree.query_ball_point(grid_coords, r, workers=N_JOBS)
        counts = np.array([len(i) for i in indices])
        
        h3_grid[f'{feature_name}_count_{r}m'] = counts
        area_km2 = np.pi * (r/1000)**2
        h3_grid[f'{feature_name}_density_{r}m'] = counts / area_km2
        
    print(f"  ✓ {feature_name}: готово за {time.time() - start_time:.2f} сек")
    return h3_grid

def parallel_area_features(h3_grid, area_data, feature_name):
    """Расчет площадных признаков с использованием Pandarallel и STRtree."""
    start_time = time.time()
    print(f"  📊 {feature_name}_coverage: начало параллельной обработки...")
    
    if area_data.empty:
        h3_grid[f'{feature_name}_coverage'] = 0
        return h3_grid

    polygons = area_data[area_data.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    if polygons.empty:
        h3_grid[f'{feature_name}_coverage'] = 0
        return h3_grid
        
    cell_area_km2 = h3.average_hexagon_area(config.H3_RESOLUTION, 'km^2')
    
    tree = STRtree(polygons.geometry.values)
    
    def calculate_coverage(cell_geom):
        possible_idx = tree.query(cell_geom, predicate='intersects')
        
        total_intersection = 0
        for idx in possible_idx:
            poly = polygons.geometry.iloc[idx]
            intersection = cell_geom.intersection(poly)
            total_intersection += intersection.area * 12321 
            
        return min(total_intersection / cell_area_km2, 1.0) if cell_area_km2 > 0 else 0

    h3_grid[f'{feature_name}_coverage'] = h3_grid.geometry.parallel_apply(calculate_coverage)
    
    print(f"  ✓ {feature_name}_coverage: готово за {time.time() - start_time:.2f} сек")
    return h3_grid

def parallel_road_features(h3_grid, roads_data):
    """Расчет дорожных признаков с использованием Pandarallel."""
    start_time = time.time()
    print(f"  📊 road_density: начало параллельной обработки ({len(roads_data)} сегментов)...")
    
    if roads_data.empty:
        h3_grid['road_density'] = 0
        return h3_grid
    
    cell_area_km2 = h3.average_hexagon_area(config.H3_RESOLUTION, 'km^2')
    tree = STRtree(roads_data.geometry.values)
    
    def calculate_density(cell_geom):
        possible_idx = tree.query(cell_geom, predicate='intersects')
        length_km = 0
        for idx in possible_idx:
            road = roads_data.geometry.iloc[idx]
            intersection = cell_geom.intersection(road)
            length_km += intersection.length * 111
            
        return length_km / cell_area_km2 if cell_area_km2 > 0 else 0
    
    h3_grid['road_density'] = h3_grid.geometry.parallel_apply(calculate_density)
    
    print(f"  ✓ road_density: готово за {time.time() - start_time:.2f} сек")
    return h3_grid

def parallel_target_variable(h3_grid, pharmacies_data):
    """Быстрый расчет целевой переменной через KDTree"""
    h3_grid = scipy_kdtree_features(h3_grid, pharmacies_data, 'temp_target', radii=[100])
    h3_grid['has_pharmacy'] = (h3_grid['temp_target_count_100m'] > 0).astype(int)
    temp_cols = [col for col in h3_grid.columns if col.startswith('temp_target')]
    h3_grid.drop(columns=temp_cols, inplace=True)
    return h3_grid

