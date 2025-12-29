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

def _ensure_crs(gdf, crs="EPSG:4326"):
    """Гарантирует наличие CRS у GeoDataFrame."""
    if gdf is None:
        return gdf
    if getattr(gdf, "crs", None) is None:
        return gdf.set_crs(crs)
    return gdf


def _get_metric_crs(reference_gdf):
    """
    Подбирает локальную метрическую CRS (обычно UTM) для корректных расчётов в метрах.

    ВАЖНО: EPSG:3857 (Web Mercator) не даёт «реальные метры» (масштаб зависит от широты).
    Для Москвы искажает расстояния примерно в ~1.7 раза.
    """
    ref = _ensure_crs(reference_gdf, "EPSG:4326")
    try:
        metric_crs = ref.estimate_utm_crs()
        if metric_crs is None:
            raise ValueError("estimate_utm_crs вернул None")
        return metric_crs
    except Exception as e:
        # Fallback на Web Mercator (менее точно), чтобы не падать на редких геометриях
        print(f"⚠️ Не удалось определить UTM CRS ({e}), используется EPSG:3857 (менее точно)")
        return "EPSG:3857"


def _project_to_meters(gdf, metric_crs):
    """Проекция в метрическую систему координат для корректных расчетов в метрах."""
    gdf = _ensure_crs(gdf, "EPSG:4326")
    return gdf.to_crs(metric_crs)

def scipy_kdtree_features(h3_grid, poi_data, feature_name, radii=config.RADII):
    """
    Расчет дистанционных признаков через KDTree.
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками
        poi_data: GeoDataFrame с точками интереса (POI)
        feature_name: Имя признака для создания колонок
        radii: Список радиусов для расчета признаков (в метрах)
    
    Returns:
        GeoDataFrame с добавленными дистанционными признаками
    """
    start_time = time.time()
    print(f"  📊 {feature_name}: начало обработки SciPy KDTree...")
    
    # ВАЛИДАЦИЯ входных данных
    if h3_grid is None or h3_grid.empty:
        raise ValueError("h3_grid не может быть пустым")
    
    if radii is None or len(radii) == 0:
        raise ValueError("radii не может быть пустым")
    
    # ВАЛИДАЦИЯ: Унифицированная обработка пустых данных
    if poi_data is None or poi_data.empty:
        for r in radii:
            h3_grid[f'{feature_name}_count_{r}m'] = 0
            h3_grid[f'{feature_name}_density_{r}m'] = 0.0
        h3_grid[f'{feature_name}_nearest_distance'] = config.FEATURE_CONFIG['default_distance_fillna']
        print(f"  ✓ {feature_name}: данные пусты, установлены значения по умолчанию")
        return h3_grid

    poi_points = poi_data.copy()
    # Убираем невалидные геометрии
    poi_points = poi_points[~poi_points.geometry.isna() & ~poi_points.geometry.is_empty].copy()

    metric_crs = _get_metric_crs(h3_grid)
    poi_proj = _project_to_meters(poi_points, metric_crs)
    grid_proj = _project_to_meters(h3_grid, metric_crs)
    
    # Берем центроиды уже в МЕТРИЧЕСКОЙ CRS (без предупреждений и с корректной геометрией)
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
    """
    Расчет площадных признаков с использованием Pandarallel и STRtree.
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками
        area_data: GeoDataFrame с полигональными данными
        feature_name: Имя признака для создания колонок
    
    Returns:
        GeoDataFrame с добавленными признаками покрытия
    """
    start_time = time.time()
    print(f"  📊 {feature_name}_coverage: начало параллельной обработки...")
    
    # ВАЛИДАЦИЯ: Унифицированная обработка пустых данных
    if area_data is None or area_data.empty:
        h3_grid[f'{feature_name}_coverage'] = 0.0
        print(f"  ✓ {feature_name}_coverage: данные пусты, установлено 0")
        return h3_grid

    # ВАЛИДАЦИЯ: Проверка наличия геометрии
    if 'geometry' not in area_data.columns:
        print(f"  ⚠️ {feature_name}: отсутствует колонка geometry, установлено 0")
        h3_grid[f'{feature_name}_coverage'] = 0.0
        return h3_grid

    polygons = area_data[area_data.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    if polygons.empty:
        h3_grid[f'{feature_name}_coverage'] = 0.0
        print(f"  ✓ {feature_name}_coverage: нет полигонов, установлено 0")
        return h3_grid

    # Проецируем в локальную метрическую CRS для корректного расчета площадей
    metric_crs = _get_metric_crs(h3_grid)
    grid_proj = _project_to_meters(h3_grid, metric_crs)
    polygons_proj = _project_to_meters(polygons, metric_crs)

    tree = STRtree(polygons_proj.geometry.values)
    
    def calculate_coverage(cell_geom):
        possible_idx = tree.query(cell_geom, predicate='intersects')

        total_intersection_m2 = 0.0
        for idx in possible_idx:
            poly = polygons_proj.geometry.iloc[idx]
            try:
                intersection = cell_geom.intersection(poly)
                if not intersection.is_empty:
                    total_intersection_m2 += intersection.area
            except Exception:
                # На редких невалидных геометриях пропускаем
                continue

        cell_area_m2 = cell_geom.area
        return min(total_intersection_m2 / cell_area_m2, 1.0) if cell_area_m2 > 0 else 0.0

    # Считаем на проецированной геометрии, записываем результат в исходный GeoDataFrame
    h3_grid[f'{feature_name}_coverage'] = grid_proj.geometry.parallel_apply(calculate_coverage)
    
    print(f"  ✓ {feature_name}_coverage: готово за {time.time() - start_time:.2f} сек")
    return h3_grid

def parallel_road_features(h3_grid, roads_data):
    """
    Расчет дорожных признаков с использованием Pandarallel.
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками
        roads_data: GeoDataFrame с дорожной сетью
    
    Returns:
        GeoDataFrame с добавленным признаком road_density
    """
    start_time = time.time()
    
    # ВАЛИДАЦИЯ: Унифицированная обработка пустых данных
    if roads_data is None or roads_data.empty:
        h3_grid['road_density'] = 0.0
        print("  ✓ road_density: данные пусты, установлено 0")
        return h3_grid
    
    print(f"  📊 road_density: начало параллельной обработки ({len(roads_data)} сегментов)...")
    
    # ВАЛИДАЦИЯ: Проверка наличия геометрии
    if 'geometry' not in roads_data.columns:
        print("  ⚠️ roads_data: отсутствует колонка geometry, установлено 0")
        h3_grid['road_density'] = 0.0
        return h3_grid
    
    # Проецируем в локальную метрическую CRS для корректного расчета длин
    metric_crs = _get_metric_crs(h3_grid)
    grid_proj = _project_to_meters(h3_grid, metric_crs)
    roads_proj = _project_to_meters(roads_data, metric_crs)

    tree = STRtree(roads_proj.geometry.values)
    
    def calculate_density(cell_geom):
        possible_idx = tree.query(cell_geom, predicate='intersects')
        length_m = 0.0
        for idx in possible_idx:
            road = roads_proj.geometry.iloc[idx]
            try:
                intersection = cell_geom.intersection(road)
                if not intersection.is_empty:
                    length_m += intersection.length
            except Exception:
                continue

        cell_area_m2 = cell_geom.area
        # км/км² = (м / 1000) / (м² / 1e6) = м * 1000 / м²
        return (length_m * 1000.0 / cell_area_m2) if cell_area_m2 > 0 else 0.0
    
    h3_grid['road_density'] = grid_proj.geometry.parallel_apply(calculate_density)
    
    print(f"  ✓ road_density: готово за {time.time() - start_time:.2f} сек")
    return h3_grid

def parallel_target_variable(h3_grid, pharmacies_data):
    """
    Быстрый расчет целевой переменной через KDTree.
    
    УЛУЧШЕНО: Учитывает не только наличие аптеки в радиусе 100м,
    но и плотность аптек в радиусе 500м для более реалистичной оценки.
    Ячейка считается положительной, если:
    - В радиусе 100м есть хотя бы одна аптека (точное попадание)
    ИЛИ
    - В радиусе 500м есть аптеки И плотность не превышает разумный порог (умеренная конкуренция)
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками
        pharmacies_data: GeoDataFrame с данными об аптеках
    
    Returns:
        GeoDataFrame с добавленной колонкой has_pharmacy
    """
    # ВАЛИДАЦИЯ входных данных
    if h3_grid is None or h3_grid.empty:
        raise ValueError("h3_grid не может быть пустым")
    
    # ВАЛИДАЦИЯ: Унифицированная обработка пустых данных
    if pharmacies_data is None or pharmacies_data.empty:
        h3_grid['has_pharmacy'] = 0
        print("  ⚠️ Данные аптек пусты, все ячейки помечены как отрицательные (has_pharmacy=0)")
        return h3_grid
    
    # Используем временные признаки для расчета целевой переменной
    h3_grid = scipy_kdtree_features(h3_grid, pharmacies_data, 'temp_target', radii=[100, 500])
    
    # Улучшенная логика: учитываем и близость, и разумную плотность
    # Ячейка положительная если:
    # 1. В радиусе 100м есть аптека (точное попадание)
    # 2. ИЛИ в радиусе 500м есть аптеки с умеренной плотностью (до 10 аптек/км²)
    has_nearby = (h3_grid['temp_target_count_100m'] > 0)
    moderate_density = (h3_grid['temp_target_count_500m'] > 0) & (h3_grid['temp_target_density_500m'] <= 10.0)
    
    h3_grid['has_pharmacy'] = (has_nearby | moderate_density).astype(int)
    
    # Удаляем временные признаки
    temp_cols = [col for col in h3_grid.columns if col.startswith('temp_target')]
    h3_grid.drop(columns=temp_cols, inplace=True)
    
    return h3_grid

