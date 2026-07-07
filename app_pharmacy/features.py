"""
Модуль генерации признаков для геомаркетингового анализа.
"""
import h3
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from geopy.distance import geodesic
from . import config


def create_h3_grid(roi_params, resolution=config.H3_RESOLUTION):
    """
    Создает гексагональную сетку H3 для области.
    
    Args:
        roi_params: Словарь с параметрами области (north, south, east, west)
        resolution: Разрешение H3 сетки (по умолчанию из конфига)
    
    Returns:
        GeoDataFrame с H3 ячейками
    """
    # ВАЛИДАЦИЯ входных данных
    if roi_params is None:
        raise ValueError("roi_params не может быть None")
    
    required_keys = ['north', 'south', 'east', 'west']
    missing_keys = [key for key in required_keys if key not in roi_params]
    if missing_keys:
        raise ValueError(f"roi_params должен содержать ключи: {missing_keys}")
    
    # ВАЛИДАЦИЯ координат
    north, south = roi_params['north'], roi_params['south']
    east, west = roi_params['east'], roi_params['west']
    
    if not (-90 <= south < north <= 90):
        raise ValueError(f"Невалидные широты: south={south}, north={north}")
    
    if not (-180 <= west < east <= 180):
        raise ValueError(f"Невалидные долготы: west={west}, east={east}")
    
    if resolution < 0 or resolution > 15:
        raise ValueError(f"Разрешение H3 должно быть от 0 до 15, получено: {resolution}")
    
    try:
        from h3 import LatLngPoly
        
        bbox_coords = [
            (south, west), (south, east), (north, east), (north, west), (south, west)
        ]
        
        h3_polygon = LatLngPoly(bbox_coords[:-1])
        h3_cells = h3.h3shape_to_cells(h3_polygon, resolution)
        
        cells_data = []
        for cell in h3_cells:
            cell_boundary = h3.cell_to_boundary(cell)
            cell_polygon = Polygon([(lng, lat) for lat, lng in cell_boundary])
            cell_center = h3.cell_to_latlng(cell)
            
            cells_data.append({
                'h3_cell': cell,
                'geometry': cell_polygon,
                'center_lat': cell_center[0],
                'center_lon': cell_center[1]
            })
            
        h3_gdf = gpd.GeoDataFrame(cells_data)
        h3_gdf.set_geometry('geometry', inplace=True)
        h3_gdf.crs = 'EPSG:4326'
        return h3_gdf
    except Exception as e:
        print(f"Ошибка создания H3 сетки: {e}")
        return gpd.GeoDataFrame()


def calculate_distance_based_features(h3_grid, poi_data, feature_name, radii=config.RADII):
    """Рассчитывает признаки на основе расстояний"""
    if poi_data.empty:
        for radius in radii:
            h3_grid[f'{feature_name}_count_{radius}m'] = 0
            h3_grid[f'{feature_name}_density_{radius}m'] = 0
        h3_grid[f'{feature_name}_nearest_distance'] = float('inf')
        return h3_grid

    # ВАЖНО: Не отбрасываем Polygon/LineString при смешанных типах геометрии.
    # Всегда приводим к точкам через centroid (для Point centroid == Point).
    poi_points = poi_data.copy()
    poi_points = poi_points[~poi_points.geometry.isna() & ~poi_points.geometry.is_empty].copy()
    poi_points.geometry = poi_points.geometry.centroid

    if poi_points.empty:
        for radius in radii:
            h3_grid[f'{feature_name}_count_{radius}m'] = 0
            h3_grid[f'{feature_name}_density_{radius}m'] = 0
        h3_grid[f'{feature_name}_nearest_distance'] = float('inf')
        return h3_grid

    for idx, cell in h3_grid.iterrows():
        distances = []
        for _, poi in poi_points.iterrows():
            if poi.geometry and not poi.geometry.is_empty:
                poi_lat, poi_lon = poi.geometry.y, poi.geometry.x
                dist = geodesic((cell.center_lat, cell.center_lon), (poi_lat, poi_lon)).meters
                distances.append(dist)
        
        if distances:
            h3_grid.loc[idx, f'{feature_name}_nearest_distance'] = min(distances)
            for radius in radii:
                count = sum(1 for d in distances if d <= radius)
                h3_grid.loc[idx, f'{feature_name}_count_{radius}m'] = count
                area_km2 = np.pi * (radius/1000)**2
                h3_grid.loc[idx, f'{feature_name}_density_{radius}m'] = count / area_km2
        else:
            h3_grid.loc[idx, f'{feature_name}_nearest_distance'] = float('inf')
            for radius in radii:
                h3_grid.loc[idx, f'{feature_name}_count_{radius}m'] = 0
                h3_grid.loc[idx, f'{feature_name}_density_{radius}m'] = 0
                
    return h3_grid


def calculate_area_based_features(h3_grid, area_data, feature_name):
    """Признаки на основе площадей"""
    if area_data.empty:
        h3_grid[f'{feature_name}_coverage'] = 0
        return h3_grid

    polygons = area_data[area_data.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    if polygons.empty:
        h3_grid[f'{feature_name}_coverage'] = 0
        return h3_grid

    # Корректный расчёт площадей: работаем в метрической CRS (обычно UTM)
    if h3_grid.crs is None:
        h3_grid = h3_grid.set_crs("EPSG:4326")
    if polygons.crs is None:
        polygons = polygons.set_crs("EPSG:4326")
    try:
        metric_crs = h3_grid.estimate_utm_crs()
        if metric_crs is None:
            raise ValueError("estimate_utm_crs вернул None")
    except Exception:
        metric_crs = "EPSG:3857"  # fallback (менее точно)

    grid_proj = h3_grid.to_crs(metric_crs)
    poly_proj = polygons.to_crs(metric_crs)

    for idx, cell in h3_grid.iterrows():
        # Берём геометрию ячейки в метрах
        cell_geom = grid_proj.loc[idx].geometry
        cell_area_m2 = cell_geom.area
        total_intersection_m2 = 0.0

        for _, area in poly_proj.iterrows():
            if area.geometry and not area.geometry.is_empty:
                if cell_geom.intersects(area.geometry):
                    intersection = cell_geom.intersection(area.geometry)
                    total_intersection_m2 += intersection.area

        coverage = min(total_intersection_m2 / cell_area_m2, 1.0) if cell_area_m2 > 0 else 0.0
        h3_grid.loc[idx, f'{feature_name}_coverage'] = coverage
        
    return h3_grid


def calculate_road_features(h3_grid, roads_data):
    """Признаки дорожной сети"""
    if roads_data.empty:
        h3_grid['road_density'] = 0
        return h3_grid

    # Корректный расчёт длин: работаем в метрической CRS (обычно UTM)
    if h3_grid.crs is None:
        h3_grid = h3_grid.set_crs("EPSG:4326")
    if roads_data.crs is None:
        roads_data = roads_data.set_crs("EPSG:4326")
    try:
        metric_crs = h3_grid.estimate_utm_crs()
        if metric_crs is None:
            raise ValueError("estimate_utm_crs вернул None")
    except Exception:
        metric_crs = "EPSG:3857"  # fallback (менее точно)

    grid_proj = h3_grid.to_crs(metric_crs)
    roads_proj = roads_data.to_crs(metric_crs)
    
    for idx, cell in h3_grid.iterrows():
        cell_geom = grid_proj.loc[idx].geometry
        cell_area_m2 = cell_geom.area
        length_m = 0.0

        for _, road in roads_proj.iterrows():
            if road.geometry and cell_geom.intersects(road.geometry):
                intersection = cell_geom.intersection(road.geometry)
                length_m += intersection.length

        # км/км² = (м / 1000) / (м² / 1e6) = м * 1000 / м²
        h3_grid.loc[idx, 'road_density'] = (length_m * 1000.0 / cell_area_m2) if cell_area_m2 > 0 else 0.0
        
    return h3_grid


def calculate_medical_synergy(distance, decay_type='linear'):
    """
    Расчёт медицинской синергии с разными функциями затухания.
    
    Args:
        distance: Расстояние до медучреждения (м)
        decay_type: 'linear', 'exponential', 'gaussian'
    
    Returns:
        Значение синергии от 0 до 1
    """
    radius = config.FEATURE_CONFIG['medical_synergy_radius']
    
    if distance >= radius:
        return 0.0
    
    if decay_type == 'linear':
        return max(0, 1 - distance / radius)
    elif decay_type == 'exponential':
        # Экспоненциальное затухание (более реалистичное)
        return np.exp(-3 * distance / radius)
    elif decay_type == 'gaussian':
        # Гауссово затухание
        return np.exp(-0.5 * (distance / (radius / 2)) ** 2)
    else:
        return max(0, 1 - distance / radius)


def calculate_custom_features(h3_grid):
    """
    Дополнительные признаки: многофункциональность и медицинская синергия.
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками и признаками
    
    Returns:
        GeoDataFrame с добавленными кастомными признаками
    """
    # ВАЛИДАЦИЯ входных данных
    if h3_grid is None or h3_grid.empty:
        raise ValueError("h3_grid не может быть пустым")
    
    features = ['transport_subway_count_500m', 'transport_ground_count_500m', 
                'residential_count_500m', 'medical_count_500m', 
                'office_count_500m', 'retail_count_500m']
    
    bool_cols = []
    for f in features:
        if f in h3_grid.columns:
            col_name = f'{f}_bool'
            h3_grid[col_name] = (h3_grid[f] > 0).astype(int)
            bool_cols.append(col_name)
            
    if bool_cols:
        h3_grid['multifunctionality_index'] = h3_grid[bool_cols].sum(axis=1)
        h3_grid.drop(columns=bool_cols, inplace=True)
    
    # Медицинская синергия с настраиваемым decay
    if 'medical_nearest_distance' in h3_grid.columns:
        decay_type = config.FEATURE_CONFIG.get('medical_synergy_decay', 'linear')
        # ВАЛИДАЦИЯ: Обработка NaN и inf перед применением функции
        medical_dist = h3_grid['medical_nearest_distance'].fillna(
            config.FEATURE_CONFIG['default_distance_fillna']
        )
        medical_dist = medical_dist.replace([np.inf, -np.inf], 
                                           config.FEATURE_CONFIG['default_distance_fillna'])
        h3_grid['medical_synergy'] = medical_dist.apply(
            lambda d: calculate_medical_synergy(d, decay_type)
        )
    
    return h3_grid


def calculate_competitor_features(h3_grid, pharmacies_data):
    """
    Анализ типов конкурентов (сетевые vs одиночные).
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками
        pharmacies_data: GeoDataFrame с данными об аптеках
    
    Returns:
        GeoDataFrame с добавленными признаками конкурентов
    """
    # ВАЛИДАЦИЯ: Унифицированная обработка пустых данных
    if pharmacies_data is None or pharmacies_data.empty:
        h3_grid['competitor_chain_count_500m'] = 0
        h3_grid['competitor_chain_density_500m'] = 0.0
        h3_grid['competitor_chain_nearest_distance'] = config.FEATURE_CONFIG['default_distance_fillna']
        return h3_grid
    
    # ВАЛИДАЦИЯ: Проверка наличия колонки 'name'
    if 'name' not in pharmacies_data.columns:
        print("⚠️ pharmacies_data не содержит колонку 'name', создаем пустые признаки конкурентов")
        h3_grid['competitor_chain_count_500m'] = 0
        h3_grid['competitor_chain_density_500m'] = 0.0
        h3_grid['competitor_chain_nearest_distance'] = config.FEATURE_CONFIG['default_distance_fillna']
        return h3_grid
    
    # Используем список сетей из конфигурации
    known_chains = config.KNOWN_PHARMACY_CHAINS
    
    pharmacies_data = pharmacies_data.copy()
    pharmacies_data['is_chain'] = pharmacies_data['name'].fillna('').str.lower().apply(
        lambda x: any(chain in x for chain in known_chains)
    )
    
    chain_pharmacies = pharmacies_data[pharmacies_data['is_chain']]

    # Используем быстрый и метрически корректный расчёт через KDTree (как и для остальных POI)
    try:
        from . import parallel_processing
        h3_grid = parallel_processing.scipy_kdtree_features(
            h3_grid, chain_pharmacies, 'competitor_chain', radii=[500]
        )
    except Exception:
        # Fallback на медленный расчёт (geodesic) если параллельный модуль недоступен
        h3_grid = calculate_distance_based_features(h3_grid, chain_pharmacies, 'competitor_chain', radii=[500])
    
    return h3_grid


def add_target_variable(h3_grid, pharmacies_data):
    """
    Целевая переменная: наличие аптеки.
    
    ВАЖНО: Эта функция используется только в features.py как fallback.
    Основная реализация находится в parallel_processing.py для оптимизации.
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками
        pharmacies_data: GeoDataFrame с данными об аптеках
    
    Returns:
        GeoDataFrame с добавленной колонкой has_pharmacy
    """
    # ВАЛИДАЦИЯ входных данных
    if h3_grid is None or h3_grid.empty:
        raise ValueError("h3_grid не может быть пустым")
    
    h3_grid['has_pharmacy'] = 0
    
    # ВАЛИДАЦИЯ: Унифицированная обработка пустых данных
    if pharmacies_data is None or pharmacies_data.empty:
        return h3_grid
    
    buffer_radius = config.FEATURE_CONFIG['target_buffer_radius']
    
    for idx, cell in h3_grid.iterrows():
        has = False
        for _, pharm in pharmacies_data.iterrows():
            pt = pharm.geometry.centroid if pharm.geometry.geom_type != 'Point' else pharm.geometry
            dist = geodesic((cell.center_lat, cell.center_lon), (pt.y, pt.x)).meters
            if dist <= buffer_radius:
                has = True
                break
        h3_grid.loc[idx, 'has_pharmacy'] = 1 if has else 0
        
    return h3_grid


def remove_highly_correlated_features(X, threshold=None):
    """
    Удаление высококоррелированных признаков.
    
    Args:
        X: DataFrame с признаками
        threshold: Порог корреляции (по умолчанию из конфига)
    
    Returns:
        X_filtered, removed_features
    """
    if threshold is None:
        threshold = config.FEATURE_CONFIG['correlation_threshold']
    
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"⚠️ Удалено {len(to_drop)} высококоррелированных признаков (r > {threshold}):")
        for col in to_drop[:5]:
            print(f"   - {col}")
        if len(to_drop) > 5:
            print(f"   ... и ещё {len(to_drop) - 5}")
    
    return X.drop(columns=to_drop), to_drop


def calculate_vif(X):
    """
    Расчёт Variance Inflation Factor для обнаружения мультиколлинеарности.
    
    Returns:
        DataFrame с VIF для каждого признака
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Удаляем константные колонки
    X_clean = X.loc[:, X.std() > 0].copy()
    
    vif_data = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_clean.values, i)
            vif_data.append({'feature': col, 'VIF': vif})
        except Exception:
            vif_data.append({'feature': col, 'VIF': np.inf})
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    return vif_df


def remove_high_vif_features(X, threshold=None):
    """
    Итеративное удаление признаков с высоким VIF.
    
    Args:
        X: DataFrame с признаками
        threshold: Порог VIF (по умолчанию из конфига)
    
    Returns:
        X_filtered, removed_features
    """
    
    if threshold is None:
        threshold = config.FEATURE_CONFIG['vif_threshold']
    
    X_work = X.copy()
    removed = []
    
    while True:
        vif_df = calculate_vif(X_work)
        max_vif = vif_df['VIF'].max()
        
        if max_vif <= threshold or len(X_work.columns) <= 2:
            break
        
        # Удаляем признак с наибольшим VIF
        worst_feature = vif_df.iloc[0]['feature']
        removed.append(worst_feature)
        X_work = X_work.drop(columns=[worst_feature])
    
    if removed:
        print(f"⚠️ Удалено {len(removed)} признаков с VIF > {threshold}:")
        for col in removed[:5]:
            print(f"   - {col}")
        if len(removed) > 5:
            print(f"   ... и ещё {len(removed) - 5}")
    
    return X_work, removed
