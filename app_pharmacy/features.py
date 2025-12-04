"""
Модуль генерации признаков для геомаркетингового анализа.
"""
import h3
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from geopy.distance import geodesic
from . import config


def create_h3_grid(roi_params, resolution=config.H3_RESOLUTION):
    """Создает гексагональную сетку H3 для области"""
    try:
        from h3 import LatLngPoly
        
        north, south = roi_params['north'], roi_params['south']
        east, west = roi_params['east'], roi_params['west']
        
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

    if 'Point' not in poi_data.geometry.geom_type.values:
        poi_points = poi_data.copy()
        poi_points.geometry = poi_points.geometry.centroid
    else:
        poi_points = poi_data[poi_data.geometry.geom_type == 'Point'].copy()

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
    cell_area_km2 = h3.average_hexagon_area(config.H3_RESOLUTION, 'km^2')
    deg2_to_km2 = config.GEOM_CONFIG['deg2_to_km2']

    for idx, cell in h3_grid.iterrows():
        total_intersection = 0
        for _, area in polygons.iterrows():
            if area.geometry and not area.geometry.is_empty:
                if cell.geometry.intersects(area.geometry):
                    intersection = cell.geometry.intersection(area.geometry)
                    area_deg2 = intersection.area
                    total_intersection += area_deg2 * deg2_to_km2
        
        coverage = min(total_intersection / cell_area_km2, 1.0) if cell_area_km2 > 0 else 0
        h3_grid.loc[idx, f'{feature_name}_coverage'] = coverage
        
    return h3_grid


def calculate_road_features(h3_grid, roads_data):
    """Признаки дорожной сети"""
    if roads_data.empty:
        h3_grid['road_density'] = 0
        return h3_grid
        
    cell_area_km2 = h3.average_hexagon_area(config.H3_RESOLUTION, 'km^2')
    deg_to_km = config.GEOM_CONFIG['deg_to_km']
    
    for idx, cell in h3_grid.iterrows():
        length_km = 0
        for _, road in roads_data.iterrows():
            if road.geometry and cell.geometry.intersects(road.geometry):
                intersection = cell.geometry.intersection(road.geometry)
                length_km += intersection.length * deg_to_km
        
        h3_grid.loc[idx, 'road_density'] = length_km / cell_area_km2 if cell_area_km2 > 0 else 0
        
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
    """Дополнительные признаки"""
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
        h3_grid['medical_synergy'] = h3_grid['medical_nearest_distance'].apply(
            lambda d: calculate_medical_synergy(d, decay_type)
        )
    
    return h3_grid


def calculate_competitor_features(h3_grid, pharmacies_data):
    """Анализ типов конкурентов (сетевые vs одиночные)"""
    if pharmacies_data.empty:
        h3_grid['competitor_chain_count_500m'] = 0
        return h3_grid
    
    # Используем список сетей из конфигурации
    known_chains = config.KNOWN_PHARMACY_CHAINS
    
    pharmacies_data = pharmacies_data.copy()
    pharmacies_data['is_chain'] = pharmacies_data['name'].fillna('').str.lower().apply(
        lambda x: any(chain in x for chain in known_chains)
    )
    
    chain_pharmacies = pharmacies_data[pharmacies_data['is_chain']]
    
    h3_grid = calculate_distance_based_features(h3_grid, chain_pharmacies, 'competitor_chain', radii=[500])
    
    return h3_grid


def add_target_variable(h3_grid, pharmacies_data):
    """Целевая переменная: наличие аптеки"""
    h3_grid['has_pharmacy'] = 0
    if pharmacies_data.empty:
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
        except:
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
    import pandas as pd
    
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


# Импорт pandas для calculate_vif
import pandas as pd
