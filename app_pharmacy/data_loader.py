import osmnx as ox
import geopandas as gpd
import numpy as np
import json
import os
from . import config

def setup_osmnx():
    ox.settings.use_cache = True
    ox.settings.log_console = True

def get_roi_geometry(place_name=config.ROI_PLACE_NAME):
    """
    Получает геометрию области интереса (ROI) с валидацией.
    
    Args:
        place_name: Название района для поиска в OSM
    
    Returns:
        tuple: (roi_geometry, roi_params, gdf_district)
    """
    try:
        gdf_district = ox.geocode_to_gdf(place_name)
        
        # ВАЛИДАЦИЯ: Проверка на пустой результат
        if gdf_district.empty:
            print(f"❌ Район '{place_name}' не найден в OSM")
            print("   Используется fallback координаты из конфига")
            return None, config.BACKUP_COORDS, None
        
        # ВАЛИДАЦИЯ: Проверка на множественные результаты
        if len(gdf_district) > 1:
            print(f"⚠️ Найдено {len(gdf_district)} районов с именем '{place_name}'")
            print("   Используется первый результат")
            # Можно добавить логику выбора лучшего совпадения
        
        # Безопасное извлечение геометрии
        roi_geometry = gdf_district.geometry.iloc[0]
        
        # ВАЛИДАЦИЯ: Проверка на валидную геометрию
        if roi_geometry is None or roi_geometry.is_empty:
            print(f"❌ Геометрия района '{place_name}' пуста или невалидна")
            return None, config.BACKUP_COORDS, None
        
        bounds = gdf_district.total_bounds
        
        # ВАЛИДАЦИЯ: Проверка на валидные границы
        if not all(np.isfinite(bounds)):
            print("⚠️ Границы района содержат невалидные значения, используются fallback координаты")
            return None, config.BACKUP_COORDS, None
        
        # north, south, east, west
        roi_params = {
            'north': bounds[3], 'south': bounds[1],
            'east': bounds[2], 'west': bounds[0],
            'center_lat': (bounds[3] + bounds[1]) / 2,
            'center_lon': (bounds[2] + bounds[0]) / 2,
            'place_name': place_name
        }
        
        return roi_geometry, roi_params, gdf_district
        
    except Exception as e:
        import traceback
        print(f"❌ Ошибка при получении границ района '{place_name}': {e}")
        print(f"   Детали: {traceback.format_exc()}")
        print("   Используется fallback координаты из конфига")
        # Fallback
        return None, config.BACKUP_COORDS, None

def safe_get_osm_data(tags, geometry, name):
    """
    Безопасное получение данных OSM с проверкой качества.
    
    Returns:
        GeoDataFrame с данными или пустой GeoDataFrame при ошибке
    """
    try:
        data = ox.features_from_polygon(geometry, tags=tags)
        
        # Проверка качества данных
        if data.empty:
            print(f"⚠️ {name}: данные пусты")
            return gpd.GeoDataFrame()
        
        # Проверка на наличие геометрии
        if 'geometry' not in data.columns:
            print(f"⚠️ {name}: отсутствует колонка geometry")
            return gpd.GeoDataFrame()
        
        # Проверка на валидные геометрии
        invalid_geom = data.geometry.isna().sum()
        if invalid_geom > 0:
            print(f"⚠️ {name}: найдено {invalid_geom} объектов с невалидной геометрией")
            data = data[~data.geometry.isna()].copy()
        
        # Проверка на пустые геометрии
        empty_geom = data.geometry.is_empty.sum()
        if empty_geom > 0:
            print(f"⚠️ {name}: найдено {empty_geom} объектов с пустой геометрией")
            data = data[~data.geometry.is_empty].copy()
        
        # Проверка CRS
        if data.crs is None:
            print(f"⚠️ {name}: CRS не установлен, устанавливаю EPSG:4326")
            data.set_crs('EPSG:4326', inplace=True)
        
        print(f"✓ {name}: найдено {len(data)} валидных объектов")
        
        # Дополнительная проверка: предупреждение если данных очень мало
        if len(data) < 5:
            print(f"⚠️ {name}: очень мало данных ({len(data)} объектов). Возможно, данные неполные.")
        
        return data
        
    except Exception as e:
        import traceback
        print(f"✗ {name}: ошибка при получении данных - {e}")
        print(f"  Детали: {traceback.format_exc()}")
        return gpd.GeoDataFrame()

def get_road_network(geometry):
    """
    Получение дорожной сети с проверкой качества.
    
    Returns:
        GeoDataFrame с дорожной сетью или пустой GeoDataFrame при ошибке
    """
    try:
        road_network = ox.graph_from_polygon(geometry, network_type='all')
        
        if road_network is None or len(road_network) == 0:
            print("⚠️ Дорожная сеть: граф пуст")
            return gpd.GeoDataFrame()
        
        roads = ox.graph_to_gdfs(road_network, edges=True, nodes=False)
        
        # Проверка качества
        if roads.empty:
            print("⚠️ Дорожная сеть: данные пусты")
            return gpd.GeoDataFrame()
        
        # Проверка геометрии
        if 'geometry' not in roads.columns:
            print("⚠️ Дорожная сеть: отсутствует колонка geometry")
            return gpd.GeoDataFrame()
        
        invalid_geom = roads.geometry.isna().sum()
        if invalid_geom > 0:
            print(f"⚠️ Дорожная сеть: найдено {invalid_geom} сегментов с невалидной геометрией")
            roads = roads[~roads.geometry.isna()].copy()
        
        if roads.crs is None:
            roads.set_crs('EPSG:4326', inplace=True)
        
        print(f"✓ Дорожная сеть: найдено {len(roads)} валидных сегментов")
        
        return roads
        
    except Exception as e:
        import traceback
        print(f"✗ Дорожная сеть: ошибка - {e}")
        print(f"  Детали: {traceback.format_exc()}")
        return gpd.GeoDataFrame()

def save_osm_data(osm_data, roi_params):
    """Сохранение собранных данных"""
    for name, data in osm_data.items():
        if not data.empty:
            filename = os.path.join(config.DATA_DIR, f"{name}_data.geojson")
            try:
                data.to_file(filename, driver="GeoJSON")
                print(f"✓ {name}: данные сохранены в {filename}")
            except Exception as e:
                print(f"✗ {name}: ошибка сохранения - {e}")
    
    with open(config.FILES['roi_params'], 'w', encoding='utf-8') as f:
        json.dump(roi_params, f, ensure_ascii=False, indent=2)

def load_osm_data():
    """Загрузка сохраненных данных"""
    osm_data = {}
    for name in config.OSM_TAGS.keys():
        filename = os.path.join(config.DATA_DIR, f"{name}_data.geojson")
        try:
            osm_data[name] = gpd.read_file(filename)
            print(f"✓ {name}: загружено {len(osm_data[name])} объектов")
        except Exception:
            print(f"✗ {name}: файл не найден, создаем пустой")
            osm_data[name] = gpd.GeoDataFrame()
            
    try:
        osm_data['roads'] = gpd.read_file(os.path.join(config.DATA_DIR, "roads_data.geojson"))
    except Exception:
        osm_data['roads'] = gpd.GeoDataFrame()
        
    return osm_data

