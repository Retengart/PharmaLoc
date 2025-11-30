import osmnx as ox
import geopandas as gpd
import pandas as pd
import json
import os
from . import config

def setup_osmnx():
    ox.settings.use_cache = True
    ox.settings.log_console = True

def get_roi_geometry(place_name=config.ROI_PLACE_NAME):
    """Получает геометрию области интереса (ROI)"""
    try:
        gdf_district = ox.geocode_to_gdf(place_name)
        roi_geometry = gdf_district.geometry.iloc[0]
        bounds = gdf_district.total_bounds
        # north, south, east, west
        roi_params = {
            'north': bounds[3], 'south': bounds[1],
            'east': bounds[2], 'west': bounds[0],
            'center_lat': (bounds[3] + bounds[1]) / 2,
            'center_lon': (bounds[2] + bounds[0]),
            'place_name': place_name
        }
        return roi_geometry, roi_params, gdf_district
    except Exception as e:
        print(f"Ошибка при получении границ района: {e}")
        # Fallback
        return None, config.BACKUP_COORDS, None

def safe_get_osm_data(tags, geometry, name):
    """Безопасное получение данных OSM"""
    try:
        data = ox.features_from_polygon(geometry, tags=tags)
        print(f"✓ {name}: найдено {len(data)} объектов")
        return data
    except Exception as e:
        print(f"✗ {name}: ошибка при получении данных - {e}")
        return gpd.GeoDataFrame()

def get_road_network(geometry):
    """Получение дорожной сети"""
    try:
        road_network = ox.graph_from_polygon(geometry, network_type='all')
        roads = ox.graph_to_gdfs(road_network, edges=True, nodes=False)
        print(f"✓ Дорожная сеть: найдено {len(roads)} сегментов")
        return roads
    except Exception as e:
        print(f"✗ Дорожная сеть: ошибка - {e}")
        return gpd.GeoDataFrame()

def save_osm_data(osm_data, roi_params):
    """Сохранение собранных данных"""
    for name, data in osm_data.items():
        if not data.empty:
            filename = os.path.join(config.DATA_DIR, f"{name}_data.geojson")
            try:
                # Convert columns with list values to string to allow saving to GeoJSON
                # Some OSM tags return lists, which GeoJSON driver might struggle with if not handled, 
                # but geopandas usually handles it or we clean it. 
                # For simplicity, we just save.
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
            
    # Load roads separately
    try:
        osm_data['roads'] = gpd.read_file(os.path.join(config.DATA_DIR, "roads_data.geojson"))
    except:
        osm_data['roads'] = gpd.GeoDataFrame()
        
    return osm_data

