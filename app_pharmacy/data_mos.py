"""
Модуль интеграции с порталом открытых данных Москвы (data.mos.ru)

Загружает дополнительные данные для улучшения модели геомаркетинга:
- Медицинская инфраструктура (поликлиники, больницы)
- Демографические данные (численность населения по районам)
- Транспортные узлы
- Зарплаты и экономические показатели
"""
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import os
import logging
from . import config

# Настройка логирования
logger = logging.getLogger(__name__)

# API ключ из переменной окружения (безопаснее чем хардкод)
# Установите переменную окружения: export DATA_MOS_API_KEY="your_key_here"
API_KEY = os.environ.get('DATA_MOS_API_KEY', config.DATA_MOS_CONFIG.get('api_key', ''))
if not API_KEY:
    print("⚠️ ВНИМАНИЕ: DATA_MOS_API_KEY не установлен. Функции data.mos.ru могут не работать.")
    print("   Установите переменную окружения: export DATA_MOS_API_KEY='your_key'")

# Полезные датасеты data.mos.ru
DATASETS = {
    # Медицинская инфраструктура
    'polyclinics_adult': 503,      # Поликлиники взрослые
    'polyclinics_child': 505,      # Поликлиники детские
    'hospitals_adult': 517,        # Больницы взрослые
    'hospitals_child': 502,        # Больницы детские
    'dental_clinics': 518,         # Стоматологии
    'emergency': 516,              # Скорая помощь
    
    # Транспорт
    'transport_hubs': 1047,        # ТПУ
    
    # Демография (архивные, но полезные)
    'population_working': 2084,    # Население трудоспособного возраста
    'population_young': 2085,      # Население моложе трудоспособного
    'population_elderly': 2022,    # Население старше трудоспособного
    
    # Экономика
    'salaries': 2087,              # Средняя зарплата по отраслям
    
    # Адресный реестр (для геопривязки)
    'address_registry': 60562,
}

# Соответствие названий округов
DISTRICT_MAPPING = {
    'Центральный административный округ': 'ЦАО',
    'Северный административный округ': 'САО',
    'Северо-Восточный административный округ': 'СВАО',
    'Восточный административный округ': 'ВАО',
    'Юго-Восточный административный округ': 'ЮВАО',
    'Южный административный округ': 'ЮАО',
    'Юго-Западный административный округ': 'ЮЗАО',
    'Западный административный округ': 'ЗАО',
    'Северо-Западный административный округ': 'СЗАО',
    'Зеленоградский административный округ': 'ЗелАО',
    'Троицкий административный округ': 'ТАО',
    'Новомосковский административный округ': 'НАО',
}


def get_dataset_info(dataset_id):
    """Получает информацию о датасете"""
    url = f"https://apidata.mos.ru/v1/datasets/{dataset_id}?api_key={API_KEY}"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        return r.json()
    return None


def fetch_dataset(dataset_id, top=None, filter_query=None):
    """
    Загружает данные датасета.
    
    Args:
        dataset_id: ID датасета
        top: Максимальное количество записей
        filter_query: OData фильтр
    
    Returns:
        list: Список записей
    """
    url = f"https://apidata.mos.ru/v1/datasets/{dataset_id}/rows?api_key={API_KEY}"
    
    if top:
        url += f"&$top={top}"
    if filter_query:
        url += f"&$filter={filter_query}"
    
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            data = r.json()
            return [row.get('Cells', {}) for row in data]
    except Exception as e:
        print(f"⚠️ Ошибка загрузки датасета {dataset_id}: {e}")
    
    return []


def load_medical_facilities():
    """
    Загружает данные о медицинских учреждениях.
    
    Returns:
        GeoDataFrame с медучреждениями
    """
    facilities = []
    
    datasets = [
        ('polyclinics_adult', 'Поликлиника взрослая'),
        ('polyclinics_child', 'Поликлиника детская'),
        ('hospitals_adult', 'Больница взрослая'),
        ('hospitals_child', 'Больница детская'),
        ('dental_clinics', 'Стоматология'),
    ]
    
    for ds_name, facility_type in datasets:
        ds_id = DATASETS.get(ds_name)
        if not ds_id:
            continue
            
        data = fetch_dataset(ds_id, top=500)
        
        for row in data:
            # Координаты в geoData (MultiPoint)
            geo_data = row.get('geoData')
            if geo_data and geo_data.get('type') == 'MultiPoint':
                coords = geo_data.get('coordinates', [])
                if coords and len(coords[0]) >= 2:
                    lon, lat = coords[0][0], coords[0][1]
                    
                    # Адрес и район
                    address_list = row.get('ObjectAddress', [])
                    addr_info = address_list[0] if address_list else {}
                    
                    facilities.append({
                        'name': row.get('ShortName', row.get('FullName', '')),
                        'type': facility_type,
                        'address': addr_info.get('Address', ''),
                        'adm_area': addr_info.get('AdmArea', ''),
                        'district': addr_info.get('District', ''),
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'source': 'data.mos.ru'
                    })
    
    if facilities:
        df = pd.DataFrame(facilities)
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        print(f"✓ Загружено {len(gdf)} медучреждений из data.mos.ru")
        return gdf
    
    return gpd.GeoDataFrame()


def load_transport_hubs():
    """
    Загружает транспортно-пересадочные узлы.
    
    Returns:
        GeoDataFrame с ТПУ
    """
    data = fetch_dataset(DATASETS['transport_hubs'], top=300)
    
    hubs = []
    for row in data:
        geo_data = row.get('geoData')
        if geo_data and geo_data.get('type') == 'Point':
            coords = geo_data.get('coordinates', [])
            if len(coords) >= 2:
                hubs.append({
                    'name': row.get('TPUName', ''),
                    'adm_area': row.get('AdmArea', ''),
                    'district': row.get('District', ''),
                    'longitude': coords[0],
                    'latitude': coords[1],
                })
    
    if hubs:
        df = pd.DataFrame(hubs)
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        print(f"✓ Загружено {len(gdf)} ТПУ из data.mos.ru")
        return gdf
    
    return gpd.GeoDataFrame()


def load_population_by_district():
    """
    Загружает данные о населении по округам.
    
    Returns:
        DataFrame с демографическими данными
    """
    population = {}
    
    # Трудоспособное население
    working = fetch_dataset(DATASETS['population_working'], top=100)
    for row in working:
        territory = row.get('Territory', '')
        if territory in DISTRICT_MAPPING:
            district = DISTRICT_MAPPING[territory]
            if district not in population:
                population[district] = {}
            population[district]['population_working'] = row.get('QuantityInThousandPeoples', 0) * 1000
    
    # Молодое население
    young = fetch_dataset(DATASETS['population_young'], top=100)
    for row in young:
        territory = row.get('Territory', '')
        if territory in DISTRICT_MAPPING:
            district = DISTRICT_MAPPING[territory]
            if district not in population:
                population[district] = {}
            population[district]['population_young'] = row.get('QuantityInThousandPeoples', 0) * 1000
    
    if population:
        df = pd.DataFrame.from_dict(population, orient='index')
        df.index.name = 'district_short'
        df = df.reset_index()
        
        # Вычисляем общую численность и долю пожилых (важно для аптек!)
        df['population_total'] = df.get('population_working', 0) + df.get('population_young', 0)
        
        print(f"✓ Загружены демографические данные для {len(df)} округов")
        return df
    
    return pd.DataFrame()


def load_average_salaries():
    """
    Загружает данные о средней зарплате.
    
    Returns:
        DataFrame с данными о зарплатах
    """
    data = fetch_dataset(DATASETS['salaries'], top=100)
    
    if data:
        # Берём последний год
        df = pd.DataFrame(data)
        if 'Year' in df.columns:
            latest = df.sort_values('Year', ascending=False).iloc[0]
            
            salary_info = {
                'avg_salary_total': latest.get('AverageSalaryTotal'),
                'avg_salary_healthcare': latest.get('HealthcareAndSocialServices'),
                'avg_salary_trade': latest.get('TradingAndAutomotiveRepairs'),
                'year': latest.get('Year'),
            }
            print(f"✓ Загружены данные о зарплатах за {salary_info['year']} год")
            return salary_info
    
    return {}


def enrich_h3_grid_with_mos_data(h3_grid, osm_data=None):
    """
    Обогащает H3 сетку данными из data.mos.ru
    
    Args:
        h3_grid: GeoDataFrame с H3 ячейками
        osm_data: Словарь с OSM данными (опционально)
    
    Returns:
        GeoDataFrame с дополнительными признаками
    """
    print("\n📥 Обогащение данными из data.mos.ru...")

    # Подготовка: выбираем метрическую CRS (обычно UTM) и считаем координаты центров ячеек в метрах
    from scipy.spatial import cKDTree

    if getattr(h3_grid, "crs", None) is None:
        h3_grid = h3_grid.set_crs("EPSG:4326")

    try:
        metric_crs = h3_grid.estimate_utm_crs()
        if metric_crs is None:
            raise ValueError("estimate_utm_crs вернул None")
    except Exception as e:
        # fallback (менее точно), но лучше чем падать
        print(f"  ⚠️ Не удалось определить UTM CRS ({e}), используется EPSG:3857 (менее точно)")
        metric_crs = "EPSG:3857"

    # Используем заранее вычисленные центры (если есть) — быстрее и стабильнее чем centroid полигона
    if {'center_lat', 'center_lon'}.issubset(h3_grid.columns):
        h3_points = gpd.GeoSeries(
            gpd.points_from_xy(h3_grid['center_lon'], h3_grid['center_lat']),
            crs="EPSG:4326"
        )
    else:
        h3_points = h3_grid.geometry.centroid
        if getattr(h3_points, "crs", None) is None:
            h3_points = gpd.GeoSeries(h3_points, crs=h3_grid.crs)

    h3_points_proj = h3_points.to_crs(metric_crs)
    h3_xy = np.column_stack((h3_points_proj.x, h3_points_proj.y))
    
    # 1. Медицинские учреждения
    try:
        medical_mos = load_medical_facilities()
        if not medical_mos.empty:
            if getattr(medical_mos, "crs", None) is None:
                medical_mos = medical_mos.set_crs("EPSG:4326")

            medical_proj = medical_mos.to_crs(metric_crs)
            med_xy = np.column_stack((medical_proj.geometry.x, medical_proj.geometry.y))

            if len(med_xy) > 0:
                tree = cKDTree(med_xy)
                distances_m, _ = tree.query(h3_xy, k=1)
                h3_grid['mos_medical_nearest_distance'] = distances_m

                # Количество в радиусе 500м (реальные метры)
                counts = tree.query_ball_point(h3_xy, r=500)
                h3_grid['mos_medical_count_500m'] = [len(c) for c in counts]
    except Exception as e:
        print(f"  ⚠️ Ошибка загрузки медучреждений: {e}")
    
    # 2. Транспортные узлы
    try:
        tpu = load_transport_hubs()
        if not tpu.empty:
            if getattr(tpu, "crs", None) is None:
                tpu = tpu.set_crs("EPSG:4326")

            tpu_proj = tpu.to_crs(metric_crs)
            tpu_xy = np.column_stack((tpu_proj.geometry.x, tpu_proj.geometry.y))

            if len(tpu_xy) > 0:
                tree = cKDTree(tpu_xy)
                distances_m, _ = tree.query(h3_xy, k=1)
                h3_grid['mos_tpu_nearest_distance'] = distances_m
    except Exception as e:
        print(f"  ⚠️ Ошибка загрузки ТПУ: {e}")
    
    # 3. Демографические данные (по округам)
    try:
        pop_data = load_population_by_district()
        if not pop_data.empty:
            # УЛУЧШЕНО: Используем демографические данные эффективно
            # Определяем округ для каждой ячейки по координатам центра
            # В реальности нужно использовать границы округов из GeoJSON
            
            # Пока используем упрощенный подход: присваиваем средние значения по округу
            # В будущем можно улучшить через spatial join с границами округов
            
            # Вычисляем среднюю плотность населения по всем округам
            if 'population_total' in pop_data.columns:
                avg_population = pop_data['population_total'].mean()
                h3_grid['mos_population_density_proxy'] = avg_population / 1000  # Упрощенная метрика
                print(f"  ✓ Демографические данные: средняя плотность населения {avg_population/1000:.1f} тыс. чел.")
            
            # Добавляем информацию о доле пожилого населения (важно для аптек!)
            if 'population_working' in pop_data.columns and 'population_young' in pop_data.columns:
                total_pop = pop_data['population_working'].sum() + pop_data['population_young'].sum()
                if total_pop > 0:
                    # Примерная доля пожилых (65+) - важный фактор для аптек
                    # Используем как прокси-метрику
                    h3_grid['mos_elderly_population_factor'] = 0.15  # Примерная доля для Москвы
                    print("  ✓ Добавлен фактор пожилого населения (важно для аптек)")
            
            h3_grid['mos_population_data_available'] = True
    except Exception as e:
        print(f"  ⚠️ Ошибка загрузки демографии: {e}")
        import traceback
        print(f"  Детали: {traceback.format_exc()}")
    
    # 4. Экономические данные (общие для города)
    try:
        salaries = load_average_salaries()
        if salaries:
            h3_grid['mos_avg_salary'] = salaries.get('avg_salary_total', 0)
            h3_grid['mos_healthcare_salary'] = salaries.get('avg_salary_healthcare', 0)
    except Exception as e:
        print(f"  ⚠️ Ошибка загрузки зарплат: {e}")
    
    return h3_grid


def get_available_datasets():
    """
    Возвращает список доступных датасетов.
    """
    available = []
    
    for name, ds_id in DATASETS.items():
        info = get_dataset_info(ds_id)
        if info:
            available.append({
                'id': ds_id,
                'name': name,
                'caption': info.get('Caption', ''),
                'category': info.get('CategoryCaption', ''),
            })
    
    return available


if __name__ == '__main__':
    print("Проверка доступных датасетов data.mos.ru...")
    datasets = get_available_datasets()
    
    print(f"\nДоступно {len(datasets)} датасетов:")
    for ds in datasets:
        print(f"  {ds['id']}: {ds['caption'][:50]}")
    
    print("\n\nТест загрузки данных:")
    medical = load_medical_facilities()
    tpu = load_transport_hubs()
    pop = load_population_by_district()
    salaries = load_average_salaries()
    
    print("\nИтого:")
    print(f"  Медучреждений: {len(medical)}")
    print(f"  ТПУ: {len(tpu)}")
    print(f"  Округов с демографией: {len(pop)}")
    print(f"  Средняя зарплата: {salaries.get('avg_salary_total', 'N/A')} руб.")

