import folium
from folium.plugins import HeatMap, MarkerCluster
from folium import LayerControl, FeatureGroup
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from . import config

def create_base_map(roi_params):
    """Базовая карта"""
    m = folium.Map(
        location=[roi_params['center_lat'], roi_params['center_lon']],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    return m

def create_potential_map(h3_grid, roi_params, competitors=None, recommendations=None, filename='potential_map.html'):
    """
    Создает подробную карту потенциала с несколькими слоями и легендой.
    """
    m = create_base_map(roi_params)
    
    # 1. Слой потенциала (Heatmap)
    heat_data = h3_grid[['center_lat', 'center_lon', 'potential_score']].dropna().values.tolist()
    hm_layer = FeatureGroup(name='Тепловая карта потенциала', show=True)
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(hm_layer)
    hm_layer.add_to(m)
    
    # 2. Слой сетки (Choropleth)
    # Color scale function
    def get_color(score):
        if score > 0.8: return '#d73027' # Red (High)
        elif score > 0.6: return '#fc8d59'
        elif score > 0.4: return '#fee08b'
        elif score > 0.2: return '#d9ef8b'
        else: return '#91cf60' # Green (Low) - Wait, usually High Potential is Green? 
        # Let's stick to logic: High Potential -> Red (Hot) or Green (Good)?
        # In business, Green usually means "Go". Let's use Green for High.
        
    def get_color_green_high(score):
        if score > 0.8: return '#1a9850' # Dark Green
        elif score > 0.6: return '#91cf60'
        elif score > 0.4: return '#fee08b'
        elif score > 0.2: return '#fc8d59'
        else: return '#d73027' # Red (Bad)

    grid_layer = FeatureGroup(name='Сетка H3 (Детализация)', show=False)
    
    # Prepare Tooltip features
    # Convert potential to readable
    
    folium.GeoJson(
        h3_grid.to_json(),
        style_function=lambda feature: {
            'fillColor': get_color_green_high(feature['properties'].get('potential_score', 0)),
            'color': 'gray',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['h3_cell', 'potential_score', 'has_pharmacy', 'cluster'],
            aliases=['H3 ID:', 'Потенциал:', 'Есть аптека:', 'Кластер:'],
            localize=True
        )
    ).add_to(grid_layer)
    grid_layer.add_to(m)
    
    # 3. Слой конкурентов (если есть)
    if competitors is not None and not competitors.empty:
        comp_layer = FeatureGroup(name='Конкуренты (Аптеки)', show=True)
        for idx, row in competitors.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x] if row.geometry.geom_type == 'Point' else [row.geometry.centroid.y, row.geometry.centroid.x],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup='Конкурент'
            ).add_to(comp_layer)
        comp_layer.add_to(m)
        
    # 4. Слой рекомендаций
    rec_coords = []
    if recommendations:
        rec_layer = FeatureGroup(name='Рекомендованные точки', show=True)
        for rec in recommendations:
            html = f"""
            <div style="font-family: sans-serif; width: 200px;">
                <h4>Ранг #{rec['rank']}</h4>
                <p><b>H3:</b> {rec['h3_cell']}</p>
                <p><b>Потенциал:</b> {rec['potential']:.3f}</p>
                <p><b>Ближайший конкурент:</b> {rec['nearest_competitor_m']:.0f} м</p>
            </div>
            """
            folium.Marker(
                [rec['lat'], rec['lon']],
                popup=folium.Popup(html, max_width=300),
                icon=folium.Icon(color='green', icon='star', prefix='fa'),
                tooltip=f"Топ-{rec['rank']}"
            ).add_to(rec_layer)
            rec_coords.append([rec['lat'], rec['lon']])
            
        rec_layer.add_to(m)
        
        # Автоматическое масштабирование карты под рекомендации
        if rec_coords:
             m.fit_bounds(rec_coords)
        
    # Add Layer Control
    LayerControl().add_to(m)
    
    # Add Custom Legend
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: auto; min-width: 150px; max-width: 250px; height: auto; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.9;
     padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2);">
     <b>Легенда</b><br>
     <i class="fa fa-star" style="color:green"></i> Рекомендация<br>
     <i class="fa fa-circle" style="color:red"></i> Конкурент<br>
     <br>
     <b>Потенциал:</b><br>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#1a9850; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Высокий (>0.8)
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#91cf60; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Выше среднего
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#fee08b; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Средний
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#fc8d59; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Ниже среднего
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#d73027; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Низкий
     </div>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    output_path = os.path.join(config.DATA_DIR, filename)
    m.save(output_path)
    print(f"Карта сохранена в {output_path}")

def plot_feature_importance(model, feature_names, filename='feature_importance.png'):
    """График важности признаков"""
    if hasattr(model, 'named_steps'):
        clf = model.named_steps['classifier']
    else:
        clf = model
        
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 20
        top_n = 20
        indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title("Важность признаков (Топ-20)", fontsize=16)
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.ylabel('Важность (Gini importance)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        output_path = os.path.join(config.DATA_DIR, filename)
        plt.savefig(output_path)
        print(f"График важности сохранен в {output_path}")
        plt.close()

def plot_correlation_matrix(df, features, filename='correlation_matrix.png'):
    """Построение матрицы корреляции признаков"""
    plt.figure(figsize=(14, 12))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    
    plt.title('Матрица корреляции признаков', fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path)
    print(f"Матрица корреляции сохранена в {output_path}")
    plt.close()
