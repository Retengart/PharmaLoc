import folium
from folium.plugins import HeatMap
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from . import config

def create_base_map(roi_params):
    """Базовая карта"""
    m = folium.Map(
        location=[roi_params['center_lat'], roi_params['center_lon']],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    return m

def add_h3_layer(m, h3_grid, column=None, color_map=None):
    """Добавляет слой H3 сетки"""
    
    def style_function(feature):
        val = feature['properties'].get(column, 0)
        color = 'blue'
        opacity = 0.1
        
        if column == 'potential_score':
            # Simple color scale
            if val > 0.8: color = '#d73027'
            elif val > 0.6: color = '#fc8d59'
            elif val > 0.4: color = '#fee08b'
            elif val > 0.2: color = '#d9ef8b'
            else: color = '#91cf60'
            opacity = 0.6
        elif column == 'has_pharmacy' and val == 1:
            color = 'red'
            opacity = 0.6
            
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': opacity
        }
        
    folium.GeoJson(
        h3_grid.to_json(),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['h3_cell', 'potential_score', 'has_pharmacy'])
    ).add_to(m)
    return m

def create_potential_map(h3_grid, roi_params, filename='potential_map.html'):
    """Карта потенциала"""
    m = create_base_map(roi_params)
    
    # Heatmap data
    heat_data = h3_grid[['center_lat', 'center_lon', 'potential_score']].dropna().values.tolist()
    HeatMap(heat_data, radius=15).add_to(m)
    
    # Top locations markers
    top_locs = h3_grid.nlargest(10, 'potential_score')
    for i, row in top_locs.iterrows():
        folium.Marker(
            [row['center_lat'], row['center_lon']],
            popup=f"Rank {i+1}: {row['potential_score']:.2f}",
            icon=folium.Icon(color='green', icon='star')
        ).add_to(m)
        
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
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        output_path = os.path.join(config.DATA_DIR, filename)
        plt.savefig(output_path)
        print(f"График важности сохранен в {output_path}")

