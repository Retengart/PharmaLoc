import pandas as pd
import numpy as np
from . import config

def calculate_potential(h3_grid, weights=config.POTENTIAL_WEIGHTS):
    """Расчет интегрального потенциала"""
    
    # Helper to normalize series to 0-1
    def normalize(series):
        if series.max() == series.min(): return 0
        return (series - series.min()) / (series.max() - series.min())

    # Initialize weights if not present
    w_med = weights.get('medical_synergy', 0)
    w_trans = weights.get('transport_accessibility', 0)
    w_res = weights.get('residential_density', 0)
    w_comm = weights.get('commercial_activity', 0)
    w_off = weights.get('office_density', 0)
    w_park = weights.get('parking_availability', 0)
    w_ped = weights.get('pedestrian_accessibility', 0)
    w_comp = weights.get('competition_penalty', 0)

    # 1. Medical Synergy
    if 'medical_synergy' in h3_grid.columns:
        val_med = h3_grid['medical_synergy']
    else:
        val_med = pd.Series(0, index=h3_grid.index)
        
    # 2. Transport (density)
    if 'transport_density_500m' in h3_grid.columns:
        val_trans = normalize(h3_grid['transport_density_500m'])
    else:
        val_trans = pd.Series(0, index=h3_grid.index)
        
    # 3. Residential
    if 'residential_coverage' in h3_grid.columns:
        val_res = h3_grid['residential_coverage']
    elif 'residential_density_500m' in h3_grid.columns:
        val_res = normalize(h3_grid['residential_density_500m'])
    else:
        val_res = pd.Series(0, index=h3_grid.index)
        
    # 4. Commercial
    if 'retail_density_500m' in h3_grid.columns:
        val_comm = normalize(h3_grid['retail_density_500m'])
    else:
        val_comm = pd.Series(0, index=h3_grid.index)
        
    # 5. Office
    if 'office_density_500m' in h3_grid.columns:
        val_off = normalize(h3_grid['office_density_500m'])
    else:
        val_off = pd.Series(0, index=h3_grid.index)

    # 6. Parking
    if 'parking_density_500m' in h3_grid.columns:
        val_park = normalize(h3_grid['parking_density_500m'])
    else:
        val_park = pd.Series(0, index=h3_grid.index)

    # 7. Pedestrian
    if 'pedestrian_density_500m' in h3_grid.columns:
        val_ped = normalize(h3_grid['pedestrian_density_500m'])
    else:
        val_ped = pd.Series(0, index=h3_grid.index)
        
    # 8. Competition
    if 'pharmacy_density_500m' in h3_grid.columns:
        val_comp = normalize(h3_grid['pharmacy_density_500m'])
    else:
        val_comp = pd.Series(0, index=h3_grid.index)
        
    # Weighted Sum
    score = (
        w_med * val_med +
        w_trans * val_trans +
        w_res * val_res +
        w_comm * val_comm +
        w_off * val_off +
        w_park * val_park +
        w_ped * val_ped +
        w_comp * val_comp
    )
    
    # Combine with ML prediction if available
    if 'prediction_score' in h3_grid.columns:
        h3_grid['potential_score'] = 0.6 * score + 0.4 * h3_grid['prediction_score']
    else:
        h3_grid['potential_score'] = score
        
    return h3_grid

def get_recommendations(h3_grid, top_n=10, min_distance_to_competitor=300):
    """
    Формирование рекомендаций.
    Исключает локации, где расстояние до ближайшей аптеки меньше min_distance_to_competitor (в метрах).
    """
    # 1. Filter cells that already have pharmacies (target=1)
    candidates = h3_grid[h3_grid['has_pharmacy'] == 0].copy()
    
    # 2. Filter by distance to nearest competitor if column exists
    if 'pharmacy_nearest_distance' in candidates.columns:
        candidates = candidates[candidates['pharmacy_nearest_distance'] >= min_distance_to_competitor]
    
    if candidates.empty:
        print("⚠️ Внимание: Не найдено кандидатов, удовлетворяющих условиям (дистанция). Возвращаем лучших из всех свободных.")
        candidates = h3_grid[h3_grid['has_pharmacy'] == 0].copy()
        if candidates.empty: # Should not happen unless grid is empty or all have pharmacies
             candidates = h3_grid.copy()
        
    top_candidates = candidates.nlargest(top_n, 'potential_score')
    
    recommendations = []
    rank = 1
    for idx, row in top_candidates.iterrows():
        rec = {
            'rank': rank,
            'h3_cell': row['h3_cell'],
            'lat': row['center_lat'],
            'lon': row['center_lon'],
            'potential': row['potential_score'],
            'cluster': row.get('cluster', -1),
            'nearest_competitor_m': row.get('pharmacy_nearest_distance', -1)
        }
        recommendations.append(rec)
        rank += 1
        
    return top_candidates, recommendations
