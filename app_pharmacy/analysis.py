import pandas as pd
import numpy as np
from . import config

def calculate_potential(h3_grid, weights=config.POTENTIAL_WEIGHTS):
    """Расчет интегрального потенциала"""
    
    # Helper to normalize series to 0-1
    def normalize(series):
        if series.max() == series.min(): return 0
        return (series - series.min()) / (series.max() - series.min())

    # 1. Medical Synergy
    if 'medical_synergy' in h3_grid.columns:
        med = h3_grid['medical_synergy']
    else:
        med = pd.Series(0, index=h3_grid.index)
        
    # 2. Transport (density)
    if 'transport_density_500m' in h3_grid.columns:
        trans = normalize(h3_grid['transport_density_500m'])
    else:
        trans = pd.Series(0, index=h3_grid.index)
        
    # 3. Residential
    if 'residential_coverage' in h3_grid.columns:
        res = h3_grid['residential_coverage']
    elif 'residential_density_500m' in h3_grid.columns:
        res = normalize(h3_grid['residential_density_500m'])
    else:
        res = pd.Series(0, index=h3_grid.index)
        
    # 4. Commercial
    if 'retail_density_500m' in h3_grid.columns:
        comm = normalize(h3_grid['retail_density_500m'])
    else:
        comm = pd.Series(0, index=h3_grid.index)
        
    # 5. Office
    if 'office_density_500m' in h3_grid.columns:
        off = normalize(h3_grid['office_density_500m'])
    else:
        off = pd.Series(0, index=h3_grid.index)
        
    # 6. Competition
    if 'pharmacy_density_500m' in h3_grid.columns:
        comp = normalize(h3_grid['pharmacy_density_500m'])
    else:
        comp = pd.Series(0, index=h3_grid.index)
        
    # Weighted Sum
    score = (
        weights['medical_synergy'] * med +
        weights['transport_accessibility'] * trans +
        weights['residential_density'] * res +
        weights['commercial_activity'] * comm +
        weights['office_density'] * off +
        weights['competition_penalty'] * comp
    )
    
    # Combine with ML prediction if available
    if 'prediction_score' in h3_grid.columns:
        h3_grid['potential_score'] = 0.5 * score + 0.5 * h3_grid['prediction_score']
    else:
        h3_grid['potential_score'] = score
        
    return h3_grid

def get_recommendations(h3_grid, top_n=10):
    """Формирование рекомендаций"""
    # Filter out cells that already have pharmacies
    candidates = h3_grid[h3_grid['has_pharmacy'] == 0].copy()
    
    if candidates.empty:
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
            'cluster': row.get('cluster', -1)
        }
        recommendations.append(rec)
        rank += 1
        
    return top_candidates, recommendations

