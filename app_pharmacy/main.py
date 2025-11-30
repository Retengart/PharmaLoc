import sys
import os
import pandas as pd

# Add current directory to path to find modules if running as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from . import utils
from . import config
from . import data_loader
from . import features
from . import modeling
from . import analysis
from . import visualization

def main():
    print("=== Geomarketing Pharmacy Analysis ===")
    
    # 1. Check Dependencies
    if not utils.check_and_install_dependencies():
        print("Please install missing dependencies and restart.")
        # We don't exit here to allow the user to see what's missing, 
        # but in a real app we might want to stop.
        # Continuing might crash.
    
    # 2. Data Collection
    print("\n--- Stage 1: Data Collection ---")
    # Try to load existing data first?
    # For this assignment, let's assume we want to refresh or download if missing.
    # But since download takes time, let's check if main files exist.
    
    roi_geometry, roi_params, gdf_district = data_loader.get_roi_geometry()
    
    # We will download data. In a real scenario, we might check if files exist.
    osm_data = {}
    for key, tags in config.OSM_TAGS.items():
        osm_data[key] = data_loader.safe_get_osm_data(tags, roi_geometry, key)
    
    osm_data['roads'] = data_loader.get_road_network(roi_geometry)
    
    data_loader.save_osm_data(osm_data, roi_params)
    
    # 3. Feature Engineering
    print("\n--- Stage 2: Feature Engineering ---")
    h3_grid = features.create_h3_grid(roi_params)
    
    if h3_grid.empty:
        print("Failed to create H3 grid. Exiting.")
        return
        
    print(f"H3 Grid created: {len(h3_grid)} cells")
    
    # Calculate features
    h3_grid = features.calculate_distance_based_features(h3_grid, osm_data.get('transport'), 'transport')
    h3_grid = features.calculate_distance_based_features(h3_grid, osm_data.get('residential'), 'residential')
    h3_grid = features.calculate_area_based_features(h3_grid, osm_data.get('residential'), 'residential')
    h3_grid = features.calculate_distance_based_features(h3_grid, osm_data.get('medical'), 'medical')
    h3_grid = features.calculate_distance_based_features(h3_grid, osm_data.get('offices'), 'office')
    h3_grid = features.calculate_area_based_features(h3_grid, osm_data.get('offices'), 'office')
    h3_grid = features.calculate_distance_based_features(h3_grid, osm_data.get('retail'), 'retail')
    h3_grid = features.calculate_distance_based_features(h3_grid, osm_data.get('pharmacies'), 'pharmacy') # Competition
    
    h3_grid = features.calculate_road_features(h3_grid, osm_data.get('roads'))
    h3_grid = features.calculate_custom_features(h3_grid)
    
    h3_grid = features.add_target_variable(h3_grid, osm_data.get('pharmacies'))
    
    print(f"Features calculated. Target variable (pharmacies): {h3_grid['has_pharmacy'].sum()}")
    
    # Save intermediate
    h3_grid.to_file(config.FILES['h3_grid_features'], driver='GeoJSON')
    
    # 4. Modeling
    print("\n--- Stage 3: Modeling ---")
    feature_cols = [c for c in h3_grid.columns if c not in ['h3_cell', 'geometry', 'center_lat', 'center_lon', 'has_pharmacy']]
    
    # Only proceed if we have positive samples
    if h3_grid['has_pharmacy'].sum() > 5:
        X, y = modeling.prepare_data(h3_grid, feature_cols)
        best_model, results, X_test, y_test = modeling.train_models(X, y)
        
        modeling.save_model(best_model, config.FILES['model'])
        
        # Predict probabilities for the whole grid
        # Need to preprocess whole grid same as training
        X_all, _ = modeling.prepare_data(h3_grid, feature_cols)
        # Note: in real pipeline, we should use the scaler fitted on training data. 
        # The modeling.train_models uses a Pipeline which includes scaler, so we can just predict.
        h3_grid['prediction_score'] = best_model.predict_proba(X_all)[:, 1]
        
        # Feature Importance
        visualization.plot_feature_importance(best_model, feature_cols)
        
        # Clustering
        cluster_labels, _ = modeling.perform_clustering(X_all)
        h3_grid['cluster'] = cluster_labels
    else:
        print("Not enough data for modeling. Skipping ML stage.")
        h3_grid['prediction_score'] = 0
        h3_grid['cluster'] = 0
        
    # 5. Analysis & Recommendations
    print("\n--- Stage 4: Analysis & Recommendations ---")
    h3_grid = analysis.calculate_potential(h3_grid)
    top_cells, recommendations = analysis.get_recommendations(h3_grid)
    
    print("Top 5 Recommended Locations:")
    for rec in recommendations[:5]:
        print(f"Rank {rec['rank']}: {rec['h3_cell']} (Score: {rec['potential']:.3f})")
        
    # Save final results
    h3_grid.to_file(config.FILES['h3_grid_final'], driver='GeoJSON')
    pd.DataFrame(recommendations).to_csv(config.FILES['top_10_summary'], index=False)
    
    # 6. Visualization
    print("\n--- Stage 5: Visualization ---")
    visualization.create_potential_map(h3_grid, roi_params)
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()

