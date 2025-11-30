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
from . import parallel_processing

def main():
    print("=== Геомаркетинговый анализ для аптек ===")
    
    # 1. Check Dependencies
    if not utils.check_and_install_dependencies():
        print("Пожалуйста, установите недостающие зависимости.")
    
    # 2. Data Collection
    print("\n--- Этап 1: Сбор данных (параллельно) ---")
    
    roi_geometry, roi_params, gdf_district = data_loader.get_roi_geometry()
    
    # Параллельная загрузка данных OSM
    osm_data = parallel_processing.parallel_load_osm_data(roi_geometry)
    
    # Дорожная сеть загружается отдельно (более тяжелая операция)
    print("Загрузка дорожной сети...")
    osm_data['roads'] = data_loader.get_road_network(roi_geometry)
    
    data_loader.save_osm_data(osm_data, roi_params)
    
    # 3. Feature Engineering
    print("\n--- Этап 2: Генерация признаков (параллельно) ---")
    h3_grid = features.create_h3_grid(roi_params)
    
    if h3_grid.empty:
        print("Ошибка создания H3 сетки. Выход.")
        return
        
    print(f"Сетка H3 создана: {len(h3_grid)} ячеек")
    
    # Используем SciPy версию для расчета признаков расстояния (O(log N))
    print("Расчет признаков расстояния (SciPy cKDTree)...")
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('transport'), 'transport')
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('residential'), 'residential')
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('medical'), 'medical')
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('offices'), 'office')
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('retail'), 'retail')
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('pharmacies'), 'pharmacy')
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('parking'), 'parking')
    h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('pedestrian'), 'pedestrian')
    
    # Площадные признаки (последовательно, т.к. требуют геометрических операций)
    h3_grid = features.calculate_area_based_features(h3_grid, osm_data.get('residential'), 'residential')
    h3_grid = features.calculate_area_based_features(h3_grid, osm_data.get('offices'), 'office')
    h3_grid = features.calculate_road_features(h3_grid, osm_data.get('roads'))
    
    # Кастомные признаки
    h3_grid = features.calculate_custom_features(h3_grid)
    
    # Целевая переменная
    h3_grid = features.add_target_variable(h3_grid, osm_data.get('pharmacies'))
    
    print(f"Признаки рассчитаны. Целевая переменная (аптеки): {h3_grid['has_pharmacy'].sum()}")
    
    # Save intermediate
    h3_grid.to_file(config.FILES['h3_grid_features'], driver='GeoJSON')
    
    feature_cols = [c for c in h3_grid.columns if c not in ['h3_cell', 'geometry', 'center_lat', 'center_lon', 'has_pharmacy']]

    # Visualization: Correlation Matrix
    # Prepare full dataframe for correlation (handle NaNs like in training)
    X_all_corr, _ = modeling.prepare_data(h3_grid, feature_cols)
    visualization.plot_correlation_matrix(X_all_corr, feature_cols)

    # 4. Modeling
    print("\n--- Этап 3: Моделирование ---")
    
    # Only proceed if we have positive samples
    if h3_grid['has_pharmacy'].sum() > 5:
        X, y = modeling.prepare_data(h3_grid, feature_cols)
        
        # Train Models
        best_model, results, X_test, y_test = modeling.train_models(X, y)
        modeling.save_model(best_model, config.FILES['model'])
        
        # Predict probabilities for the whole grid
        X_all, _ = modeling.prepare_data(h3_grid, feature_cols)
        h3_grid['prediction_score'] = best_model.predict_proba(X_all)[:, 1]
        
        # Feature Importance
        visualization.plot_feature_importance(best_model, feature_cols)
        
        # Clustering Analysis
        print("\n--- Кластерный анализ ---")
        best_k = modeling.analyze_clusters_optimal_k(X_all, max_k=10)
        cluster_labels, _ = modeling.perform_clustering(X_all, n_clusters=best_k)
        h3_grid['cluster'] = cluster_labels
        print(f"Кластеризация завершена с k={best_k}")
        
    else:
        print("Недостаточно данных для моделирования. Пропуск ML этапа.")
        h3_grid['prediction_score'] = 0
        h3_grid['cluster'] = 0
        
    # 5. Analysis & Recommendations
    print("\n--- Этап 4: Анализ и рекомендации ---")
    h3_grid = analysis.calculate_potential(h3_grid)
    
    # Filter out too close competitors
    top_cells, recommendations = analysis.get_recommendations(h3_grid, min_distance_to_competitor=300)
    
    print("Топ-5 рекомендованных локаций (без прямой конкуренции):")
    for rec in recommendations[:5]:
        print(f"Ранг {rec['rank']}: {rec['h3_cell']} (Score: {rec['potential']:.3f}, Дист. до ближ.: {rec['nearest_competitor_m']:.1f}м)")
        
    # Save final results
    h3_grid.to_file(config.FILES['h3_grid_final'], driver='GeoJSON')
    pd.DataFrame(recommendations).to_csv(config.FILES['top_10_summary'], index=False)
    
    # 6. Visualization
    print("\n--- Этап 5: Визуализация ---")
    # Pass competitors explicitly to map
    competitors = osm_data.get('pharmacies')
    visualization.create_potential_map(h3_grid, roi_params, competitors=competitors, recommendations=recommendations)
    
    print("\n=== Анализ завершен ===")

if __name__ == "__main__":
    main()
