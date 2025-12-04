import sys
import os
import argparse
import pandas as pd
import geopandas as gpd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from . import utils
from . import config
from . import data_loader
from . import features
from . import modeling
from . import analysis
from . import visualization
from . import parallel_processing
from . import data_mos


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description='Геомаркетинговый анализ для размещения аптек',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python3 -m app_pharmacy.main                    # Интерактивный режим
  python3 -m app_pharmacy.main --train            # Обучить модель с нуля
  python3 -m app_pharmacy.main --load             # Загрузить существующую модель
  python3 -m app_pharmacy.main --skip-data        # Пропустить загрузку данных (использовать кэш)
  python3 -m app_pharmacy.main --train --no-leakage  # Обучить без "утекающих" признаков
  python3 -m app_pharmacy.main --validate         # Валидация на другом округе (ЗАО)
  python3 -m app_pharmacy.main --enrich           # Обогатить данными из data.mos.ru
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--train', action='store_true', 
                           help='Обучить модель с нуля')
    mode_group.add_argument('--load', action='store_true',
                           help='Загрузить существующую модель')
    mode_group.add_argument('--validate', action='store_true',
                           help='Валидация модели на другом округе (out-of-domain)')
    
    parser.add_argument('--skip-data', action='store_true',
                       help='Пропустить загрузку данных OSM (использовать кэш)')
    
    parser.add_argument('--no-leakage', action='store_true',
                       help='Исключить признаки с утечкой данных (pharmacy_*) для честной оценки')
    
    parser.add_argument('--enrich', action='store_true',
                       help='Обогатить данные из портала data.mos.ru (медучреждения, ТПУ, демография)')
    
    return parser.parse_args()


def check_existing_model():
    """Проверяет наличие обученной модели"""
    model_path = config.FILES['model']
    if os.path.exists(model_path):
        import datetime
        mtime = os.path.getmtime(model_path)
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return True, model_path, mtime_str, size_mb
    return False, None, None, None


def load_cached_data():
    """Загружает кэшированные данные"""
    try:
        h3_grid = gpd.read_file(config.FILES['h3_grid_features'])
        osm_data = data_loader.load_osm_data()
        
        with open(config.FILES['roi_params'], 'r') as f:
            import json
            roi_params = json.load(f)
        
        print("✓ Данные загружены из кэша")
        return h3_grid, osm_data, roi_params
    except Exception as e:
        print(f"⚠️ Не удалось загрузить кэш: {e}")
        return None, None, None


def run_validation_mode():
    """Режим валидации на другом округе (out-of-domain)"""
    print("="*80)
    print("  ВАЛИДАЦИЯ МОДЕЛИ НА НЕЗАВИСИМОМ РЕГИОНЕ")
    print(f"  Обучение: {config.ROI_PLACE_NAME}")
    print(f"  Валидация: {config.VALIDATION_PLACE_NAME}")
    print("="*80)
    
    # Загружаем модель
    model_exists, model_path, _, _ = check_existing_model()
    if not model_exists:
        print("❌ Модель не найдена. Сначала обучите модель:")
        print("   python3 -m app_pharmacy.main --train --no-leakage")
        return
    
    best_model, saved_features, exclude_leakage = modeling.load_model(model_path)
    print(f"✓ Модель загружена из {model_path}")
    if exclude_leakage:
        print("   (обучена без признаков с утечкой)")
    
    # Загружаем данные валидационного региона
    print(f"\n📥 Загрузка данных для {config.VALIDATION_PLACE_NAME}...")
    
    # Сохраняем оригинальные параметры
    original_place = config.ROI_PLACE_NAME
    original_coords = config.BACKUP_COORDS
    
    # Переключаемся на валидационный регион
    config.ROI_PLACE_NAME = config.VALIDATION_PLACE_NAME
    config.BACKUP_COORDS = config.VALIDATION_COORDS
    
    try:
        # Получаем данные для валидационного региона
        val_roi_params = data_loader.get_region_bounds()
        val_osm_data = data_loader.load_osm_data(use_cache=False)  # Без кэша для нового региона
        
        # Создаём H3 сетку
        from .features import create_h3_grid, generate_all_features
        val_h3_grid = create_h3_grid(val_roi_params)
        print(f"✓ Создана H3 сетка: {len(val_h3_grid)} ячеек")
        
        # Генерируем признаки
        val_h3_grid = generate_all_features(val_h3_grid, val_osm_data)
        print("✓ Сгенерированы признаки")
        print(f"   Ячеек с аптеками: {val_h3_grid['has_pharmacy'].sum()}")
        
        # Подготавливаем данные
        feature_cols = [c for c in val_h3_grid.columns if c not in [
            'h3_index', 'geometry', 'center_lat', 'center_lon', 
            'has_pharmacy', 'prediction_score', 'potential_score', 'cluster'
        ]]
        
        X_val, y_val = modeling.prepare_data(val_h3_grid, feature_cols, exclude_leakage=exclude_leakage)
        
        # Добавляем кластерные признаки (используем те же k что и при обучении)
        X_val_ext, _, _, _ = modeling.add_cluster_features(X_val, n_clusters=2)
        
        # Проверяем совпадение признаков
        if saved_features:
            missing = set(saved_features) - set(X_val_ext.columns)
            extra = set(X_val_ext.columns) - set(saved_features)
            if missing:
                print(f"⚠️ Отсутствуют признаки: {missing}")
                # Добавляем недостающие с нулями
                for col in missing:
                    X_val_ext[col] = 0
            if extra:
                # Удаляем лишние
                X_val_ext = X_val_ext[[c for c in X_val_ext.columns if c in saved_features]]
            
            # Приводим к порядку как при обучении
            X_val_ext = X_val_ext[saved_features]
        
        # Валидация
        metrics, y_pred, y_proba = modeling.validate_on_region(
            best_model, X_val_ext, y_val, 
            region_name=config.VALIDATION_PLACE_NAME
        )
        
        # Сохраняем результаты валидации
        import json
        val_results = {
            'train_region': original_place,
            'validation_region': config.VALIDATION_PLACE_NAME,
            'metrics': metrics,
            'validation_samples': len(y_val),
            'positive_samples': int(y_val.sum()),
        }
        
        val_file = os.path.join(config.DATA_DIR, 'validation_results.json')
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Результаты валидации сохранены в {val_file}")
        
    finally:
        # Восстанавливаем оригинальные параметры
        config.ROI_PLACE_NAME = original_place
        config.BACKUP_COORDS = original_coords


def main():
    args = parse_args()
    
    # Режим валидации на другом округе
    if args.validate:
        run_validation_mode()
        return
    
    print("="*80)
    print("  ГЕОМАРКЕТИНГОВЫЙ АНАЛИЗ ДЛЯ РАЗМЕЩЕНИЯ АПТЕК")
    print("  Определение оптимальных локаций с использованием ML")
    print("="*80)
    
    if not utils.check_and_install_dependencies():
        print("Пожалуйста, установите недостающие зависимости.")
    
    model_exists, model_path, model_time, model_size = check_existing_model()
    
    if not args.train and not args.load:
        if model_exists:
            print("\n📦 Найдена обученная модель:")
            print(f"   Путь: {model_path}")
            print(f"   Дата: {model_time}")
            print(f"   Размер: {model_size:.2f} MB")
            print("\nВыберите режим:")
            print("  [1] Загрузить существующую модель (быстро)")
            print("  [2] Обучить модель с нуля (долго, ~5-10 мин)")
            
            choice = input("\nВаш выбор [1/2]: ").strip()
            train_new_model = (choice == '2')
        else:
            print("\n⚠️ Обученная модель не найдена. Будет выполнено обучение с нуля.")
            train_new_model = True
    else:
        train_new_model = args.train
    
    use_cached_data = args.skip_data
    
    print("\n" + "="*80)
    print("📥 ЭТАП 1: СБОР ГЕОПРОСТРАНСТВЕННЫХ ДАННЫХ")
    print("="*80)
    
    if use_cached_data:
        h3_grid, osm_data, roi_params = load_cached_data()
        if h3_grid is None:
            print("Кэш не найден, загружаем данные...")
            use_cached_data = False
    
    if not use_cached_data:
        roi_geometry, roi_params, gdf_district = data_loader.get_roi_geometry()
        print(f"✓ ROI: {roi_params.get('place_name', 'Unknown')}")
        
        osm_data = parallel_processing.parallel_load_osm_data(roi_geometry)
        
        print("Загрузка дорожной сети...")
        osm_data['roads'] = data_loader.get_road_network(roi_geometry)
        
        data_loader.save_osm_data(osm_data, roi_params)
        h3_grid = None
    else:
        roi_geometry = None
    
    print("\n" + "="*80)
    print("🔧 ЭТАП 2: ГЕНЕРАЦИЯ ПРИЗНАКОВ (Feature Engineering)")
    print("="*80)
    
    need_features = h3_grid is None or 'has_pharmacy' not in h3_grid.columns
    
    if need_features:
        if h3_grid is None:
            h3_grid = features.create_h3_grid(roi_params)
        
        if h3_grid.empty:
            print("❌ Ошибка создания H3 сетки. Выход.")
            return
            
        print(f"✓ Сетка H3 создана: {len(h3_grid)} ячеек (разрешение {config.H3_RESOLUTION})")
        
        print("\n📊 Расчет признаков расстояния (SciPy cKDTree)...")
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('transport_subway'), 'transport_subway')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('transport_ground'), 'transport_ground')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('residential'), 'residential')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('medical'), 'medical')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('offices'), 'office')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('retail'), 'retail')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('pharmacies'), 'pharmacy')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('parking'), 'parking')
        h3_grid = parallel_processing.scipy_kdtree_features(h3_grid, osm_data.get('pedestrian'), 'pedestrian')
        
        print("\n📊 Расчет площадных признаков (Pandarallel + STRtree)...")
        h3_grid = parallel_processing.parallel_area_features(h3_grid, osm_data.get('residential'), 'residential')
        h3_grid = parallel_processing.parallel_area_features(h3_grid, osm_data.get('offices'), 'office')
        h3_grid = parallel_processing.parallel_road_features(h3_grid, osm_data.get('roads'))
        
        h3_grid = features.calculate_custom_features(h3_grid)
        
        print("\n📊 Анализ конкурентной среды...")
        h3_grid = features.calculate_competitor_features(h3_grid, osm_data.get('pharmacies'))
        
        h3_grid = parallel_processing.parallel_target_variable(h3_grid, osm_data.get('pharmacies'))
        
        # Обогащение данными из data.mos.ru (опционально)
        if args.enrich or config.DATA_MOS_CONFIG.get('enabled', False):
            print("\n📊 Обогащение данными из data.mos.ru...")
            try:
                h3_grid = data_mos.enrich_h3_grid_with_mos_data(h3_grid, osm_data)
            except Exception as e:
                print(f"  ⚠️ Ошибка интеграции data.mos.ru: {e}")
        
        print("\n✓ Признаки рассчитаны")
        print(f"  - Всего признаков: {len([c for c in h3_grid.columns if c not in ['h3_cell', 'geometry', 'center_lat', 'center_lon', 'has_pharmacy']])}")
        print(f"  - Ячеек с аптеками: {h3_grid['has_pharmacy'].sum()} ({h3_grid['has_pharmacy'].mean()*100:.1f}%)")
        
        h3_grid.to_file(config.FILES['h3_grid_features'], driver='GeoJSON')
    else:
        print(f"✓ Признаки загружены из кэша: {len(h3_grid)} ячеек")
        print(f"  - Ячеек с аптеками: {h3_grid['has_pharmacy'].sum()} ({h3_grid['has_pharmacy'].mean()*100:.1f}%)")
    
    feature_cols = [c for c in h3_grid.columns if c not in ['h3_cell', 'geometry', 'center_lat', 'center_lon', 'has_pharmacy']]

    print("\n" + "="*80)
    print("📊 ЭТАП 3: РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")
    print("="*80)
    
    X_all_eda, _ = modeling.prepare_data(h3_grid, feature_cols)
    
    print("\n📈 Построение матрицы корреляции...")
    visualization.plot_correlation_matrix(X_all_eda, feature_cols)
    
    print("📈 Анализ распределения целевой переменной...")
    visualization.plot_target_distribution(h3_grid, target_col='has_pharmacy')
    
    print("📈 Построение распределений признаков...")
    key_features = [f for f in feature_cols if any(x in f for x in ['density_500m', 'coverage', 'synergy', 'index'])][:16]
    visualization.plot_feature_distributions(X_all_eda, key_features)
    
    print("📈 Построение box plots...")
    visualization.plot_feature_boxplots(h3_grid, key_features[:12], target_col='has_pharmacy')

    print("\n" + "="*80)
    print("📦 ЭТАП 4: КЛАСТЕРНЫЙ АНАЛИЗ ТЕРРИТОРИЙ")
    print("="*80)
    
    X_cluster, _ = modeling.prepare_data(h3_grid, feature_cols)
    
    print("\n🔍 Определение оптимального числа кластеров...")
    best_k = modeling.analyze_clusters_optimal_k(X_cluster, max_k=10)
    
    cluster_labels, kmeans_model, _ = modeling.perform_clustering(X_cluster, n_clusters=best_k)
    h3_grid['cluster'] = cluster_labels
    
    print(f"\n✓ Кластеризация завершена (k={best_k})")
    
    print("\n📊 Визуализация кластеров...")
    visualization.plot_cluster_pca(X_cluster, cluster_labels)
    
    if len(h3_grid) < 5000:
        visualization.plot_cluster_tsne(X_cluster, cluster_labels)
    else:
        print("  ⚠️ t-SNE пропущен (слишком много точек, используйте PCA)")
    
    visualization.plot_cluster_profiles(h3_grid, cluster_col='cluster', features=key_features[:10])
    
    cluster_profiles, cluster_descriptions = analysis.describe_clusters(h3_grid, cluster_col='cluster')

    print("\n" + "="*80)
    print("🔬 ЭТАП 4.5: АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ (VIF)")
    print("="*80)
    
    # VIF анализ
    print("\n📊 Расчёт VIF для обнаружения мультиколлинеарности...")
    try:
        vif_df = features.calculate_vif(X_cluster)
        visualization.plot_vif_analysis(vif_df)
        
        # Удаление высококоррелированных признаков
        print("\n📊 Удаление высококоррелированных признаков...")
        X_cluster_clean_corr, removed_corr = features.remove_highly_correlated_features(X_cluster)
        
        high_vif_count = (vif_df['VIF'] > config.FEATURE_CONFIG['vif_threshold']).sum()
        if high_vif_count > 0:
            print(f"⚠️ Найдено {high_vif_count} признаков с VIF > {config.FEATURE_CONFIG['vif_threshold']}")
    except Exception as e:
        print(f"⚠️ Ошибка VIF анализа: {e}")
        X_cluster_clean_corr = X_cluster
        removed_corr = []
    
    print("\n" + "="*80)
    print("🤖 ЭТАП 5: МАШИННОЕ ОБУЧЕНИЕ")
    print("="*80)
    
    results = None
    feature_cols_extended = None
    
    exclude_leakage = args.no_leakage if hasattr(args, 'no_leakage') else False
    
    if h3_grid['has_pharmacy'].sum() > 5:
        if exclude_leakage:
            print("\n🔒 Режим без утечки данных (--no-leakage)")
            feature_cols_clean = modeling.filter_leakage_features(feature_cols, exclude_leakage=True)
            # Также исключаем высококоррелированные
            feature_cols_clean = [c for c in feature_cols_clean if c not in removed_corr]
            X_cluster_clean, _ = modeling.prepare_data(h3_grid, feature_cols_clean)
        else:
            feature_cols_clean = [c for c in feature_cols if c not in removed_corr]
            X_cluster_clean = X_cluster_clean_corr
        
        print("\n📊 Добавление кластерных признаков...")
        X_with_clusters, _, _, _ = modeling.add_cluster_features(X_cluster_clean, n_clusters=best_k)
        
        feature_cols_extended = list(X_with_clusters.columns)
        
        y = h3_grid['has_pharmacy']
        
        if train_new_model:
            print("\n🚀 Обучение моделей с нуля...")
            if exclude_leakage:
                print("   (без признаков с утечкой — честная оценка)")
            
            best_model, results, X_test, y_test = modeling.train_models(X_with_clusters, y, h3_grid=h3_grid)
            
            best_f1 = max(r.get('f1', 0) for k, r in results.items() if not k.startswith('_'))
            if best_f1 >= 0.99 and not exclude_leakage:
                print("\n" + "⚠️"*20)
                print("⚠️ ВНИМАНИЕ: F1-score ≈ 1.0 может указывать на переобучение!")
                print("   Возможные причины:")
                print("   1. Сильный дисбаланс классов")
                print("   2. Утечка данных (data leakage) — признаки pharmacy_* напрямую связаны с целевой")
                print("   3. Недостаточно данных для надежной оценки")
                print("\n   💡 Рекомендация: перезапустите с флагом --no-leakage")
                print("      python3 -m app_pharmacy.main --train --no-leakage")
                print("⚠️"*20 + "\n")
            
            modeling.save_model(
                best_model, 
                config.FILES['model'],
                feature_names=feature_cols_extended,
                exclude_leakage=exclude_leakage
            )
        else:
            print("\n📂 Загрузка существующей модели...")
            best_model, saved_features, saved_exclude_leakage = modeling.load_model(config.FILES['model'])
            print(f"✓ Модель загружена из {config.FILES['model']}")
            
            # Используем признаки, с которыми модель была обучена
            if saved_features is not None:
                print(f"   Признаков при обучении: {len(saved_features)}")
                if saved_exclude_leakage:
                    print("   (обучена без признаков с утечкой)")
                    exclude_leakage = True
                    feature_cols_clean = modeling.filter_leakage_features(feature_cols, exclude_leakage=True)
                    X_cluster_clean, _ = modeling.prepare_data(h3_grid, feature_cols_clean)
                feature_cols_extended = saved_features
        
        if results is not None:
            print("\n" + "="*80)
            print("📈 ЭТАП 6: ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
            print("="*80)
            
            print("\n📊 Сравнение моделей...")
            visualization.plot_models_comparison(results)
            
            test_data = results.get('_test_data', {})
            if test_data:
                y_test_viz = test_data['y_test']
                
                for model_name in ['Baseline (LogReg)', 'RandomForest', 'CatBoost']:
                    if model_name in results and 'y_proba' in results[model_name]:
                        y_proba = results[model_name]['y_proba']
                        y_pred = results[model_name]['y_pred']
                        
                        visualization.plot_roc_pr_curves(
                            y_test_viz, y_proba, 
                            model_name=model_name,
                            filename=f'roc_pr_{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
                        )
                        
                        visualization.plot_confusion_matrix(
                            y_test_viz, y_pred,
                            model_name=model_name,
                            filename=f'confusion_matrix_{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'
                        )
            
            print("\n📊 Анализ важности признаков...")
            visualization.plot_feature_importance(best_model, feature_cols_extended)
            
            # Визуализация бизнес-метрик
            best_model_name = max(
                [k for k in results.keys() if not k.startswith('_')],
                key=lambda k: results[k].get('f1', 0)
            )
            print("\n📊 Визуализация бизнес-метрик...")
            visualization.plot_business_metrics(results[best_model_name], model_name=best_model_name)
        else:
            print("\n📊 Этап 6 пропущен (модель загружена из кэша)")
        
        print("\n🔮 Генерация предсказаний для всей территории...")
        X_all_extended, _, _, _ = modeling.add_cluster_features(X_cluster_clean, n_clusters=best_k)
        h3_grid['prediction_score'] = best_model.predict_proba(X_all_extended)[:, 1]
        
    else:
        print("⚠️ Недостаточно данных для моделирования. Пропуск ML этапа.")
        h3_grid['prediction_score'] = 0
        
    print("\n" + "="*80)
    print("💡 ЭТАП 7: АНАЛИЗ И ФОРМИРОВАНИЕ РЕКОМЕНДАЦИЙ")
    print("="*80)
    
    h3_grid = analysis.calculate_potential(h3_grid)
    
    top_cells, recommendations = analysis.get_recommendations(h3_grid, min_distance_to_competitor=300)
    
    analysis.print_detailed_recommendations(recommendations[:10])
    
    analysis.save_detailed_report(recommendations, roi_params)
    
    analysis.generate_cluster_report(h3_grid, cluster_col='cluster')
        
    h3_grid.to_file(config.FILES['h3_grid_final'], driver='GeoJSON')
    pd.DataFrame(recommendations).to_csv(config.FILES['top_10_summary'], index=False)
    
    print("\n" + "="*80)
    print("🗺️ ЭТАП 8: СОЗДАНИЕ ИНТЕРАКТИВНЫХ КАРТ")
    print("="*80)
    
    competitors = osm_data.get('pharmacies')
    visualization.create_potential_map(h3_grid, roi_params, competitors=competitors, recommendations=recommendations)
    
    print("\n" + "="*80)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("="*80)
    
    print("\n📁 Сгенерированные файлы:")
    print("  📊 Данные:")
    print(f"     - {config.FILES['h3_grid_features']}")
    print(f"     - {config.FILES['h3_grid_final']}")
    print(f"     - {config.FILES['top_10_summary']}")
    print("  🤖 Модель:")
    print(f"     - {config.FILES['model']}")
    print("  📈 Визуализации:")
    print(f"     - {config.DATA_DIR}/correlation_matrix.png")
    print(f"     - {config.DATA_DIR}/feature_importance.png")
    print(f"     - {config.DATA_DIR}/roc_pr_*.png")
    print(f"     - {config.DATA_DIR}/confusion_matrix_*.png")
    print(f"     - {config.DATA_DIR}/models_comparison.png")
    print(f"     - {config.DATA_DIR}/cluster_*.png")
    print("  🗺️ Карты:")
    print(f"     - {config.DATA_DIR}/potential_map.html")
    
    print("\n🎯 Основные результаты:")
    print(f"  - Проанализировано территорий: {len(h3_grid)}")
    print(f"  - Выявлено кластеров: {best_k}")
    print(f"  - Рекомендовано локаций: {len(recommendations)}")
    
    if 'prediction_score' in h3_grid.columns:
        high_potential = (h3_grid['potential_score'] > 0.6).sum()
        print(f"  - Территорий с высоким потенциалом: {high_potential}")

if __name__ == "__main__":
    main()
