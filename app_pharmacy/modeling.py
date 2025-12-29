"""
Модуль машинного обучения для геомаркетингового анализа.
Включает: подготовку данных, обучение моделей, калибровку, оценку качества.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, silhouette_score, average_precision_score)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from collections import Counter
import optuna
from catboost import CatBoostClassifier

# Бэкенд без GUI для избежания конфликтов с многопоточностью
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config

LEAKAGE_FEATURES = config.LEAKAGE_FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_model_feature_names(model):
    """
    Возвращает имена признаков, которые ожидает модель при predict/predict_proba.

    Нужно для корректного выравнивания DataFrame колонок при инференсе,
    особенно когда модель обёрнута в CalibratedClassifierCV / Pipeline.
    """
    def _extract(obj):
        if obj is None:
            return None

        # 1) Стандартное поле sklearn (есть у многих трансформеров/пайплайнов)
        if hasattr(obj, "feature_names_in_"):
            try:
                return list(obj.feature_names_in_)
            except Exception:
                pass

        # 2) Pipeline-подобные объекты (sklearn / imblearn)
        if hasattr(obj, "named_steps"):
            # Иногда у самого pipeline нет feature_names_in_, но есть у шагов
            try:
                for step in obj.named_steps.values():
                    names = _extract(step)
                    if names:
                        return names
            except Exception:
                pass

        # 3) CalibratedClassifierCV хранит список калиброванных классификаторов
        if hasattr(obj, "calibrated_classifiers_"):
            try:
                for cc in obj.calibrated_classifiers_:
                    for attr in ("estimator", "estimator_", "base_estimator", "base_estimator_"):
                        names = _extract(getattr(cc, attr, None))
                        if names:
                            return names
            except Exception:
                pass

        # 4) Meta-estimators часто имеют estimator/base_estimator внутри
        for attr in ("estimator", "estimator_", "base_estimator", "base_estimator_"):
            names = _extract(getattr(obj, attr, None))
            if names:
                return names

        return None

    return _extract(model)


def filter_leakage_features(feature_columns, exclude_leakage=True):
    """Фильтрует признаки с утечкой данных."""
    if not exclude_leakage:
        return feature_columns
    
    filtered = [col for col in feature_columns if col not in LEAKAGE_FEATURES]
    
    excluded = [col for col in feature_columns if col in LEAKAGE_FEATURES]
    if excluded:
        print(f"⚠️ Исключены признаки с утечкой данных ({len(excluded)}):")
        for col in excluded:
            print(f"   - {col}")
    
    return filtered


def prepare_data(h3_grid, feature_columns, exclude_leakage=False):
    """
    Подготовка данных для обучения: обработка пропусков и бесконечностей.
    
    Args:
        h3_grid: GeoDataFrame с признаками и целевой переменной
        feature_columns: Список колонок-признаков
        exclude_leakage: Исключить признаки с утечкой данных
    
    Returns:
        tuple: (X, y) - признаки и целевая переменная
    """
    # ВАЛИДАЦИЯ входных данных
    if h3_grid is None or h3_grid.empty:
        raise ValueError("h3_grid не может быть пустым")
    
    if 'has_pharmacy' not in h3_grid.columns:
        raise ValueError("h3_grid должен содержать колонку 'has_pharmacy'")
    
    if not feature_columns:
        raise ValueError("feature_columns не может быть пустым")
    
    if exclude_leakage:
        feature_columns = filter_leakage_features(feature_columns, exclude_leakage=True)
    
    df = h3_grid.copy()
    
    default_distance = config.FEATURE_CONFIG['default_distance_fillna']
    default_count = config.FEATURE_CONFIG['default_count_fillna']
    
    # Обработка пропусков
    for col in feature_columns:
        if col not in df.columns:
            continue
        if 'distance' in col:
            df[col] = df[col].fillna(default_distance)
        elif 'density' in col or 'count' in col:
            df[col] = df[col].fillna(default_count)
        else:
            # ИСПРАВЛЕНО: Проверка на NaN в median() - если все значения NaN, используем 0
            median_val = df[col].median()
            if pd.isna(median_val):
                # Все значения NaN - используем 0 как fallback
                df[col] = df[col].fillna(0)
                print(f"⚠️ Все значения в '{col}' NaN, заменены на 0")
            else:
                df[col] = df[col].fillna(median_val)
            
    # Обработка бесконечных значений
    for col in feature_columns:
        if col not in df.columns:
            continue
        if np.isinf(df[col]).any():
            # ВАЛИДАЦИЯ: Проверка наличия конечных значений
            finite_vals = df[df[col] != np.inf][col]
            if len(finite_vals) > 0:
                # Есть конечные значения - используем максимум
                max_val = finite_vals.max()
                # Проверка на NaN (на случай если max_val тоже NaN)
                if np.isfinite(max_val):
                    df[col] = df[col].replace([np.inf, -np.inf], max_val)
                else:
                    # Все конечные значения тоже NaN - используем default
                    replacement = default_distance if 'distance' in col else default_count
                    df[col] = df[col].replace([np.inf, -np.inf], replacement)
            else:
                # Все значения inf - заменяем на default значение
                replacement = default_distance if 'distance' in col else default_count
                df[col] = df[col].replace([np.inf, -np.inf], replacement)
                print(f"⚠️ Все значения в '{col}' бесконечны, заменены на {replacement}")

    existing_cols = [c for c in feature_columns if c in df.columns]
    X = df[existing_cols]
    y = df['has_pharmacy']
    
    return X, y


def get_district_groups(h3_grid):
    """
    Определяет группы ячеек по районам для Spatial CV.
    Использует кластеризацию координат как прокси для районов.
    """
    coords = h3_grid[['center_lat', 'center_lon']].values
    n_groups = config.ML_CONFIG['spatial_cv_n_splits']
    
    kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
    groups = kmeans.fit_predict(coords)
    
    return groups


def spatial_group_kfold_cv(model, X, y, groups, scoring='f1'):
    """
    Пространственная кросс-валидация с использованием GroupKFold.
    Гарантирует, что соседние ячейки не попадают одновременно в train и test.
    """
    
    gkf = GroupKFold(n_splits=config.ML_CONFIG['spatial_cv_n_splits'])
    
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Проверяем наличие положительных примеров
        if y_train.sum() == 0 or y_test.sum() == 0:
            continue
        
        # Обучаем с SMOTE
        try:
            smote = SMOTE(k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
                         random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except Exception:
            X_train_res, y_train_res = X_train, y_train
        
        model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
        model_clone.fit(X_train_res, y_train_res)
        
        y_pred = model_clone.predict(X_test)
        
        if scoring == 'f1':
            score = f1_score(y_test, y_pred, zero_division=0)
        elif scoring == 'precision':
            score = precision_score(y_test, y_pred, zero_division=0)
        elif scoring == 'recall':
            score = recall_score(y_test, y_pred, zero_division=0)
        else:
            score = f1_score(y_test, y_pred, zero_division=0)
        
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


def bootstrap_metrics(y_true, y_pred, y_proba, n_iterations=None, ci=None):
    """
    Bootstrap для оценки доверительных интервалов метрик.
    
    Returns:
        dict: Метрики с CI
    """
    if n_iterations is None:
        n_iterations = config.BUSINESS_CONFIG['bootstrap_n_iterations']
    if ci is None:
        ci = config.BUSINESS_CONFIG['bootstrap_ci']
    
    n_samples = len(y_true)
    
    metrics_boot = {
        'f1': [], 'precision': [], 'recall': [], 'roc_auc': [], 'ap': []
    }
    
    for _ in range(n_iterations):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_boot = y_pred[indices] if isinstance(y_pred, np.ndarray) else y_pred.iloc[indices]
        y_proba_boot = y_proba[indices]
        
        # Пропускаем если только один класс
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        metrics_boot['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics_boot['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics_boot['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
        metrics_boot['roc_auc'].append(roc_auc_score(y_true_boot, y_proba_boot))
        metrics_boot['ap'].append(average_precision_score(y_true_boot, y_proba_boot))
    
    # Расчёт CI
    alpha = (1 - ci) / 2
    results = {}
    for metric, values in metrics_boot.items():
        if values:
            results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, alpha * 100),
                'ci_upper': np.percentile(values, (1 - alpha) * 100),
            }
    
    return results


def precision_at_k(y_true, y_proba, k_values=None):
    """
    Precision@K — какая доля из топ-K предсказаний действительно положительные.
    Критическая бизнес-метрика для задачи рекомендаций.
    """
    if k_values is None:
        k_values = config.BUSINESS_CONFIG['precision_at_k']
    
    # Сортируем по вероятности
    sorted_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = np.array(y_true)[sorted_indices]
    
    results = {}
    for k in k_values:
        if k > len(y_true):
            k = len(y_true)
        top_k_true = y_true_sorted[:k]
        p_at_k = np.sum(top_k_true) / k
        results[f'P@{k}'] = p_at_k
    
    return results


def calculate_lift(y_true, y_proba, k_values=None):
    """
    Lift@K — во сколько раз модель лучше случайного выбора.
    """
    if k_values is None:
        k_values = config.BUSINESS_CONFIG['precision_at_k']
    
    baseline = np.mean(y_true)  # Базовая вероятность
    p_at_k = precision_at_k(y_true, y_proba, k_values)
    
    results = {}
    for key, value in p_at_k.items():
        k = int(key.split('@')[1])
        lift = value / baseline if baseline > 0 else 0
        results[f'Lift@{k}'] = lift
    
    return results


def expected_value_analysis(y_proba, potential_scores):
    """
    Расчёт ожидаемой ценности локаций.
    
    Simplified model:
    EV = P(success) * potential_score * avg_revenue * location_impact
    """
    avg_revenue = config.BUSINESS_CONFIG['avg_pharmacy_revenue_monthly']
    impact = config.BUSINESS_CONFIG['location_quality_impact']
    
    # Нормализуем potential_scores
    pot_norm = (potential_scores - potential_scores.min()) / (potential_scores.max() - potential_scores.min() + 1e-10)
    
    # EV = вероятность * качество * базовая выручка * влияние качества
    ev = y_proba * pot_norm * avg_revenue * (1 + impact * pot_norm)
    
    return ev


def train_baseline_model(X_train, X_test, y_train, y_test):
    """
    Обучение baseline модели (Logistic Regression) для сравнения.
    
    ВАЖНО: Использует уже разделенные train/test данные для согласованности
    с остальными моделями.
    
    Args:
        X_train: Обучающие признаки
        X_test: Тестовые признаки
        y_train: Обучающая целевая переменная
        y_test: Тестовая целевая переменная
    
    Returns:
        tuple: (pipeline, metrics, X_test, y_test, y_pred, y_proba)
    """
    # ВАЛИДАЦИЯ входных данных
    if X_train is None or X_train.empty:
        raise ValueError("X_train не может быть пустым")
    if X_test is None or X_test.empty:
        raise ValueError("X_test не может быть пустым")
    if len(y_train) == 0:
        raise ValueError("y_train не может быть пустым")
    if len(y_test) == 0:
        raise ValueError("y_test не может быть пустым")
    
    # ВАЛИДАЦИЯ: Проверка наличия положительных примеров
    if y_train.sum() == 0:
        raise ValueError("y_train не содержит положительных примеров (все нули)")
    if y_test.sum() == 0:
        print("⚠️ y_test не содержит положительных примеров - метрики могут быть ненадежными")
    
    print(f"   Размер обучающей выборки: {len(y_train)} (положительных: {y_train.sum()})")
    
    # SMOTETomek — комбинация oversampling и cleaning
    if config.ML_CONFIG.get('use_tomek_links', False):
        resampler = SMOTETomek(
            smote=SMOTE(k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
                       random_state=42),
            random_state=42
        )
    else:
        resampler = SMOTE(
            k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
            random_state=42
        )
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('resampler', resampler),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
    }
    
    print(f"  ✓ Baseline F1: {metrics['f1']:.4f} (CV: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f})")
    
    return pipeline, metrics, X_test, y_test, y_pred, y_proba


def objective_rf(trial, X_train, y_train, cv_folds):
    """Optuna objective для Random Forest"""
    params = {
        'n_estimators': config.ML_CONFIG['rf_n_estimators'],
        'max_depth': trial.suggest_int('max_depth', *config.ML_CONFIG['rf_max_depth_range']),
        'min_samples_split': trial.suggest_int('min_samples_split', *config.ML_CONFIG['rf_min_samples_split_range']),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', *config.ML_CONFIG['rf_min_samples_leaf_range']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    clf = RandomForestClassifier(**params)
    
    if config.ML_CONFIG.get('use_tomek_links', False):
        resampler = SMOTETomek(
            smote=SMOTE(k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
                       random_state=42),
            random_state=42
        )
    else:
        resampler = SMOTE(
            k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
            random_state=42
        )
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('resampler', resampler),
        ('classifier', clf)
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    return scores.mean()


def objective_catboost(trial, X_train, y_train, cv_folds):
    """Optuna objective для CatBoost"""
    params = {
        'iterations': config.ML_CONFIG['cb_iterations'],
        'depth': trial.suggest_int('depth', *config.ML_CONFIG['cb_depth_range']),
        'learning_rate': trial.suggest_float('learning_rate', *config.ML_CONFIG['cb_learning_rate_range'], log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *config.ML_CONFIG['cb_l2_leaf_reg_range'], log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 255]),
        'auto_class_weights': 'Balanced',
        'random_seed': 42,
        'verbose': False
    }
    
    if config.ML_CONFIG.get('use_tomek_links', False):
        resampler = SMOTETomek(
            smote=SMOTE(k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
                       random_state=42),
            random_state=42
        )
    else:
        resampler = SMOTE(
            k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
            random_state=42
        )
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('resampler', resampler),
        ('classifier', CatBoostClassifier(**params))
    ])
    
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    return scores.mean()


def train_models(X, y, h3_grid=None, use_spatial_cv=None, n_clusters=None):
    """
    Обучение моделей с оптимизацией гиперпараметров через Optuna.
    Включает калибровку вероятностей и расчёт бизнес-метрик.
    
    Args:
        X: Признаки
        y: Целевая переменная
        h3_grid: GeoDataFrame с H3 ячейками (для Spatial CV)
        use_spatial_cv: Использовать пространственную кросс-валидацию.
                       Если None, автоматически True если h3_grid предоставлен.
        n_clusters: Число кластеров для feature engineering. Если None, определяется автоматически.
    """
    # Автоматически включаем Spatial CV если доступны пространственные данные
    if use_spatial_cv is None:
        use_spatial_cv = (h3_grid is not None)
    
    if use_spatial_cv and h3_grid is None:
        print("⚠️ Spatial CV запрошен, но h3_grid не предоставлен. Используется обычный train_test_split.")
        use_spatial_cv = False
    
    # Используем пространственное разбиение если доступно
    if use_spatial_cv:
        print("\n🗺️ Использование пространственного разбиения данных (Spatial CV)...")
        from sklearn.model_selection import GroupShuffleSplit
        
        # Создаем группы по пространственной близости
        groups = get_district_groups(h3_grid)
        
        # Проверяем соответствие размеров
        if len(groups) != len(X):
            print(f"⚠️ Размер групп ({len(groups)}) не совпадает с размером X ({len(X)})")
            print("   Используется обычный train_test_split вместо Spatial CV")
            use_spatial_cv = False
        else:
            # Используем GroupShuffleSplit для пространственного разбиения
            gss = GroupShuffleSplit(n_splits=1, test_size=config.ML_CONFIG['test_size'], 
                                    random_state=config.ML_CONFIG['random_state'])
            train_idx, test_idx = next(gss.split(X, y, groups))
            
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            
            print(f"   Пространственное разбиение: train={len(y_train)}, test={len(y_test)}")
            print(f"   Групп: {len(np.unique(groups))}")
        
        # Если Spatial CV не сработал из-за несоответствия размеров, используем обычный split
        if not use_spatial_cv:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.ML_CONFIG['test_size'], 
                random_state=config.ML_CONFIG['random_state'],
                stratify=y
            )
    else:
        # Если Spatial CV не использовался, используем обычный split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.ML_CONFIG['test_size'], 
            random_state=config.ML_CONFIG['random_state'],
            stratify=y
        )
    
    print(f"Обучение моделей. Баланс классов в обучении: {Counter(y_train)}")
    print(f"Размер тестовой выборки: {len(y_test)} (положительных: {y_test.sum()})")
    print(f"Количество признаков: {X.shape[1]}")
    
    if len(y_test) < 20 or y_test.sum() < 3:
        print("⚠️ ВНИМАНИЕ: Тестовая выборка очень маленькая. Метрики могут быть ненадежными.")
        print("   Рекомендуется использовать bootstrap CI для оценки неопределённости.")
    
    # Добавляем кластерные признаки ПОСЛЕ разделения на train/test
    # Это критично для избежания переобучения
    if h3_grid is not None:
        # Определяем оптимальное число кластеров на train данных
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # ВАЛИДАЦИЯ: Определяем число кластеров
        if n_clusters is None:
            # Автоматическое определение оптимального числа кластеров
            # Используем простую эвристику: sqrt(n_samples / 2) с ограничениями
            n_samples = len(X_train)
            n_clusters_for_features = max(2, min(10, int(np.sqrt(n_samples / 2))))
            print(f"   Автоматически определено число кластеров для feature engineering: {n_clusters_for_features}")
        else:
            # ВАЛИДАЦИЯ: Проверка корректности переданного значения
            if not isinstance(n_clusters, int) or n_clusters < 2:
                print(f"⚠️ Некорректное число кластеров ({n_clusters}), используется значение по умолчанию")
                n_clusters_for_features = 5
            else:
                n_clusters_for_features = min(n_clusters, 20)  # Ограничение сверху
                print(f"   Используется заданное число кластеров для feature engineering: {n_clusters_for_features}")
        
        scaler_temp = StandardScaler()
        X_train_scaled_temp = scaler_temp.fit_transform(X_train)
        
        kmeans_temp = KMeans(n_clusters=n_clusters_for_features, random_state=42, n_init=10)
        kmeans_temp.fit(X_train_scaled_temp)
        
        # Добавляем кластерные признаки к train и test данным
        X_train_with_clusters, X_test_with_clusters, kmeans_model, scaler_model = add_cluster_features(
            X_train, X_test, n_clusters=n_clusters_for_features,
            kmeans_model=kmeans_temp, scaler_model=scaler_temp
        )
        
        X_train = X_train_with_clusters
        X_test = X_test_with_clusters if X_test_with_clusters is not None else X_test
    
    results = {}
    best_model = None
    best_f1 = 0
    best_name = ""
    
    n_trials = config.ML_CONFIG['optuna_trials']
    cv_folds = config.ML_CONFIG['optuna_cv_folds']
    
    # --- Baseline ---
    print("\n📊 Обучение Baseline модели (Logistic Regression)...")
    # Используем те же train/test данные, что и для остальных моделей
    baseline_pipeline, baseline_metrics, _, _, baseline_pred, baseline_proba = train_baseline_model(
        X_train, X_test, y_train, y_test
    )
    results['Baseline (LogReg)'] = baseline_metrics
    results['Baseline (LogReg)']['y_proba'] = baseline_proba
    results['Baseline (LogReg)']['y_pred'] = baseline_pred
    
    if baseline_metrics['f1'] > best_f1:
        best_f1 = baseline_metrics['f1']
        best_model = baseline_pipeline
        best_name = 'Baseline (LogReg)'
    
    # --- Random Forest ---
    print(f"\n🔍 Оптимизация Random Forest с Optuna ({n_trials} trials)...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, cv_folds), 
                      n_trials=n_trials, show_progress_bar=False)
    
    best_params_rf = study_rf.best_params
    print(f"  Лучшие параметры RF: {best_params_rf}")
    print(f"  Лучший CV F1: {study_rf.best_value:.4f}")
    
    # Resampler
    if config.ML_CONFIG.get('use_tomek_links', False):
        resampler = SMOTETomek(
            smote=SMOTE(k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
                       random_state=42),
            random_state=42
        )
    else:
        resampler = SMOTE(
            k_neighbors=min(config.ML_CONFIG['smote_k_neighbors'], y_train.sum()-1),
            random_state=42
        )
    
    rf_clf = RandomForestClassifier(
        n_estimators=config.ML_CONFIG['rf_n_estimators'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        **best_params_rf
    )
    
    rf_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('resampler', resampler),
        ('classifier', rf_clf)
    ])
    
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    f1_rf = f1_score(y_test, y_pred_rf)
    
    cv_scores_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    
    results['RandomForest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'f1': f1_rf,
        'roc_auc': roc_auc_score(y_test, y_proba_rf) if y_test.sum() > 0 else 0,
        'cv_f1_mean': cv_scores_rf.mean(),
        'cv_f1_std': cv_scores_rf.std(),
        'y_proba': y_proba_rf,
        'y_pred': y_pred_rf
    }
    
    if f1_rf > best_f1:
        best_f1 = f1_rf
        best_model = rf_pipeline
        best_name = 'RandomForest'
    
    # --- CatBoost ---
    print(f"\n🔍 Оптимизация CatBoost с Optuna ({n_trials} trials)...")
    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(lambda trial: objective_catboost(trial, X_train, y_train, cv_folds),
                      n_trials=n_trials, show_progress_bar=False)
    
    best_params_cb = study_cb.best_params
    print(f"  Лучшие параметры CatBoost: {best_params_cb}")
    print(f"  Лучший CV F1: {study_cb.best_value:.4f}")
    
    cb_clf = CatBoostClassifier(
        iterations=config.ML_CONFIG['cb_iterations'],
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=False,
        **best_params_cb
    )
    
    cb_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('resampler', resampler),
        ('classifier', cb_clf)
    ])
    
    cb_pipeline.fit(X_train, y_train)
    y_pred_cb = cb_pipeline.predict(X_test)
    y_proba_cb = cb_pipeline.predict_proba(X_test)[:, 1]
    f1_cb = f1_score(y_test, y_pred_cb)
    
    cv_scores_cb = cross_val_score(cb_pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    
    results['CatBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred_cb),
        'precision': precision_score(y_test, y_pred_cb, zero_division=0),
        'recall': recall_score(y_test, y_pred_cb, zero_division=0),
        'f1': f1_cb,
        'roc_auc': roc_auc_score(y_test, y_proba_cb) if y_test.sum() > 0 else 0,
        'cv_f1_mean': cv_scores_cb.mean(),
        'cv_f1_std': cv_scores_cb.std(),
        'y_proba': y_proba_cb,
        'y_pred': y_pred_cb
    }
    
    if f1_cb > best_f1:
        best_f1 = f1_cb
        best_model = cb_pipeline
        best_name = 'CatBoost'
    
    # Предупреждение о переобучении
    if f1_cb == 1.0 and len(y_test) < 10:
        print("⚠️ ВНИМАНИЕ: F1=1.0 на маленькой тестовой выборке может указывать на переобучение.")
    
    print(f"\n🏆 Лучшая модель: {best_name} (F1 на тесте={best_f1:.4f})")
    
    # --- Калибровка вероятностей ---
    print("\n🔧 Калибровка вероятностей...")
    try:
        calibrated_model = CalibratedClassifierCV(
            best_model, 
            method=config.ML_CONFIG['calibration_method'],
            cv=min(config.ML_CONFIG['calibration_cv'], y_train.sum())
        )
        calibrated_model.fit(X_train, y_train)
        y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Проверяем улучшение по Brier Score
        from sklearn.metrics import brier_score_loss
        brier_before = brier_score_loss(y_test, results[best_name]['y_proba'])
        brier_after = brier_score_loss(y_test, y_proba_calibrated)
        
        if brier_after < brier_before:
            print(f"  ✓ Калибровка улучшила Brier Score: {brier_before:.4f} → {brier_after:.4f}")
            best_model = calibrated_model
            results[best_name]['y_proba_calibrated'] = y_proba_calibrated
        else:
            print(f"  ⚠️ Калибровка не улучшила модель (Brier: {brier_before:.4f} → {brier_after:.4f})")
    except Exception as e:
        print(f"  ⚠️ Ошибка калибровки: {e}")
    
    # --- Бизнес-метрики ---
    print("\n📊 Расчёт бизнес-метрик...")
    best_proba = results[best_name]['y_proba']
    best_pred = results[best_name]['y_pred']
    
    # Precision@K
    p_at_k = precision_at_k(y_test, best_proba)
    print(f"  Precision@K: {p_at_k}")
    results[best_name]['precision_at_k'] = p_at_k
    
    # Lift
    lift = calculate_lift(y_test, best_proba)
    print(f"  Lift@K: {lift}")
    results[best_name]['lift'] = lift
    
    # Bootstrap CI
    if len(y_test) >= 10:
        print("  Расчёт Bootstrap CI (это может занять время)...")
        bootstrap_results = bootstrap_metrics(y_test, best_pred, best_proba, n_iterations=500)
        results[best_name]['bootstrap_ci'] = bootstrap_results
        print(f"  F1 95% CI: [{bootstrap_results['f1']['ci_lower']:.4f}, {bootstrap_results['f1']['ci_upper']:.4f}]")
    
    # Spatial CV если доступен h3_grid
    if h3_grid is not None and use_spatial_cv:
        print("\n🗺️ Пространственная кросс-валидация...")
        groups = get_district_groups(h3_grid)
        spatial_mean, spatial_std = spatial_group_kfold_cv(
            rf_clf, X, y, groups
        )
        print(f"  Spatial CV F1: {spatial_mean:.4f} ± {spatial_std:.4f}")
        results[best_name]['spatial_cv_f1'] = spatial_mean
        results[best_name]['spatial_cv_std'] = spatial_std
    
    # Сводка метрик
    print(f"\n📈 Метрики на тестовой выборке ({best_name}):")
    for m, v in results[best_name].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            print(f"  {m}: {v:.4f}")
    
    if 'cv_f1_mean' in results[best_name]:
        print(f"\nКросс-валидация (CV) F1: {results[best_name]['cv_f1_mean']:.4f} ± {results[best_name]['cv_f1_std']:.4f}")
        print("(CV метрики более объективны при маленькой тестовой выборке)")
    
    results['_test_data'] = {
        'y_test': y_test,
        'X_test': X_test
    }
    
    # Сохраняем модели кластеризации для использования при предсказаниях
    if 'kmeans_model' in locals() and 'scaler_model' in locals():
        results['_cluster_models'] = {
            'kmeans': kmeans_model,
            'scaler': scaler_model,
            'n_clusters': n_clusters_for_features if 'n_clusters_for_features' in locals() else 5
        }
    
    return best_model, results, X_test, y_test


def add_cluster_features(X_train, X_test=None, n_clusters=5, kmeans_model=None, scaler_model=None):
    """
    Добавляет кластерные признаки к данным перед обучением модели.
    
    ВАЖНО: Кластеризация выполняется ТОЛЬКО на train данных для избежания переобучения.
    Test данные кластеризуются с использованием обученной модели кластеризации.
    
    Args:
        X_train: Обучающие данные
        X_test: Тестовые данные (опционально)
        n_clusters: Число кластеров
        kmeans_model: Предобученная модель KMeans (если None, обучается на X_train)
        scaler_model: Предобученный StandardScaler (если None, обучается на X_train)
    
    Returns:
        X_train_with_clusters, X_test_with_clusters (или None), kmeans_model, scaler_model
    """
    print(f"\n📊 Предварительная кластеризация для feature engineering (k={n_clusters})...")
    print("   ⚠️ Кластеризация выполняется ТОЛЬКО на train данных для избежания переобучения")
    
    def _align_to_expected_columns(X, expected_cols):
        """
        Приводит DataFrame к нужному набору колонок (добавляет недостающие нулями,
        удаляет лишние, сохраняет порядок). Нужен для совместимости с sklearn
        (StandardScaler.transform проверяет feature_names).
        """
        if X is None:
            return None
        if expected_cols is None:
            return X

        X_aligned = X.copy()

        # Добавляем недостающие
        missing = [c for c in expected_cols if c not in X_aligned.columns]
        for c in missing:
            X_aligned[c] = 0

        # Удаляем лишние и сохраняем порядок
        X_aligned = X_aligned[[c for c in expected_cols if c in X_aligned.columns]]
        return X_aligned

    # Обучаем scaler и kmeans на train данных
    if scaler_model is None:
        scaler_model = StandardScaler()
        X_train_scaled = scaler_model.fit_transform(X_train)
    else:
        # КРИТИЧНО: выравниваем признаки под то, на чём scaler был обучен
        expected = list(getattr(scaler_model, "feature_names_in_", [])) or None
        if expected is not None:
            X_train = _align_to_expected_columns(X_train, expected)
        X_train_scaled = scaler_model.transform(X_train)
    
    if kmeans_model is None:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_train = kmeans_model.fit_predict(X_train_scaled)
    else:
        cluster_labels_train = kmeans_model.predict(X_train_scaled)
    
    # Добавляем кластерные признаки к train данным
    X_train_with_clusters = X_train.copy()
    X_train_with_clusters['cluster_feature'] = cluster_labels_train
    
    for i in range(n_clusters):
        X_train_with_clusters[f'cluster_{i}'] = (cluster_labels_train == i).astype(int)
    
    # Обрабатываем test данные если они предоставлены
    X_test_with_clusters = None
    if X_test is not None:
        expected = list(getattr(scaler_model, "feature_names_in_", [])) or None
        if expected is not None:
            X_test = _align_to_expected_columns(X_test, expected)
        X_test_scaled = scaler_model.transform(X_test)
        cluster_labels_test = kmeans_model.predict(X_test_scaled)
        
        X_test_with_clusters = X_test.copy()
        X_test_with_clusters['cluster_feature'] = cluster_labels_test
        
        for i in range(n_clusters):
            X_test_with_clusters[f'cluster_{i}'] = (cluster_labels_test == i).astype(int)
    
    print(f"   ✓ Добавлено {n_clusters + 1} кластерных признаков")
    if X_test is not None:
        print("   ✓ Test данные кластеризованы с использованием обученной модели")
    
    return X_train_with_clusters, X_test_with_clusters, kmeans_model, scaler_model


def analyze_clusters_optimal_k(X, max_k=10):
    """Анализ оптимального числа кластеров (Elbow Method и Silhouette)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    K = range(2, max_k + 1)
    inertias = []
    silhouettes = []
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        
    # Plot Elbow
    plt.figure(figsize=(10, 5))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.savefig(config.FILES['elbow_plot'], dpi=100, bbox_inches='tight')
    plt.close()
    
    # Plot Silhouette
    plt.figure(figsize=(10, 5))
    plt.plot(K, silhouettes, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.savefig(config.FILES['silhouette_plot'], dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Графики анализа кластеров сохранены в {config.DATA_DIR}")
    
    best_k = K[np.argmax(silhouettes)]
    print(f"Оптимальное число кластеров (по Silhouette): {best_k}")
    
    return best_k


def perform_clustering(X, n_clusters):
    """Выполнение кластеризации"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, scaler


def save_model(model, filename, feature_names=None, exclude_leakage=False, results=None):
    """Сохранение модели с метаданными и результатами обучения"""
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'exclude_leakage': exclude_leakage,
        'results': results,  # Сохраняем результаты для визуализации
    }
    joblib.dump(model_data, filename)
    print(f"💾 Модель сохранена в {filename}")
    if exclude_leakage:
        print("   (обучена без признаков с утечкой)")
    if results:
        print("   (с результатами обучения для визуализации)")


def load_model(filename):
    """Загрузка модели с метаданными и результатами"""
    data = joblib.load(filename)
    
    # Поддержка старого формата
    if not isinstance(data, dict) or 'model' not in data:
        return data, None, False, None
    
    return (
        data['model'], 
        data.get('feature_names'), 
        data.get('exclude_leakage', False),
        data.get('results')  # Возвращаем сохранённые результаты
    )


def validate_on_region(model, X_val, y_val, region_name="Валидационный регион"):
    """
    Валидация модели на данных из другого региона (out-of-domain validation).
    """
    print(f"\n{'='*80}")
    print(f"🔬 ВАЛИДАЦИЯ НА НЕЗАВИСИМОМ РЕГИОНЕ: {region_name}")
    print(f"{'='*80}")
    
    print(f"\n📊 Размер валидационной выборки: {len(y_val)}")
    print(f"   Положительных примеров: {y_val.sum()} ({100*y_val.sum()/len(y_val):.1f}%)")
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_proba) if y_val.sum() > 0 else 0.0
    }
    
    print("\n📈 Метрики на валидационном регионе:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-score:  {metrics['f1']:.4f}")
    print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Бизнес-метрики
    p_at_k = precision_at_k(y_val, y_proba)
    lift = calculate_lift(y_val, y_proba)
    
    print("\n📊 Бизнес-метрики:")
    print(f"   {p_at_k}")
    print(f"   {lift}")
    
    # Интерпретация
    print("\n💡 Интерпретация:")
    if metrics['f1'] >= 0.5:
        print("   ✅ Модель хорошо обобщается на новый регион (F1 ≥ 0.5)")
    elif metrics['f1'] >= 0.3:
        print("   ⚠️ Модель умеренно обобщается (0.3 ≤ F1 < 0.5)")
    else:
        print("   ❌ Модель плохо обобщается на новый регион (F1 < 0.3)")
        print("      Возможно, регионы слишком различаются по характеристикам")
    
    return metrics, y_pred, y_proba
