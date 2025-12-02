import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
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
    """Подготовка данных для обучения: обработка пропусков и бесконечностей."""
    if exclude_leakage:
        feature_columns = filter_leakage_features(feature_columns, exclude_leakage=True)
    
    df = h3_grid.copy()
    
    # Обработка пропусков
    for col in feature_columns:
        if col not in df.columns:
            continue
        if 'distance' in col:
            df[col] = df[col].fillna(10000)
        elif 'density' in col or 'count' in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(df[col].median())
            
    # Обработка бесконечных значений
    for col in feature_columns:
        if col not in df.columns:
            continue
        if np.isinf(df[col]).any():
            max_val = df[df[col] != np.inf][col].max()
            df[col] = df[col].replace([np.inf, -np.inf], max_val)

    existing_cols = [c for c in feature_columns if c in df.columns]
    X = df[existing_cols]
    y = df['has_pharmacy']
    
    return X, y


def spatial_cross_validation(X, y, h3_grid, n_splits=5):
    """Пространственная кросс-валидация на основе кластеров H3 ячеек."""
    from sklearn.cluster import KMeans
    
    coords = h3_grid[['center_lat', 'center_lon']].values
    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
    spatial_clusters = kmeans.fit_predict(coords)
    
    splits = []
    for fold in range(n_splits):
        test_idx = np.where(spatial_clusters == fold)[0]
        train_idx = np.where(spatial_clusters != fold)[0]
        
        if y.iloc[test_idx].sum() > 0:
            splits.append((train_idx, test_idx))
    
    return splits


def spatial_cv_score(model, X, y, h3_grid, scoring='f1'):
    """Оценка модели с пространственной кросс-валидацией."""
    splits = spatial_cross_validation(X, y, h3_grid, n_splits=5)
    
    if len(splits) < 2:
        print("⚠️ Недостаточно пространственных фолдов с положительными примерами")
        return [], 0.0
    
    scores = []
    for train_idx, test_idx in splits:
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
        
        try:
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)
            
            if scoring == 'f1':
                score = f1_score(y_test, y_pred, zero_division=0)
            elif scoring == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            else:
                score = f1_score(y_test, y_pred, zero_division=0)
            
            scores.append(score)
        except Exception as e:
            print(f"   ⚠️ Ошибка в фолде: {e}")
            continue
    
    return scores, np.mean(scores) if scores else 0.0

def objective_rf(trial, X, y, cv=3):
    """Objective function for Random Forest optimization"""
    params = {
        'n_estimators': 50,
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }
    
    clf = RandomForestClassifier(**params)
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42, k_neighbors=min(5, y.sum()-1))),
        ('classifier', clf)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    return scores.mean()

def objective_catboost(trial, X, y, cv=3):
    """Objective function for CatBoost optimization"""
    params = {
        'iterations': 50,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-2, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': False,
        'random_state': 42,
        'allow_writing_files': False
    }
    
    clf = CatBoostClassifier(**params)
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42, k_neighbors=min(5, y.sum()-1))),
        ('classifier', clf)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    return scores.mean()

def train_baseline_model(X, y):
    """Обучение baseline модели (Logistic Regression) для сравнения."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.ML_CONFIG['test_size'], 
        stratify=y, 
        random_state=config.ML_CONFIG['random_state']
    )
    
    print("\n📊 Обучение Baseline модели (Logistic Regression)...")
    print(f"   Размер обучающей выборки: {len(y_train)} (положительных: {y_train.sum()})")
    
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    
    results = {
        'model': pipeline,
        'f1': f1_score(y_test, y_pred),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
        },
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"   ✓ Baseline F1: {results['f1']:.4f}")
    print(f"   ✓ Baseline CV F1: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
    print(f"   ✓ Baseline ROC AUC: {results['metrics']['roc_auc']:.4f}")
    
    return pipeline, results


def train_models(X, y, h3_grid=None, use_spatial_cv=False):
    """Обучение моделей с оптимизацией гиперпараметров через Optuna."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.ML_CONFIG['test_size'], 
        stratify=y, 
        random_state=config.ML_CONFIG['random_state']
    )
    
    print(f"Обучение моделей. Баланс классов в обучении: {Counter(y_train)}")
    print(f"Размер тестовой выборки: {len(y_test)} (положительных: {y_test.sum()})")
    print(f"Количество признаков: {X.shape[1]}")
    
    if len(y_test) < 20 or y_test.sum() < 3:
        print("⚠️ ВНИМАНИЕ: Тестовая выборка очень маленькая. Метрики могут быть ненадежными.")
        print("   Рекомендуется использовать кросс-валидацию для оценки качества.")
    
    results = {}
    best_model = None
    best_score = -1
    best_name = ""
    
    n_trials = config.ML_CONFIG['optuna_trials']
    cv_folds = config.ML_CONFIG['optuna_cv_folds']
    print("\n📊 Обучение Baseline модели (Logistic Regression)...")
    baseline_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    
    baseline_pipeline.fit(X_train, y_train)
    y_pred_baseline = baseline_pipeline.predict(X_test)
    y_proba_baseline = baseline_pipeline.predict_proba(X_test)[:, 1]
    f1_baseline = f1_score(y_test, y_pred_baseline)
    cv_scores_baseline = cross_val_score(baseline_pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    
    results['Baseline (LogReg)'] = {
        'model': baseline_pipeline,
        'f1': f1_baseline,
        'cv_f1_mean': cv_scores_baseline.mean(),
        'cv_f1_std': cv_scores_baseline.std(),
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred_baseline),
            'precision': precision_score(y_test, y_pred_baseline, zero_division=0),
            'recall': recall_score(y_test, y_pred_baseline, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_baseline) if len(np.unique(y_test)) > 1 else 0.0
        },
        'y_pred': y_pred_baseline,
        'y_proba': y_proba_baseline
    }
    
    print(f"  ✓ Baseline F1: {f1_baseline:.4f} (CV: {cv_scores_baseline.mean():.4f} ± {cv_scores_baseline.std():.4f})")
    
    if f1_baseline > best_score:
        best_score = f1_baseline
        best_model = baseline_pipeline
        best_name = 'Baseline (LogReg)'
    
    # --- Random Forest ---
    print(f"\n🔍 Оптимизация Random Forest с Optuna ({n_trials} trials)...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, cv=cv_folds), n_trials=n_trials)
    
    print(f"  Лучшие параметры RF: {study_rf.best_params}")
    print(f"  Лучший CV F1: {study_rf.best_value:.4f}")
    
    rf_params = study_rf.best_params
    rf_params['n_estimators'] = 1000
    rf_params['random_state'] = 42
    rf_params['n_jobs'] = -1
    
    rf_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))),
        ('classifier', RandomForestClassifier(**rf_params))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf)
    
    cv_scores_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    
    y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    
    results['RandomForest'] = {
        'model': rf_pipeline,
        'f1': f1_rf,
        'cv_f1_mean': cv_scores_rf.mean(),
        'cv_f1_std': cv_scores_rf.std(),
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf, zero_division=0),
            'recall': recall_score(y_test, y_pred_rf, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_rf) if len(np.unique(y_test)) > 1 else 0.0
        },
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf
    }
    
    if f1_rf > best_score:
        best_score = f1_rf
        best_model = rf_pipeline
        best_name = 'RandomForest'

    print(f"\n🔍 Оптимизация CatBoost с Optuna ({n_trials} trials)...")
    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(lambda trial: objective_catboost(trial, X_train, y_train, cv=cv_folds), n_trials=n_trials)
    
    print(f"  Лучшие параметры CatBoost: {study_cb.best_params}")
    print(f"  Лучший CV F1: {study_cb.best_value:.4f}")
    
    cb_params = study_cb.best_params
    cb_params['iterations'] = 1000
    cb_params['verbose'] = False
    cb_params['random_state'] = 42
    cb_params['allow_writing_files'] = False
    
    cb_pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('sampler', SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))),
        ('classifier', CatBoostClassifier(**cb_params))
    ])
    
    cb_pipeline.fit(X_train, y_train)
    y_pred_cb = cb_pipeline.predict(X_test)
    f1_cb = f1_score(y_test, y_pred_cb)
    
    cv_scores_cb = cross_val_score(cb_pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    
    y_proba_cb = cb_pipeline.predict_proba(X_test)[:, 1]
    
    results['CatBoost'] = {
        'model': cb_pipeline,
        'f1': f1_cb,
        'cv_f1_mean': cv_scores_cb.mean(),
        'cv_f1_std': cv_scores_cb.std(),
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred_cb),
            'precision': precision_score(y_test, y_pred_cb, zero_division=0),
            'recall': recall_score(y_test, y_pred_cb, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_cb) if len(np.unique(y_test)) > 1 else 0.0
        },
        'y_pred': y_pred_cb,
        'y_proba': y_proba_cb
    }
    
    if f1_cb == 1.0 and len(y_test) < 10:
        print("⚠️ ВНИМАНИЕ: F1=1.0 на маленькой тестовой выборке может указывать на переобучение.")
        print(f"   CV F1 (более объективная оценка): {cv_scores_cb.mean():.4f} ± {cv_scores_cb.std():.4f}")
    
    if f1_cb > best_score:
        best_score = f1_cb
        best_model = cb_pipeline
        best_name = 'CatBoost'
            
    print(f"\n🏆 Лучшая модель: {best_name} (F1 на тесте={best_score:.4f})")
    
    best_metrics = results[best_name]['metrics']
    print("Метрики на тестовой выборке:")
    for m, v in best_metrics.items():
        print(f"  {m}: {v:.4f}")
    
    if 'cv_f1_mean' in results[best_name]:
        print(f"\nКросс-валидация (CV) F1: {results[best_name]['cv_f1_mean']:.4f} ± {results[best_name]['cv_f1_std']:.4f}")
        print("(CV метрики более объективны при маленькой тестовой выборке)")
    
    results['_test_data'] = {
        'y_test': y_test,
        'X_test': X_test
    }
        
    return best_model, results, X_test, y_test


def add_cluster_features(X, n_clusters=5):
    """Добавляет кластерные признаки к данным перед обучением модели."""
    print(f"\n📊 Предварительная кластеризация для feature engineering (k={n_clusters})...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    X_with_clusters = X.copy()
    X_with_clusters['cluster_feature'] = cluster_labels
    
    for i in range(n_clusters):
        X_with_clusters[f'cluster_{i}'] = (cluster_labels == i).astype(int)
    
    print(f"   ✓ Добавлено {n_clusters + 1} кластерных признаков")
    
    return X_with_clusters, cluster_labels, kmeans, scaler

def analyze_clusters_optimal_k(X, max_k=10):
    """Анализ оптимального числа кластеров (Elbow Method и Silhouette)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouettes = []
    K = range(2, max_k + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        
    plt.figure(figsize=(10, 5))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Инерция (Inertia)')
    plt.title('Метод локтя для выбора оптимального k')
    plt.grid(True)
    plt.savefig(config.FILES['elbow_plot'])
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(K, silhouettes, 'rx-')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Силуэтный коэффициент (Silhouette Score)')
    plt.title('Анализ силуэта для выбора оптимального k')
    plt.grid(True)
    plt.savefig(config.FILES['silhouette_plot'])
    plt.close()
    
    print(f"Графики анализа кластеров сохранены в {config.DATA_DIR}")
    
    best_k = K[np.argmax(silhouettes)]
    print(f"Оптимальное число кластеров (по Silhouette): {best_k}")
    
    return best_k

def perform_clustering(X, n_clusters=5):
    """Кластеризация территорий с использованием KMeans"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    return labels, kmeans

def save_model(model, filename, feature_names=None, exclude_leakage=False):
    """Сохранение модели с метаданными"""
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'exclude_leakage': exclude_leakage,
    }
    joblib.dump(model_data, filename)
    print(f"💾 Модель сохранена в {filename}")
    if exclude_leakage:
        print("   (обучена без признаков с утечкой)")


def load_model(filename):
    """Загрузка модели с метаданными"""
    data = joblib.load(filename)
    
    # Поддержка старого формата
    if not isinstance(data, dict) or 'model' not in data:
        return data, None, False
    
    return data['model'], data.get('feature_names'), data.get('exclude_leakage', False)


def validate_on_region(model, X_val, y_val, region_name="Валидационный регион"):
    """
    Валидация модели на данных из другого региона (out-of-domain validation).
    
    Это самый честный способ оценки модели для геопространственных задач.
    """
    print(f"\n{'='*80}")
    print(f"🔬 ВАЛИДАЦИЯ НА НЕЗАВИСИМОМ РЕГИОНЕ: {region_name}")
    print(f"{'='*80}")
    
    print(f"\n📊 Размер валидационной выборки: {len(y_val)}")
    print(f"   Положительных примеров: {y_val.sum()} ({100*y_val.sum()/len(y_val):.1f}%)")
    
    # Предсказания
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Метрики
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_proba) if y_val.sum() > 0 else 0.0
    }
    
    print(f"\n📈 Метрики на валидационном регионе:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-score:  {metrics['f1']:.4f}")
    print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Интерпретация
    print(f"\n💡 Интерпретация:")
    if metrics['f1'] >= 0.5:
        print(f"   ✅ Модель хорошо обобщается на новый регион (F1 ≥ 0.5)")
    elif metrics['f1'] >= 0.3:
        print(f"   ⚠️ Модель умеренно обобщается (0.3 ≤ F1 < 0.5)")
    else:
        print(f"   ❌ Модель плохо обобщается на новый регион (F1 < 0.3)")
        print(f"      Возможно, регионы слишком различаются по характеристикам")
    
    return metrics, y_pred, y_proba
