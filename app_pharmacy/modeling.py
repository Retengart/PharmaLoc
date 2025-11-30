import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report, silhouette_score)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from collections import Counter
import optuna
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from . import config

# Настройка логирования для Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def prepare_data(h3_grid, feature_columns):
    """
    Подготовка данных для обучения: обработка пропусков и бесконечностей.
    """
    df = h3_grid.copy()
    
    # Обработка пропусков
    for col in feature_columns:
        if col not in df.columns:
            continue
        if 'distance' in col:
            # Для расстояний используем большое значение вместо пропуска
            df[col] = df[col].fillna(10000)
        elif 'density' in col or 'count' in col:
            # Для плотности и количества - 0
            df[col] = df[col].fillna(0)
        else:
            # Для остальных - медиана
            df[col] = df[col].fillna(df[col].median())
            
    # Обработка бесконечных значений
    for col in feature_columns:
        if col not in df.columns: continue
        if np.isinf(df[col]).any():
            max_val = df[df[col] != np.inf][col].max()
            df[col] = df[col].replace([np.inf, -np.inf], max_val)

    X = df[feature_columns]
    y = df['has_pharmacy']
    
    return X, y

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
    
    # Pipeline with SMOTE
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

def train_models(X, y):
    """
    Обучение моделей (RandomForest и CatBoost) с оптимизацией гиперпараметров через Optuna.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    print(f"Обучение моделей. Баланс классов в обучении: {Counter(y_train)}")
    print(f"Размер тестовой выборки: {len(y_test)} (положительных: {y_test.sum()})")
    
    # Предупреждение о маленькой выборке
    if len(y_test) < 20 or y_test.sum() < 3:
        print("⚠️ ВНИМАНИЕ: Тестовая выборка очень маленькая. Метрики могут быть ненадежными.")
        print("   Рекомендуется использовать кросс-валидацию для оценки качества.")
    
    results = {}
    best_model = None
    best_score = -1
    best_name = ""
    
    n_trials = 50  # Увеличено до 50 итераций
    cv_folds = 5  # Увеличено для более надежной оценки
    
    # --- Random Forest ---
    print("\n🔍 Оптимизация Random Forest с Optuna (n_estimators=1000)...")
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
    
    # Дополнительная кросс-валидация для более объективной оценки
    cv_scores_rf = cross_val_score(rf_pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    
    results['RandomForest'] = {
        'model': rf_pipeline,
        'f1': f1_rf,
        'cv_f1_mean': cv_scores_rf.mean(),
        'cv_f1_std': cv_scores_rf.std(),
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf, zero_division=0),
            'recall': recall_score(y_test, y_pred_rf, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_rf) if len(np.unique(y_test)) > 1 else 0.0
        }
    }
    
    if f1_rf > best_score:
        best_score = f1_rf
        best_model = rf_pipeline
        best_name = 'RandomForest'

    # --- CatBoost ---
    print("\n🔍 Оптимизация CatBoost с Optuna (iterations=1000)...")
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
    
    # Дополнительная кросс-валидация для более объективной оценки
    cv_scores_cb = cross_val_score(cb_pipeline, X_train, y_train, cv=cv_folds, scoring='f1')
    
    results['CatBoost'] = {
        'model': cb_pipeline,
        'f1': f1_cb,
        'cv_f1_mean': cv_scores_cb.mean(),
        'cv_f1_std': cv_scores_cb.std(),
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred_cb),
            'precision': precision_score(y_test, y_pred_cb, zero_division=0),
            'recall': recall_score(y_test, y_pred_cb, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_cb) if len(np.unique(y_test)) > 1 else 0.0
        }
    }
    
    # Проверка на переобучение
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
    
    # Показываем CV метрики для лучшей модели
    if 'cv_f1_mean' in results[best_name]:
        print(f"\nКросс-валидация (CV) F1: {results[best_name]['cv_f1_mean']:.4f} ± {results[best_name]['cv_f1_std']:.4f}")
        print("(CV метрики более объективны при маленькой тестовой выборке)")
        
    return best_model, results, X_test, y_test

def analyze_clusters_optimal_k(X, max_k=10):
    """
    Анализ оптимального числа кластеров (Elbow Method и Silhouette).
    Сохраняет графики в файлы.
    """
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
        
    # Plot Elbow
    plt.figure(figsize=(10, 5))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Инерция (Inertia)')
    plt.title('Метод локтя для выбора оптимального k')
    plt.grid(True)
    plt.savefig(config.FILES['elbow_plot'])
    plt.close()
    
    # Plot Silhouette
    plt.figure(figsize=(10, 5))
    plt.plot(K, silhouettes, 'rx-')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Силуэтный коэффициент (Silhouette Score)')
    plt.title('Анализ силуэта для выбора оптимального k')
    plt.grid(True)
    plt.savefig(config.FILES['silhouette_plot'])
    plt.close()
    
    print(f"Графики анализа кластеров сохранены в {config.DATA_DIR}")
    
    # Simple heuristic: max silhouette
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

def save_model(model, filename):
    """Сохранение модели на диск"""
    joblib.dump(model, filename)
    print(f"💾 Модель сохранена в {filename}")
