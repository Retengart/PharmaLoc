import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report)
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from collections import Counter
import optuna
from catboost import CatBoostClassifier
import logging

# Настройка логирования для Optuna, чтобы не засорять вывод
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
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
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
    
    # Используем F1-score как метрику для оптимизации
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    return scores.mean()

def objective_catboost(trial, X, y, cv=3):
    """Objective function for CatBoost optimization"""
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
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
    
    # Pipeline with SMOTE
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
    Используется балансировка классов SMOTE.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    print(f"Обучение моделей. Баланс классов в обучении: {Counter(y_train)}")
    
    results = {}
    best_model = None
    best_score = -1
    best_name = ""
    
    n_trials = 20  # Количество итераций оптимизации
    cv_folds = 3
    
    # --- Random Forest ---
    print("\n🔍 Оптимизация Random Forest с Optuna...")
    study_rf = optuna.create_study(direction='maximize')
    study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, cv=cv_folds), n_trials=n_trials)
    
    print(f"  Лучшие параметры RF: {study_rf.best_params}")
    print(f"  Лучший CV F1: {study_rf.best_value:.4f}")
    
    # Обучаем финальную RF модель
    rf_params = study_rf.best_params
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
    
    results['RandomForest'] = {
        'model': rf_pipeline,
        'f1': f1_rf,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf, zero_division=0),
            'recall': recall_score(y_test, y_pred_rf, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_rf)
        }
    }
    
    if f1_rf > best_score:
        best_score = f1_rf
        best_model = rf_pipeline
        best_name = 'RandomForest'

    # --- CatBoost ---
    print("\n🔍 Оптимизация CatBoost с Optuna...")
    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(lambda trial: objective_catboost(trial, X_train, y_train, cv=cv_folds), n_trials=n_trials)
    
    print(f"  Лучшие параметры CatBoost: {study_cb.best_params}")
    print(f"  Лучший CV F1: {study_cb.best_value:.4f}")
    
    # Обучаем финальную CatBoost модель
    cb_params = study_cb.best_params
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
    
    results['CatBoost'] = {
        'model': cb_pipeline,
        'f1': f1_cb,
        'metrics': {
            'accuracy': accuracy_score(y_test, y_pred_cb),
            'precision': precision_score(y_test, y_pred_cb, zero_division=0),
            'recall': recall_score(y_test, y_pred_cb, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_cb)
        }
    }
    
    if f1_cb > best_score:
        best_score = f1_cb
        best_model = cb_pipeline
        best_name = 'CatBoost'
            
    print(f"\n🏆 Лучшая модель: {best_name} (F1={best_score:.4f})")
    
    # Вывод метрик для лучшей модели
    best_metrics = results[best_name]['metrics']
    print("Метрики на тестовой выборке:")
    for m, v in best_metrics.items():
        print(f"  {m}: {v:.4f}")
        
    return best_model, results, X_test, y_test

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
