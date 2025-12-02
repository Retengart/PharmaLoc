import folium
from folium.plugins import HeatMap
from folium import LayerControl, FeatureGroup

# Бэкенд без GUI для избежания конфликтов с многопоточностью
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from . import config

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_base_map(roi_params):
    """Базовая карта"""
    m = folium.Map(
        location=[roi_params['center_lat'], roi_params['center_lon']],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    return m

def create_potential_map(h3_grid, roi_params, competitors=None, recommendations=None, filename='potential_map.html'):
    """Создает карту потенциала с несколькими слоями и легендой."""
    m = create_base_map(roi_params)
    
    heat_data = h3_grid[['center_lat', 'center_lon', 'potential_score']].dropna().values.tolist()
    hm_layer = FeatureGroup(name='Тепловая карта потенциала', show=True)
    HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(hm_layer)
    hm_layer.add_to(m)
    
    def get_color(score):
        if score > 0.8:
            return '#d73027'  # Red (High)
        elif score > 0.6:
            return '#fc8d59'
        elif score > 0.4:
            return '#fee08b'
        elif score > 0.2:
            return '#d9ef8b'
        else:
            return '#91cf60'
        
    def get_color_green_high(score):
        if score > 0.8:
            return '#1a9850'  # Dark Green
        elif score > 0.6:
            return '#91cf60'
        elif score > 0.4:
            return '#fee08b'
        elif score > 0.2:
            return '#fc8d59'
        else:
            return '#d73027'

    grid_layer = FeatureGroup(name='Сетка H3 (Детализация)', show=False)
    
    folium.GeoJson(
        h3_grid.to_json(),
        style_function=lambda feature: {
            'fillColor': get_color_green_high(feature['properties'].get('potential_score', 0)),
            'color': 'gray',
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['h3_cell', 'potential_score', 'has_pharmacy', 'cluster'],
            aliases=['H3 ID:', 'Потенциал:', 'Есть аптека:', 'Кластер:'],
            localize=True
        )
    ).add_to(grid_layer)
    grid_layer.add_to(m)
    
    if competitors is not None and not competitors.empty:
        comp_layer = FeatureGroup(name='Конкуренты (Аптеки)', show=True)
        for idx, row in competitors.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x] if row.geometry.geom_type == 'Point' else [row.geometry.centroid.y, row.geometry.centroid.x],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup='Конкурент'
            ).add_to(comp_layer)
        comp_layer.add_to(m)
        
    rec_coords = []
    if recommendations:
        rec_layer = FeatureGroup(name='Рекомендованные точки', show=True)
        for rec in recommendations:
            html_parts = [
                "<div style='font-family: Arial, sans-serif; width: 350px; max-height: 600px; overflow-y: auto;'>",
                f"<h3 style='color: #1a9850; margin-top: 0;'>🏆 Ранг #{rec['rank']}</h3>",
                "<hr style='margin: 10px 0;'>",
                f"<p><strong>📍 Координаты:</strong><br>{rec['lat']:.6f}, {rec['lon']:.6f}</p>",
                f"<p><strong>🔑 H3 ячейка:</strong><br><code>{rec['h3_cell']}</code></p>",
            ]
            
            if rec.get('cluster', -1) >= 0:
                html_parts.append(f"<p><strong>📦 Кластер:</strong> {rec['cluster']}</p>")
            
            html_parts.extend([
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>💎 Потенциал</h4>",
                f"<p><strong>Интегральный:</strong> {rec['potential_score']:.4f}</p>",
            ])
            
            if 'ml_prediction_score' in rec:
                html_parts.append(f"<p><strong>ML предсказание:</strong> {rec['ml_prediction_score']:.4f}</p>")
            
            html_parts.extend([
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>⚔️ Конкуренция</h4>",
                f"<p><strong>Ближайший конкурент:</strong> {rec['nearest_competitor_m']:.1f} м</p>",
                f"<p><strong>Аптек в 500м:</strong> {rec['pharmacy_count_500m']}</p>",
                f"<p><strong>Аптек в 1000м:</strong> {rec['pharmacy_count_1000m']}</p>",
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>🏥 Медицина</h4>",
            ])
            
            if rec.get('medical_nearest_distance_m', -1) > 0:
                html_parts.append(f"<p><strong>Ближайшее:</strong> {rec['medical_nearest_distance_m']:.1f} м</p>")
            html_parts.append(f"<p><strong>В 500м:</strong> {rec.get('medical_count_500m', 0)} учреждений</p>")
            html_parts.append(f"<p><strong>Синергия:</strong> {rec.get('medical_synergy', 0):.3f}</p>")
            
            html_parts.extend([
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>🚇 Транспорт</h4>",
            ])
            
            if rec.get('transport_subway_nearest_dist', -1) > 0:
                html_parts.append(f"<p><strong>Метро:</strong> {rec['transport_subway_nearest_dist']:.1f} м</p>")
            
            if rec.get('transport_ground_nearest_dist', -1) > 0:
                html_parts.append(f"<p><strong>Наземный:</strong> {rec['transport_ground_nearest_dist']:.1f} м</p>")
                
            html_parts.append(f"<p><strong>Остановок (метро):</strong> {rec.get('transport_subway_count_500m', 0)}</p>")
            html_parts.append(f"<p><strong>Остановок (назем.):</strong> {rec.get('transport_ground_count_500m', 0)}</p>")
            
            html_parts.extend([
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>🏘️ Жилье</h4>",
                f"<p><strong>Жилых зданий в 500м:</strong> {rec.get('residential_count_500m', 0)}</p>",
                f"<p><strong>Покрытие:</strong> {rec.get('residential_coverage', 0):.1%}</p>",
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>🛍️ Коммерция</h4>",
                f"<p><strong>Торговых точек в 500м:</strong> {rec.get('retail_count_500m', 0)}</p>",
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>🏢 Офисы</h4>",
                f"<p><strong>Офисных зданий в 500м:</strong> {rec.get('office_count_500m', 0)}</p>",
                f"<p><strong>Покрытие:</strong> {rec.get('office_coverage', 0):.1%}</p>",
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>🅿️ Парковки</h4>",
                f"<p><strong>Парковок в 500м:</strong> {rec.get('parking_count_500m', 0)}</p>",
                "<hr style='margin: 10px 0;'>",
                "<h4 style='color: #d73027; margin: 15px 0 10px 0;'>📈 Индексы</h4>",
                f"<p><strong>Многофункциональность:</strong> {rec.get('multifunctionality_index', 0)}/5</p>",
                f"<p><strong>Плотность дорог:</strong> {rec.get('road_density', 0):.2f} км/км²</p>",
                "<hr style='margin: 10px 0;'>",
                "<p style='text-align: center; margin-top: 15px;'>",
                f"<a href='https://www.google.com/maps?q={rec['lat']},{rec['lon']}' target='_blank' ",
                "style='background-color: #1a9850; color: white; padding: 8px 15px; ",
                "text-decoration: none; border-radius: 5px; display: inline-block;'>",
                "🗺️ Открыть в Google Maps</a></p>",
                "</div>"
            ])
            
            html = ''.join(html_parts)
            
            folium.Marker(
                [rec['lat'], rec['lon']],
                popup=folium.Popup(html, max_width=400),
                icon=folium.Icon(color='green', icon='star', prefix='fa'),
                tooltip=f"Топ-{rec['rank']}: Потенциал {rec['potential_score']:.3f}"
            ).add_to(rec_layer)
            rec_coords.append([rec['lat'], rec['lon']])
            
        rec_layer.add_to(m)
        
        if rec_coords:
             m.fit_bounds(rec_coords)
        
    LayerControl().add_to(m)
    
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: auto; min-width: 150px; max-width: 250px; height: auto; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity: 0.9;
     padding: 10px; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2);">
     <b>Легенда</b><br>
     <i class="fa fa-star" style="color:green"></i> Рекомендация<br>
     <i class="fa fa-circle" style="color:red"></i> Конкурент<br>
     <br>
     <b>Потенциал:</b><br>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#1a9850; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Высокий (>0.8)
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#91cf60; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Выше среднего
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#fee08b; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Средний
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#fc8d59; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Ниже среднего
     </div>
     <div style="display: flex; align-items: center; margin-bottom: 3px;">
         <span style="background:#d73027; width:12px; height:12px; display:inline-block; margin-right: 5px;"></span> Низкий
     </div>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

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
        
        top_n = 20
        indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title("Важность признаков (Топ-20)", fontsize=16)
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.ylabel('Важность (Gini importance)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        output_path = os.path.join(config.DATA_DIR, filename)
        plt.savefig(output_path)
        print(f"График важности сохранен в {output_path}")
        plt.close()

def plot_correlation_matrix(df, features, filename='correlation_matrix.png'):
    """Построение матрицы корреляции признаков"""
    plt.figure(figsize=(14, 12))
    corr = df[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    
    plt.title('Матрица корреляции признаков', fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path)
    print(f"Матрица корреляции сохранена в {output_path}")
    plt.close()


def plot_roc_pr_curves(y_true, y_proba, model_name='Model', filename='roc_pr_curves.png'):
    """Построение ROC и PR кривых для оценки качества классификации."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='#1a9850', lw=2, 
                 label=f'{model_name} (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='#d73027', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    axes[0].fill_between(fpr, tpr, alpha=0.3, color='#1a9850')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Специфичность)')
    axes[0].set_ylabel('True Positive Rate (Чувствительность)')
    axes[0].set_title('ROC кривая', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    baseline = y_true.sum() / len(y_true)
    
    axes[1].plot(recall, precision, color='#1a9850', lw=2,
                 label=f'{model_name} (AP = {avg_precision:.3f})')
    axes[1].axhline(y=baseline, color='#d73027', lw=2, linestyle='--', 
                    label=f'Baseline (AP = {baseline:.3f})')
    axes[1].fill_between(recall, precision, alpha=0.3, color='#1a9850')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Полнота)')
    axes[1].set_ylabel('Precision (Точность)')
    axes[1].set_title('Precision-Recall кривая', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📈 ROC и PR кривые сохранены в {output_path}")
    plt.close()
    
    return roc_auc, avg_precision


def plot_confusion_matrix(y_true, y_pred, model_name='Model', filename='confusion_matrix.png'):
    """Построение матрицы ошибок (Confusion Matrix)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Нет аптеки', 'Есть аптека'])
    disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title(f'{model_name}: Абсолютные значения', fontsize=12, fontweight='bold')
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['Нет аптеки', 'Есть аптека'])
    disp2.plot(ax=axes[1], cmap='Greens', values_format='.2%')
    axes[1].set_title(f'{model_name}: Нормализованные (по классам)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Матрица ошибок (Confusion Matrix)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Confusion Matrix сохранена в {output_path}")
    plt.close()
    
    return cm


def plot_models_comparison(results_dict, filename='models_comparison.png'):
    """Сравнительный график метрик разных моделей."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    
    models = [k for k in results_dict.keys() if not k.startswith('_')]
    
    if not models:
        print("⚠️ Нет моделей для сравнения")
        return
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']
    
    for i, model in enumerate(models):
        values = []
        for m in metrics:
            if m == 'f1':
                values.append(results_dict[model].get('f1', results_dict[model]['metrics'].get('f1', 0)))
            else:
                values.append(results_dict[model]['metrics'].get(m, 0))
        
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)], alpha=0.8)
        
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Значение метрики')
    ax.set_title('Сравнение моделей по метрикам качества', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Сравнение моделей сохранено в {output_path}")
    plt.close()


def plot_feature_distributions(df, features, n_cols=4, filename='feature_distributions.png'):
    """Построение гистограмм распределений признаков."""
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        if feature in df.columns:
            data = df[feature].dropna()
            data = data[~np.isinf(data)]
            
            axes[i].hist(data, bins=30, color='#1a9850', alpha=0.7, edgecolor='black', linewidth=0.5)
            axes[i].set_title(feature, fontsize=10, fontweight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Частота')
            
            mean_val = data.mean()
            median_val = data.median()
            axes[i].axvline(mean_val, color='#d73027', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='#4575b4', linestyle=':', label=f'Median: {median_val:.2f}')
            axes[i].legend(fontsize=7, loc='upper right')
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Распределения признаков', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Распределения признаков сохранены в {output_path}")
    plt.close()


def plot_target_distribution(df, target_col='has_pharmacy', filename='target_distribution.png'):
    """Визуализация распределения целевой переменной."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    counts = df[target_col].value_counts()
    labels = ['Нет аптеки (0)', 'Есть аптека (1)']
    colors = ['#91cf60', '#d73027']
    
    axes[0].pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors,
                explode=(0, 0.05), shadow=True, startangle=90)
    axes[0].set_title('Соотношение классов', fontsize=12, fontweight='bold')
    
    bars = axes[1].bar(labels, counts.values, color=colors, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('Количество ячеек')
    axes[1].set_title('Абсолютные значения по классам', fontsize=12, fontweight='bold')
    
    for bar, count in zip(bars, counts.values):
        axes[1].annotate(f'{count:,}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')
    
    imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    fig.text(0.5, 0.01, f'Коэффициент дисбаланса: {imbalance_ratio:.1f}:1', 
             ha='center', fontsize=11, style='italic')
    
    plt.suptitle('Распределение целевой переменной', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Распределение целевой переменной сохранено в {output_path}")
    plt.close()


def plot_feature_boxplots(df, features, target_col='has_pharmacy', n_cols=4, filename='feature_boxplots.png'):
    """Построение box plots для сравнения распределений по классам."""
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, feature in enumerate(features):
        if feature in df.columns and target_col in df.columns:
            data = df[[feature, target_col]].copy()
            data = data[~np.isinf(data[feature])]
            
            sns.boxplot(x=target_col, y=feature, data=data, ax=axes[i],
                       hue=target_col, palette={0: '#91cf60', 1: '#d73027'}, legend=False)
            axes[i].set_title(feature, fontsize=10, fontweight='bold')
            axes[i].set_xlabel('Класс')
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(['Нет аптеки', 'Есть аптека'])
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Сравнение признаков по классам (Box Plots)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Box plots сохранены в {output_path}")
    plt.close()


def plot_cluster_tsne(X, cluster_labels, filename='cluster_tsne.png', perplexity=30):
    """Визуализация кластеров с помощью t-SNE."""
    print("  🔄 Вычисление t-SNE проекции...")
    
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X) - 1), 
                random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        mask = cluster_labels == cluster
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color], 
                  label=f'Кластер {cluster}', alpha=0.6, s=30)
    
    ax.set_xlabel('t-SNE компонента 1')
    ax.set_ylabel('t-SNE компонента 2')
    ax.set_title('Визуализация кластеров (t-SNE)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 t-SNE визуализация сохранена в {output_path}")
    plt.close()


def plot_cluster_pca(X, cluster_labels, filename='cluster_pca.png'):
    """Визуализация кластеров с помощью PCA."""
    print("  🔄 Вычисление PCA проекции...")
    
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        mask = cluster_labels == cluster
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color],
                       label=f'Кластер {cluster}', alpha=0.6, s=30)
    
    axes[0].set_xlabel(f'PC1 ({explained_var[0]:.1%} дисперсии)')
    axes[0].set_ylabel(f'PC2 ({explained_var[1]:.1%} дисперсии)')
    axes[0].set_title('Визуализация кластеров (PCA)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = min(10, len(pca.explained_variance_ratio_))
    
    pca_full = PCA(n_components=min(20, X_scaled.shape[1]))
    pca_full.fit(X_scaled)
    
    axes[1].bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                pca_full.explained_variance_ratio_, color='#1a9850', alpha=0.7,
                label='Индивидуальная')
    axes[1].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                 np.cumsum(pca_full.explained_variance_ratio_), 'o-', color='#d73027',
                 label='Кумулятивная')
    axes[1].axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% порог')
    axes[1].set_xlabel('Номер компоненты')
    axes[1].set_ylabel('Объясненная дисперсия')
    axes[1].set_title('Объясненная дисперсия (PCA)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='center right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 PCA визуализация сохранена в {output_path}")
    plt.close()
    
    return explained_var


def plot_cluster_profiles(df, cluster_col='cluster', features=None, filename='cluster_profiles.png'):
    """Построение профилей кластеров."""
    if features is None:
        features = [col for col in df.columns if any(x in col for x in 
                    ['density_500m', 'coverage', 'nearest_distance', 'synergy', 'index'])]
        features = features[:10]
    
    if not features:
        print("⚠️ Нет признаков для построения профилей кластеров")
        return
    
    cluster_means = df.groupby(cluster_col)[features].mean()
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cluster_means_normalized = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    n_clusters = len(cluster_means_normalized)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(features))
    width = 0.8 / n_clusters
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    for i, (cluster, row) in enumerate(cluster_means_normalized.iterrows()):
        bars = ax.bar(x + i * width, row.values, width, label=f'Кластер {cluster}', 
                     color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Нормализованное значение')
    ax.set_title('Профили кластеров', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_clusters - 1) / 2)
    ax.set_xticklabels([f.replace('_', '\n') for f in features], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(config.DATA_DIR, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Профили кластеров сохранены в {output_path}")
    plt.close()
    
    return cluster_means
