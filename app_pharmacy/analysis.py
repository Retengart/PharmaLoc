import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from . import config

def calculate_potential(h3_grid, weights=config.POTENTIAL_WEIGHTS, use_rule_based=True):
    """
    Расчет интегрального потенциала.
    
    Args:
        h3_grid: GeoDataFrame с признаками
        weights: Веса для rule-based оценки
        use_rule_based: Использовать ли rule-based оценку или только ML предсказания
    
    Returns:
        GeoDataFrame с добавленным столбцом potential_score
    """
    # ВАЛИДАЦИЯ входных данных
    if h3_grid is None or h3_grid.empty:
        raise ValueError("h3_grid не может быть пустым")
    
    if not isinstance(weights, dict):
        raise TypeError("weights должен быть словарем")
    
    def normalize(series):
        """Нормализация признака к [0, 1]"""
        if series is None or len(series) == 0:
            return pd.Series(0, index=h3_grid.index)
        if series.max() == series.min():
            return pd.Series(0, index=h3_grid.index)
        return (series - series.min()) / (series.max() - series.min())

    w_med = weights.get('medical_synergy', 0)
    w_subway = weights.get('transport_subway_accessibility', 0)
    w_ground = weights.get('transport_ground_accessibility', 0)
    w_res = weights.get('residential_density', 0)
    w_comm = weights.get('commercial_activity', 0)
    w_off = weights.get('office_density', 0)
    w_park = weights.get('parking_availability', 0)
    w_ped = weights.get('pedestrian_accessibility', 0)
    w_comp = weights.get('competition_penalty', 0)

    if 'medical_synergy' in h3_grid.columns:
        val_med = h3_grid['medical_synergy']
    else:
        val_med = pd.Series(0, index=h3_grid.index)
        
    if 'transport_subway_density_500m' in h3_grid.columns:
        val_subway = normalize(h3_grid['transport_subway_density_500m'])
    else:
        val_subway = pd.Series(0, index=h3_grid.index)
        
    if 'transport_ground_density_500m' in h3_grid.columns:
        val_ground = normalize(h3_grid['transport_ground_density_500m'])
    else:
        val_ground = pd.Series(0, index=h3_grid.index)
        
    if 'residential_coverage' in h3_grid.columns:
        val_res = h3_grid['residential_coverage']
    elif 'residential_density_500m' in h3_grid.columns:
        val_res = normalize(h3_grid['residential_density_500m'])
    else:
        val_res = pd.Series(0, index=h3_grid.index)
        
    if 'retail_density_500m' in h3_grid.columns:
        val_comm = normalize(h3_grid['retail_density_500m'])
    else:
        val_comm = pd.Series(0, index=h3_grid.index)
        
    if 'office_density_500m' in h3_grid.columns:
        val_off = normalize(h3_grid['office_density_500m'])
    else:
        val_off = pd.Series(0, index=h3_grid.index)

    if 'parking_density_500m' in h3_grid.columns:
        val_park = normalize(h3_grid['parking_density_500m'])
    else:
        val_park = pd.Series(0, index=h3_grid.index)

    if 'pedestrian_density_500m' in h3_grid.columns:
        val_ped = normalize(h3_grid['pedestrian_density_500m'])
    else:
        val_ped = pd.Series(0, index=h3_grid.index)
        
    if 'pharmacy_density_500m' in h3_grid.columns:
        val_comp = normalize(h3_grid['pharmacy_density_500m'])
    else:
        val_comp = pd.Series(0, index=h3_grid.index)
        
    if 'competitor_chain_count_500m' in h3_grid.columns:
        val_comp_chain = normalize(h3_grid['competitor_chain_count_500m'])
        val_comp = val_comp * 0.7 + val_comp_chain * 0.3
        
    score = (
        w_med * val_med +
        w_subway * val_subway +
        w_ground * val_ground +
        w_res * val_res +
        w_comm * val_comm +
        w_off * val_off +
        w_park * val_park +
        w_ped * val_ped +
        w_comp * val_comp
    )
    
    # ВАЛИДАЦИЯ: Проверка на NaN в score
    if score.isna().any():
        print("⚠️ Обнаружены NaN в rule-based оценке, заменяем на 0")
        score = score.fillna(0)
    
    # Смешивание rule-based и ML оценок
    if 'prediction_score' in h3_grid.columns:
        # ВАЛИДАЦИЯ: Проверка на NaN в prediction_score
        if h3_grid['prediction_score'].isna().any():
            print("⚠️ Обнаружены NaN в ML предсказаниях, заменяем на медиану")
            median_pred = h3_grid['prediction_score'].median()
            h3_grid['prediction_score'] = h3_grid['prediction_score'].fillna(median_pred if np.isfinite(median_pred) else 0)
        
        if use_rule_based:
            # Используем смешанный подход
            rule_w = config.POTENTIAL_BLEND['rule_weight']
            ml_w = config.POTENTIAL_BLEND['ml_weight']
            
            # ВАЛИДАЦИЯ: Проверка сумм весов (должна быть ~1.0)
            total_weight = rule_w + ml_w
            if abs(total_weight - 1.0) > 0.01:
                print(f"⚠️ Сумма весов rule-based и ML не равна 1.0 ({total_weight:.2f}), нормализуем")
                rule_w = rule_w / total_weight
                ml_w = ml_w / total_weight
            
            h3_grid['potential_score'] = rule_w * score + ml_w * h3_grid['prediction_score']
            h3_grid['rule_based_score'] = score  # Сохраняем для анализа
        else:
            # Используем только ML предсказания (более честный подход)
            print("   ⚠️ Используется только ML предсказание (rule-based отключен)")
            h3_grid['potential_score'] = h3_grid['prediction_score']
            h3_grid['rule_based_score'] = score  # Сохраняем для сравнения
    else:
        # Если ML модель не обучена, используем только rule-based
        h3_grid['potential_score'] = score
        if not use_rule_based:
            print("   ⚠️ ML модель не найдена, используется rule-based оценка")
    
    # ВАЛИДАЦИЯ: Финальная проверка на NaN в potential_score
    if h3_grid['potential_score'].isna().any():
        print("⚠️ Обнаружены NaN в potential_score после расчета, заменяем на 0")
        h3_grid['potential_score'] = h3_grid['potential_score'].fillna(0)
        
    return h3_grid

def get_recommendations(h3_grid, top_n=10, min_distance_to_competitor=300):
    """Формирование рекомендаций с расширенными метриками."""
    candidates = h3_grid[h3_grid['has_pharmacy'] == 0].copy()
    
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
            'lat': round(row['center_lat'], 6),
            'lon': round(row['center_lon'], 6),
            'potential_score': round(row['potential_score'], 4),
            'cluster': int(row.get('cluster', -1)),
            'nearest_competitor_m': round(row.get('pharmacy_nearest_distance', -1), 1),
        }
        
        if 'prediction_score' in row:
            rec['ml_prediction_score'] = round(row['prediction_score'], 4)
        
        rec['pharmacy_count_500m'] = int(row.get('pharmacy_count_500m', 0))
        rec['pharmacy_count_1000m'] = int(row.get('pharmacy_count_1000m', 0))
        rec['pharmacy_density_500m'] = round(row.get('pharmacy_density_500m', 0), 2)
        
        rec['medical_nearest_distance_m'] = round(row.get('medical_nearest_distance', -1), 1)
        rec['medical_count_500m'] = int(row.get('medical_count_500m', 0))
        rec['medical_synergy'] = round(row.get('medical_synergy', 0), 3)
        
        rec['transport_subway_nearest_dist'] = round(row.get('transport_subway_nearest_distance', -1), 1)
        rec['transport_subway_count_500m'] = int(row.get('transport_subway_count_500m', 0))
        
        rec['transport_ground_nearest_dist'] = round(row.get('transport_ground_nearest_distance', -1), 1)
        rec['transport_ground_count_500m'] = int(row.get('transport_ground_count_500m', 0))
        
        rec['residential_nearest_distance_m'] = round(row.get('residential_nearest_distance', -1), 1)
        rec['residential_count_500m'] = int(row.get('residential_count_500m', 0))
        rec['residential_coverage'] = round(row.get('residential_coverage', 0), 3)
        rec['residential_density_500m'] = round(row.get('residential_density_500m', 0), 2)
        
        rec['retail_nearest_distance_m'] = round(row.get('retail_nearest_distance', -1), 1)
        rec['retail_count_500m'] = int(row.get('retail_count_500m', 0))
        rec['retail_density_500m'] = round(row.get('retail_density_500m', 0), 2)
        
        rec['office_nearest_distance_m'] = round(row.get('office_nearest_distance', -1), 1)
        rec['office_count_500m'] = int(row.get('office_count_500m', 0))
        rec['office_coverage'] = round(row.get('office_coverage', 0), 3)
        rec['office_density_500m'] = round(row.get('office_density_500m', 0), 2)
        
        rec['parking_nearest_distance_m'] = round(row.get('parking_nearest_distance', -1), 1)
        rec['parking_count_500m'] = int(row.get('parking_count_500m', 0))
        rec['parking_density_500m'] = round(row.get('parking_density_500m', 0), 2)
        
        rec['pedestrian_count_500m'] = int(row.get('pedestrian_count_500m', 0))
        rec['pedestrian_density_500m'] = round(row.get('pedestrian_density_500m', 0), 2)
        
        rec['road_density'] = round(row.get('road_density', 0), 2)
        
        rec['multifunctionality_index'] = int(row.get('multifunctionality_index', 0))
        
        recommendations.append(rec)
        rank += 1
        
    return top_candidates, recommendations

def print_detailed_recommendations(recommendations, top_n=10):
    """Детальный вывод рекомендаций в консоль."""
    print("\n" + "="*80)
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ РЕКОМЕНДОВАННЫХ ЛОКАЦИЙ")
    print("="*80)
    
    for rec in recommendations[:top_n]:
        print(f"\n{'─'*80}")
        print(f"🏆 РАНГ #{rec['rank']}")
        print(f"{'─'*80}")
        
        print(f"📍 Координаты: {rec['lat']:.6f}, {rec['lon']:.6f}")
        print(f"🔑 H3 ячейка: {rec['h3_cell']}")
        if rec.get('cluster', -1) >= 0:
            print(f"📦 Кластер: {rec['cluster']}")
        
        print("\n💎 ПОТЕНЦИАЛ ЛОКАЦИИ:")
        print(f"   Интегральный потенциал: {rec['potential_score']:.4f}")
        if 'ml_prediction_score' in rec:
            print(f"   ML предсказание: {rec['ml_prediction_score']:.4f}")
        
        print("\n⚔️  КОНКУРЕНЦИЯ:")
        print(f"   Ближайший конкурент: {rec['nearest_competitor_m']:.1f} м")
        print(f"   Аптек в радиусе 500м: {rec['pharmacy_count_500m']}")
        print(f"   Аптек в радиусе 1000м: {rec['pharmacy_count_1000m']}")
        print(f"   Плотность конкурентов (500м): {rec['pharmacy_density_500m']:.2f} аптек/км²")
        
        print("\n🏥 МЕДИЦИНСКАЯ ИНФРАСТРУКТУРА:")
        if rec.get('medical_nearest_distance_m', -1) > 0:
            print(f"   Ближайшее мед. учреждение: {rec['medical_nearest_distance_m']:.1f} м")
        else:
            print("   Ближайшее мед. учреждение: не найдено")
        print(f"   Мед. учреждений в 500м: {rec.get('medical_count_500m', 0)}")
        print(f"   Медицинская синергия: {rec.get('medical_synergy', 0):.3f}")
        
        print("\n🚇 ТРАНСПОРТНАЯ ДОСТУПНОСТЬ:")
        if rec.get('transport_subway_nearest_dist', -1) > 0:
             print(f"   Метро: {rec['transport_subway_nearest_dist']:.1f} м (кол-во в 500м: {rec['transport_subway_count_500m']})")
        else:
             print("   Метро: не найдено поблизости")
             
        if rec.get('transport_ground_nearest_dist', -1) > 0:
             print(f"   Наземный транспорт: {rec['transport_ground_nearest_dist']:.1f} м (кол-во в 500м: {rec['transport_ground_count_500m']})")
        
        print("\n🏘️  ЖИЛАЯ ЗАСТРОЙКА:")
        if rec.get('residential_nearest_distance_m', -1) > 0:
            print(f"   Ближайшее жилое здание: {rec['residential_nearest_distance_m']:.1f} м")
        print(f"   Жилых зданий в 500м: {rec.get('residential_count_500m', 0)}")
        print(f"   Покрытие жильем: {rec.get('residential_coverage', 0):.1%}")
        print(f"   Плотность жилья: {rec.get('residential_density_500m', 0):.2f} зданий/км²")
        
        print("\n🛍️  КОММЕРЧЕСКАЯ АКТИВНОСТЬ:")
        if rec.get('retail_nearest_distance_m', -1) > 0:
            print(f"   Ближайший магазин/кафе: {rec['retail_nearest_distance_m']:.1f} м")
        print(f"   Торговых точек в 500м: {rec.get('retail_count_500m', 0)}")
        print(f"   Плотность торговли: {rec.get('retail_density_500m', 0):.2f} точек/км²")
        
        print("\n🏢 ОФИСНАЯ ЗАСТРОЙКА:")
        if rec.get('office_nearest_distance_m', -1) > 0:
            print(f"   Ближайшее офисное здание: {rec['office_nearest_distance_m']:.1f} м")
        print(f"   Офисных зданий в 500м: {rec.get('office_count_500m', 0)}")
        print(f"   Покрытие офисами: {rec.get('office_coverage', 0):.1%}")
        print(f"   Плотность офисов: {rec.get('office_density_500m', 0):.2f} зданий/км²")
        
        print("\n🅿️  ПАРКОВКИ:")
        if rec.get('parking_nearest_distance_m', -1) > 0:
            print(f"   Ближайшая парковка: {rec['parking_nearest_distance_m']:.1f} м")
        print(f"   Парковок в 500м: {rec.get('parking_count_500m', 0)}")
        print(f"   Плотность парковок: {rec.get('parking_density_500m', 0):.2f} парковок/км²")
        
        print("\n🚶 ПЕШЕХОДНАЯ ДОСТУПНОСТЬ:")
        print(f"   Пешеходных зон в 500м: {rec.get('pedestrian_count_500m', 0)}")
        print(f"   Плотность пешеходных зон: {rec.get('pedestrian_density_500m', 0):.2f} зон/км²")
        
        print("\n🛣️  ДОРОЖНАЯ СЕТЬ:")
        print(f"   Плотность дорог: {rec.get('road_density', 0):.2f} км/км²")
        
        print("\n📈 ДОПОЛНИТЕЛЬНЫЕ ИНДЕКСЫ:")
        print(f"   Индекс многофункциональности: {rec.get('multifunctionality_index', 0)}/5")
        
        print(f"\n🗺️  Ссылка на карту: https://www.google.com/maps?q={rec['lat']},{rec['lon']}")
    
    print(f"\n{'='*80}")
    print(f"✅ Всего проанализировано локаций: {len(recommendations)}")
    print(f"{'='*80}\n")

def save_detailed_report(recommendations, roi_params):
    """Сохранение детального отчета в JSON и CSV форматах."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_json = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'roi_name': roi_params.get('place_name', 'Unknown'),
            'total_recommendations': len(recommendations),
            'analysis_parameters': {
                'min_distance_to_competitor': 300,
                'h3_resolution': config.H3_RESOLUTION
            }
        },
        'recommendations': recommendations
    }
    
    json_path = os.path.join(config.DATA_DIR, f'detailed_report_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)
    print(f"📄 Детальный JSON отчет сохранен: {json_path}")
    
    csv_path = os.path.join(config.DATA_DIR, f'detailed_report_{timestamp}.csv')
    df_report = pd.DataFrame(recommendations)
    df_report.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"📊 Детальный CSV отчет сохранен: {csv_path}")
    
    return json_path, csv_path


def describe_clusters(h3_grid, cluster_col='cluster', feature_cols=None):
    """Автоматическое описание профилей кластеров."""
    if cluster_col not in h3_grid.columns:
        print("⚠️ Колонка с кластерами не найдена")
        return None, {}
    
    if feature_cols is None:
        exclude_cols = ['h3_cell', 'geometry', 'center_lat', 'center_lon', 'has_pharmacy', 
                       'potential_score', 'prediction_score', cluster_col]
        feature_cols = [col for col in h3_grid.columns 
                       if col not in exclude_cols and h3_grid[col].dtype in ['float64', 'int64']]
    
    key_features = {
        'residential_density_500m': 'Жилая застройка',
        'residential_coverage': 'Покрытие жильём',
        'transport_subway_count_500m': 'Метро',
        'transport_ground_count_500m': 'Наземный транспорт',
        'medical_count_500m': 'Медицинские учреждения',
        'medical_synergy': 'Синергия с медициной',
        'retail_density_500m': 'Торговля',
        'office_density_500m': 'Офисы',
        'parking_density_500m': 'Парковки',
        'pharmacy_count_500m': 'Конкуренция (аптеки)',
        'multifunctionality_index': 'Многофункциональность',
        'road_density': 'Дорожная сеть'
    }
    
    available_key_features = {k: v for k, v in key_features.items() if k in feature_cols}
    
    cluster_means = h3_grid.groupby(cluster_col)[feature_cols].mean()
    overall_means = h3_grid[feature_cols].mean()
    overall_stds = h3_grid[feature_cols].std()
    
    cluster_zscores = (cluster_means - overall_means) / overall_stds
    
    cluster_descriptions = {}
    
    print("\n" + "="*80)
    print("📊 ПРОФИЛИ КЛАСТЕРОВ")
    print("="*80)
    
    for cluster in sorted(h3_grid[cluster_col].unique()):
        cluster_size = (h3_grid[cluster_col] == cluster).sum()
        cluster_pct = cluster_size / len(h3_grid) * 100
        
        print(f"\n{'─'*80}")
        print(f"📦 КЛАСТЕР {cluster} ({cluster_size} ячеек, {cluster_pct:.1f}%)")
        print(f"{'─'*80}")
        
        description_parts = []
        
        for feature, name in available_key_features.items():
            zscore = cluster_zscores.loc[cluster, feature]
            mean_val = cluster_means.loc[cluster, feature]
            
            if zscore > 1.5:
                level = "очень высокий"
                emoji = "🔺"
            elif zscore > 0.5:
                level = "выше среднего"
                emoji = "▲"
            elif zscore < -1.5:
                level = "очень низкий"
                emoji = "🔻"
            elif zscore < -0.5:
                level = "ниже среднего"
                emoji = "▼"
            else:
                level = "средний"
                emoji = "●"
            
            print(f"  {emoji} {name}: {level} (z={zscore:.2f}, avg={mean_val:.2f})")
            
            if abs(zscore) > 0.5:
                description_parts.append(f"{level} {name.lower()}")
        
        if description_parts:
            description = f"Кластер {cluster}: " + ", ".join(description_parts[:5])
        else:
            description = f"Кластер {cluster}: типичная территория без выраженных особенностей"
        
        cluster_descriptions[cluster] = description
        
        if 'has_pharmacy' in h3_grid.columns:
            pharmacy_rate = h3_grid[h3_grid[cluster_col] == cluster]['has_pharmacy'].mean() * 100
            print(f"\n  🏥 Доля ячеек с аптеками: {pharmacy_rate:.1f}%")
        
        if 'potential_score' in h3_grid.columns:
            avg_potential = h3_grid[h3_grid[cluster_col] == cluster]['potential_score'].mean()
            print(f"  💎 Средний потенциал: {avg_potential:.3f}")
    
    print(f"\n{'='*80}")
    
    summary_data = []
    for cluster in sorted(h3_grid[cluster_col].unique()):
        row = {
            'Кластер': cluster,
            'Размер': (h3_grid[cluster_col] == cluster).sum(),
            'Доля (%)': (h3_grid[cluster_col] == cluster).sum() / len(h3_grid) * 100
        }
        
        if 'has_pharmacy' in h3_grid.columns:
            row['Аптеки (%)'] = h3_grid[h3_grid[cluster_col] == cluster]['has_pharmacy'].mean() * 100
        
        if 'potential_score' in h3_grid.columns:
            row['Ср. потенциал'] = h3_grid[h3_grid[cluster_col] == cluster]['potential_score'].mean()
        
        row['Описание'] = cluster_descriptions[cluster]
        summary_data.append(row)
    
    cluster_profiles = pd.DataFrame(summary_data)
    
    output_path = os.path.join(config.DATA_DIR, 'cluster_profiles.csv')
    cluster_profiles.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n📊 Профили кластеров сохранены в {output_path}")
    
    return cluster_profiles, cluster_descriptions


def get_cluster_recommendations(h3_grid, cluster_col='cluster'):
    """Генерирует рекомендации по каждому кластеру."""
    if cluster_col not in h3_grid.columns:
        return {}
    
    recommendations = {}
    
    for cluster in sorted(h3_grid[cluster_col].unique()):
        cluster_data = h3_grid[h3_grid[cluster_col] == cluster]
        
        rec = {
            'cluster_id': cluster,
            'size': len(cluster_data),
            'recommendations': []
        }
        
        if 'potential_score' in cluster_data.columns:
            avg_potential = cluster_data['potential_score'].mean()
            
            if avg_potential > 0.5:
                rec['recommendations'].append("✅ Высокий потенциал — приоритетный кластер для размещения")
            elif avg_potential > 0.3:
                rec['recommendations'].append("⚡ Средний потенциал — требуется детальный анализ конкретных локаций")
            else:
                rec['recommendations'].append("⚠️ Низкий потенциал — не рекомендуется для размещения")
        
        if 'pharmacy_density_500m' in cluster_data.columns:
            avg_competition = cluster_data['pharmacy_density_500m'].mean()
            if avg_competition > 5:
                rec['recommendations'].append("🔴 Высокая конкуренция — рынок насыщен")
            elif avg_competition < 1:
                rec['recommendations'].append("🟢 Низкая конкуренция — есть возможности для входа")
        
        if 'medical_synergy' in cluster_data.columns:
            avg_synergy = cluster_data['medical_synergy'].mean()
            if avg_synergy > 0.5:
                rec['recommendations'].append("🏥 Высокая синергия с медициной — идеально для аптек")
        
        transport_cols = [c for c in cluster_data.columns if 'transport' in c and 'count_500m' in c]
        if transport_cols:
            avg_transport = cluster_data[transport_cols].sum(axis=1).mean()
            if avg_transport > 3:
                rec['recommendations'].append("🚇 Хорошая транспортная доступность")
        
        recommendations[cluster] = rec
    
    return recommendations


def generate_cluster_report(h3_grid, cluster_col='cluster'):
    """Генерирует полный отчет по кластерам."""
    profiles, descriptions = describe_clusters(h3_grid, cluster_col)
    recommendations = get_cluster_recommendations(h3_grid, cluster_col)
    
    descriptions_clean = {int(k): v for k, v in descriptions.items()}
    recommendations_clean = {int(k): v for k, v in recommendations.items()}
    
    report = {
        'profiles': profiles.to_dict('records') if profiles is not None else [],
        'descriptions': descriptions_clean,
        'recommendations': recommendations_clean,
        'generated_at': datetime.now().isoformat()
    }
    
    output_path = os.path.join(config.DATA_DIR, 'cluster_analysis_report.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 Отчет по кластерам сохранен в {output_path}")
    
    return report
