"""
1분 단위 집계 - 알고리즘별 시계열 트렌드 시각화
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import StandardScaler
from scipy import stats

print("=" * 70)
print("1분 단위 집계 - 알고리즘별 시계열 트렌드")
print("=" * 70)

# =============================================================================
# 1. 데이터 로딩 및 1분 집계
# =============================================================================
print("\n[1/3] 데이터 로딩 및 1분 집계...")
data_path = r"D:\24346_ESWA_PKG_11_1_AL_FORMING_PRESS_SVM.csv"
df = pd.read_csv(data_path)

pv_cols = ['SVM_Z_CURRENT', 'SVM_Z_EFFECTIVE_LOAD_RATIO',
           'SVM_Z_PEAK_LOAD_RATIO', 'SVM_Z_POSITION']

df['time_idx'] = range(len(df))
df['window_1min'] = df['time_idx'] // 60  # 1분 = 60초

print(f"  원본 데이터: {len(df):,} 행")
print(f"  1분 윈도우 수: {df['window_1min'].nunique():,} 개")

# 1분 집계
def aggregate_1min(group):
    result = {}
    for col in pv_cols:
        data = group[col].dropna()
        if len(data) == 0:
            result[f'{col}_mean'] = np.nan
            result[f'{col}_std'] = np.nan
            result[f'{col}_min'] = np.nan
            result[f'{col}_max'] = np.nan
            result[f'{col}_range'] = np.nan
            continue
        result[f'{col}_mean'] = data.mean()
        result[f'{col}_std'] = data.std()
        result[f'{col}_min'] = data.min()
        result[f'{col}_max'] = data.max()
        result[f'{col}_range'] = data.max() - data.min()
    result['n_samples'] = len(group)
    return pd.Series(result)

print("  1분 집계 중...")
df_1min = df.groupby('window_1min').apply(aggregate_1min).reset_index()
df_1min = df_1min.dropna().reset_index(drop=True)

# 시계열 특성 추가
for col in pv_cols:
    df_1min[f'{col}_mean_diff'] = df_1min[f'{col}_mean'].diff()
    df_1min[f'{col}_mean_ma5'] = df_1min[f'{col}_mean'].rolling(5, min_periods=1).mean()

df_1min = df_1min.fillna(0)

feature_cols = [c for c in df_1min.columns if c not in ['window_1min', 'n_samples']]
print(f"  집계 완료: {len(df_1min):,} 행, {len(feature_cols)} 특성")

# =============================================================================
# 2. 알고리즘별 이상 점수 계산
# =============================================================================
print("\n[2/3] 알고리즘별 이상 점수 계산...")

from anomaly_detection.algorithms import (
    ZScoreDetector, CUSUMDetector, SPCDetector,
    IsolationForestDetector, LOFDetector
)

scaler = StandardScaler()
X = scaler.fit_transform(df_1min[feature_cols].values)

algorithms = {
    'Z-Score': ZScoreDetector(threshold=3.0),
    'CUSUM': CUSUMDetector(threshold=5.0, drift=0.5),
    'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
    'LOF': LOFDetector(n_neighbors=20, contamination=0.05),
}

for name, detector in algorithms.items():
    print(f"  {name} 계산 중...")
    detector.fit(X)
    df_1min[f'{name}_score'] = detector.predict_score(X)
    df_1min[f'{name}_pred'] = detector.predict(X)

    # 점수 정규화 (0~1 범위로)
    score_col = f'{name}_score'
    min_score = df_1min[score_col].min()
    max_score = df_1min[score_col].max()
    df_1min[f'{name}_score_norm'] = (df_1min[score_col] - min_score) / (max_score - min_score + 1e-10)

# =============================================================================
# 3. 시각화
# =============================================================================
print("\n[3/3] 시각화...")

output_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(output_dir, exist_ok=True)

# 시간 축 (분 -> 시간)
time_hours = df_1min['window_1min'].values / 60

# ----- Figure 1: 센서 데이터 + 알고리즘 점수 트렌드 -----
fig, axes = plt.subplots(6, 1, figsize=(20, 16), sharex=True)
fig.suptitle('1-Minute Aggregated Time Series with Anomaly Scores', fontsize=16, fontweight='bold')

# 센서 데이터 (상위 2개)
colors_sensor = ['#1f77b4', '#ff7f0e']
for i, col in enumerate(pv_cols[:2]):
    ax = axes[i]
    mean_col = f'{col}_mean'
    ax.plot(time_hours, df_1min[mean_col], color=colors_sensor[i], linewidth=0.5, alpha=0.8)
    ax.fill_between(time_hours,
                    df_1min[mean_col] - df_1min[f'{col}_std'],
                    df_1min[mean_col] + df_1min[f'{col}_std'],
                    alpha=0.2, color=colors_sensor[i])
    ax.set_ylabel(col.replace('SVM_Z_', ''), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{col.replace("SVM_Z_", "")} (Mean ± Std)', fontsize=11)

# 알고리즘별 이상 점수 (정규화)
algo_colors = {'Z-Score': '#2ca02c', 'CUSUM': '#d62728',
               'Isolation Forest': '#9467bd', 'LOF': '#8c564b'}

for i, (name, color) in enumerate(algo_colors.items()):
    ax = axes[i + 2]
    score_col = f'{name}_score_norm'
    pred_col = f'{name}_pred'

    ax.plot(time_hours, df_1min[score_col], color=color, linewidth=0.5, alpha=0.7, label='Score')

    # 이상 포인트 강조
    anomaly_mask = df_1min[pred_col] == 1
    ax.scatter(time_hours[anomaly_mask], df_1min.loc[anomaly_mask, score_col],
               color='red', s=5, alpha=0.8, zorder=5, label='Anomaly')

    # 임계값 라인
    threshold = df_1min.loc[df_1min[pred_col] == 1, score_col].min() if anomaly_mask.sum() > 0 else 0.95
    ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)

    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(f'{name} (Anomaly: {anomaly_mask.sum():,} / {len(df_1min):,} = {anomaly_mask.mean()*100:.1f}%)',
                 fontsize=11, color=color)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

axes[-1].set_xlabel('Time (hours)', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, 'ts_1min_algorithm_trends.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_1min_algorithm_trends.png")

# ----- Figure 2: 알고리즘 점수 비교 (동일 축) -----
fig2, axes2 = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
fig2.suptitle('Algorithm Score Comparison (1-min Windows)', fontsize=16, fontweight='bold')

# 모든 알고리즘 점수 오버레이
ax = axes2[0]
for name, color in algo_colors.items():
    score_col = f'{name}_score_norm'
    ax.plot(time_hours, df_1min[score_col], color=color, linewidth=0.5, alpha=0.7, label=name)

ax.set_ylabel('Normalized Score', fontsize=12)
ax.set_title('All Algorithm Scores Overlay', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# 앙상블 점수 (평균)
ax = axes2[1]
ensemble_score = np.mean([df_1min[f'{name}_score_norm'].values for name in algo_colors.keys()], axis=0)
ax.plot(time_hours, ensemble_score, color='black', linewidth=0.8, alpha=0.8, label='Ensemble (Mean)')
ax.fill_between(time_hours, 0, ensemble_score, alpha=0.3, color='gray')

# 상위 5% 임계값
threshold_95 = np.percentile(ensemble_score, 95)
ax.axhline(y=threshold_95, color='red', linestyle='--', label=f'95th Percentile ({threshold_95:.2f})')

# 이상 구간 표시
high_score_mask = ensemble_score > threshold_95
ax.scatter(time_hours[high_score_mask], ensemble_score[high_score_mask],
           color='red', s=10, zorder=5, label='High Score Points')

ax.set_xlabel('Time (hours)', fontsize=12)
ax.set_ylabel('Ensemble Score', fontsize=12)
ax.set_title('Ensemble Score (Mean of All Algorithms)', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'ts_1min_algorithm_comparison.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_1min_algorithm_comparison.png")

# ----- Figure 3: 센서별 + 앙상블 점수 상세 -----
fig3, axes3 = plt.subplots(5, 1, figsize=(20, 14), sharex=True)
fig3.suptitle('Sensor Values with Ensemble Anomaly Score (1-min)', fontsize=16, fontweight='bold')

sensor_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, col in enumerate(pv_cols):
    ax = axes3[i]
    mean_col = f'{col}_mean'

    # 정상/이상 구분
    ax.plot(time_hours, df_1min[mean_col], color=sensor_colors[i], linewidth=0.5, alpha=0.6)

    # 이상 구간 음영
    ax.fill_between(time_hours, df_1min[mean_col].min(), df_1min[mean_col].max(),
                    where=high_score_mask, alpha=0.3, color='red')

    ax.set_ylabel(col.replace('SVM_Z_', ''), fontsize=10)
    ax.grid(True, alpha=0.3)

# 앙상블 점수
ax = axes3[4]
ax.plot(time_hours, ensemble_score, color='black', linewidth=0.8)
ax.fill_between(time_hours, 0, ensemble_score, where=high_score_mask, alpha=0.5, color='red', label='Anomaly Region')
ax.axhline(y=threshold_95, color='orange', linestyle='--', alpha=0.8)
ax.set_ylabel('Ensemble Score', fontsize=10)
ax.set_xlabel('Time (hours)', fontsize=12)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'ts_1min_sensor_ensemble.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_1min_sensor_ensemble.png")

# ----- Figure 4: 이동평균 트렌드 -----
fig4, axes4 = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
fig4.suptitle('Moving Average Trend Analysis (5-min MA)', fontsize=16, fontweight='bold')

for i, (name, color) in enumerate(algo_colors.items()):
    ax = axes4[i]
    score_col = f'{name}_score_norm'

    # 원본 점수
    ax.plot(time_hours, df_1min[score_col], color=color, linewidth=0.3, alpha=0.3, label='Raw')

    # 5분 이동평균
    ma5 = df_1min[score_col].rolling(5, min_periods=1).mean()
    ax.plot(time_hours, ma5, color=color, linewidth=1.5, alpha=0.9, label='5-min MA')

    # 15분 이동평균
    ma15 = df_1min[score_col].rolling(15, min_periods=1).mean()
    ax.plot(time_hours, ma15, color='black', linewidth=1.5, alpha=0.7, linestyle='--', label='15-min MA')

    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(f'{name} - Moving Average Trend', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

axes4[-1].set_xlabel('Time (hours)', fontsize=12)
plt.tight_layout()
fig4.savefig(os.path.join(output_dir, 'ts_1min_ma_trend.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_1min_ma_trend.png")

# =============================================================================
# 결과 출력
# =============================================================================
print("\n" + "=" * 70)
print("결과 요약")
print("=" * 70)

print(f"""
[데이터]
- 원본: {len(df):,} 행 (1초 단위)
- 집계: {len(df_1min):,} 행 (1분 단위)
- 총 시간: {len(df_1min)/60:.1f} 시간

[알고리즘별 이상 탐지]
""")

for name in algo_colors.keys():
    pred_col = f'{name}_pred'
    n_anomaly = df_1min[pred_col].sum()
    print(f"  {name}: {n_anomaly:,}개 ({n_anomaly/len(df_1min)*100:.1f}%)")

n_ensemble = high_score_mask.sum()
print(f"  앙상블 (95th): {n_ensemble:,}개 ({n_ensemble/len(df_1min)*100:.1f}%)")

print(f"""
[저장된 파일]
- ts_1min_algorithm_trends.png: 센서 + 알고리즘별 점수 트렌드
- ts_1min_algorithm_comparison.png: 알고리즘 점수 비교
- ts_1min_sensor_ensemble.png: 센서값 + 앙상블 점수
- ts_1min_ma_trend.png: 이동평균 트렌드
""")

print("=" * 70)
print("완료!")
print("=" * 70)
