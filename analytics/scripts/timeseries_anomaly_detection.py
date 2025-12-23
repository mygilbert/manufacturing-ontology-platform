"""
10분 단위 집계 기반 시계열 이상감지
- 1초 데이터를 10분(600초) 단위로 집계
- 다양한 통계 지표 추출
- 시계열 이상감지 알고리즘 적용
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
import time

print("=" * 70)
print("10분 단위 집계 기반 시계열 이상감지")
print("=" * 70)

# =============================================================================
# 1. 데이터 로딩
# =============================================================================
print("\n[1/5] 데이터 로딩...")
data_path = r"D:\24346_ESWA_PKG_11_1_AL_FORMING_PRESS_SVM.csv"
df = pd.read_csv(data_path)

pv_cols = ['SVM_Z_CURRENT', 'SVM_Z_EFFECTIVE_LOAD_RATIO',
           'SVM_Z_PEAK_LOAD_RATIO', 'SVM_Z_POSITION']

# 인덱스를 시간처럼 사용 (1초 단위)
df['time_idx'] = range(len(df))
df['window_10min'] = df['time_idx'] // 600  # 10분 = 600초

print(f"  원본 데이터: {len(df):,} 행 (약 {len(df)//3600:.1f} 시간)")
print(f"  10분 윈도우 수: {df['window_10min'].nunique():,} 개")

# =============================================================================
# 2. 10분 단위 집계 (Feature Engineering)
# =============================================================================
print("\n[2/5] 10분 단위 집계 및 특성 추출...")

def aggregate_window(group):
    """10분 윈도우에서 통계 지표 추출"""
    result = {}

    for col in pv_cols:
        data = group[col].dropna()

        if len(data) == 0:
            for stat in ['mean', 'std', 'min', 'max', 'median', 'range', 'cv', 'skew', 'kurt', 'q25', 'q75', 'iqr']:
                result[f'{col}_{stat}'] = np.nan
            continue

        # 기본 통계량
        result[f'{col}_mean'] = data.mean()
        result[f'{col}_std'] = data.std()
        result[f'{col}_min'] = data.min()
        result[f'{col}_max'] = data.max()
        result[f'{col}_median'] = data.median()
        result[f'{col}_range'] = data.max() - data.min()

        # 변동계수 (Coefficient of Variation)
        if data.mean() != 0:
            result[f'{col}_cv'] = data.std() / abs(data.mean())
        else:
            result[f'{col}_cv'] = 0

        # 왜도, 첨도
        if len(data) >= 3:
            result[f'{col}_skew'] = stats.skew(data)
            result[f'{col}_kurt'] = stats.kurtosis(data)
        else:
            result[f'{col}_skew'] = 0
            result[f'{col}_kurt'] = 0

        # 사분위수
        result[f'{col}_q25'] = data.quantile(0.25)
        result[f'{col}_q75'] = data.quantile(0.75)
        result[f'{col}_iqr'] = result[f'{col}_q75'] - result[f'{col}_q25']

    # Run 정보
    result['n_samples'] = len(group)
    result['n_runs'] = group['Run'].nunique()

    return pd.Series(result)

# 집계 수행
print("  집계 중... (잠시 기다려주세요)")
df_agg = df.groupby('window_10min').apply(aggregate_window).reset_index()

# 결측치 처리
df_agg = df_agg.dropna()
print(f"  집계 완료: {len(df_agg)} 개 윈도우 (10분 단위)")

# 변화율 특성 추가 (시계열)
print("  시계열 변화율 특성 추가 중...")
for col in pv_cols:
    # 이전 윈도우 대비 변화율
    df_agg[f'{col}_mean_diff'] = df_agg[f'{col}_mean'].diff()
    df_agg[f'{col}_mean_pct_change'] = df_agg[f'{col}_mean'].pct_change()

    # 이동 평균 대비 편차
    df_agg[f'{col}_mean_ma3'] = df_agg[f'{col}_mean'].rolling(window=3, min_periods=1).mean()
    df_agg[f'{col}_mean_ma_dev'] = df_agg[f'{col}_mean'] - df_agg[f'{col}_mean_ma3']

# 결측치 처리
df_agg = df_agg.fillna(0)

# 특성 컬럼 선택
feature_cols = [c for c in df_agg.columns if c not in ['window_10min', 'n_samples', 'n_runs']]
print(f"  총 특성 수: {len(feature_cols)} 개")

# =============================================================================
# 3. 데이터 특성 확인
# =============================================================================
print("\n[3/5] 집계 데이터 특성 확인...")

print("\n  [주요 통계량 - 10분 평균 기준]")
for col in pv_cols:
    mean_col = f'{col}_mean'
    print(f"  {col}:")
    print(f"    Mean: {df_agg[mean_col].mean():.2f}, Std: {df_agg[mean_col].std():.2f}")
    print(f"    Min: {df_agg[mean_col].min():.2f}, Max: {df_agg[mean_col].max():.2f}")

# =============================================================================
# 4. 시계열 이상감지 알고리즘 적용
# =============================================================================
print("\n[4/5] 시계열 이상감지 알고리즘 적용...")

from anomaly_detection.algorithms import (
    ZScoreDetector, CUSUMDetector, SPCDetector,
    IsolationForestDetector, LOFDetector
)

# 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(df_agg[feature_cols].values)

algorithms = {
    'Z-Score': ZScoreDetector(threshold=3.0),
    'CUSUM': CUSUMDetector(threshold=5.0, drift=0.5),
    'SPC': SPCDetector(n_sigma=3.0, use_western_electric_rules=True),
    'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
    'LOF': LOFDetector(n_neighbors=10, contamination=0.05),
}

results = {}

for name, detector in algorithms.items():
    print(f"\n  [{name}] 학습 및 예측...")

    try:
        start = time.time()
        detector.fit(X)
        y_pred = detector.predict(X)
        y_score = detector.predict_score(X)
        elapsed = time.time() - start

        n_anomalies = y_pred.sum()

        results[name] = {
            'predictions': y_pred,
            'scores': y_score,
            'n_anomalies': n_anomalies,
            'anomaly_ratio': n_anomalies / len(y_pred) * 100,
            'time': elapsed
        }

        print(f"    탐지 이상: {n_anomalies} 개 ({n_anomalies/len(y_pred)*100:.1f}%)")

    except Exception as e:
        print(f"    오류: {e}")

# 앙상블 결과
print("\n  [앙상블 분석]")
all_preds = np.stack([r['predictions'] for r in results.values()], axis=1)
ensemble_vote = (all_preds.sum(axis=1) >= 3).astype(int)  # 3개 이상 일치
n_ensemble = ensemble_vote.sum()
print(f"    3개 이상 알고리즘 일치: {n_ensemble} 개 ({n_ensemble/len(ensemble_vote)*100:.1f}%)")

df_agg['anomaly_ensemble'] = ensemble_vote
df_agg['anomaly_if'] = results['Isolation Forest']['predictions']
df_agg['anomaly_score'] = results['Isolation Forest']['scores']

# =============================================================================
# 5. 시각화
# =============================================================================
print("\n[5/5] 시각화...")

output_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(output_dir, exist_ok=True)

# 색상 설정
colors_if = np.where(df_agg['anomaly_if'] == 1, 'red', 'blue')
colors_ensemble = np.where(df_agg['anomaly_ensemble'] == 1, 'red', 'blue')

# ----- Figure 1: 10분 집계 시계열 -----
fig1, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig1.suptitle('10-Minute Aggregated Time Series - Anomaly Detection\n(Red: Anomaly, Blue: Normal)',
              fontsize=14, fontweight='bold')

for i, col in enumerate(pv_cols):
    ax = axes[i]
    mean_col = f'{col}_mean'
    std_col = f'{col}_std'

    x = df_agg['window_10min'].values
    y_mean = df_agg[mean_col].values
    y_std = df_agg[std_col].values

    # 평균 선
    ax.plot(x, y_mean, 'b-', alpha=0.5, linewidth=0.8, label='Mean')

    # 표준편차 범위 (음영)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color='blue')

    # 이상 포인트
    anomaly_mask = df_agg['anomaly_if'] == 1
    ax.scatter(x[anomaly_mask], y_mean[anomaly_mask], c='red', s=30, zorder=5, label='Anomaly')

    ax.set_ylabel(col.replace('SVM_Z_', ''))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

axes[-1].set_xlabel('Window Index (10-min)')
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'ts_10min_timeseries.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_10min_timeseries.png")

# ----- Figure 2: 이상 점수 시계열 -----
fig2, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig2.suptitle('Anomaly Score Time Series (10-min Windows)', fontsize=14, fontweight='bold')

# 이상 점수
ax = axes[0]
ax.plot(df_agg['window_10min'], df_agg['anomaly_score'], 'b-', alpha=0.7, linewidth=0.8)
ax.scatter(df_agg.loc[anomaly_mask, 'window_10min'],
           df_agg.loc[anomaly_mask, 'anomaly_score'],
           c='red', s=30, zorder=5, label='Anomaly')
threshold = np.percentile(df_agg['anomaly_score'], 95)
ax.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold (95th)')
ax.set_ylabel('Anomaly Score')
ax.legend()
ax.grid(True, alpha=0.3)

# 알고리즘별 이상 탐지 결과 (히트맵 스타일)
ax = axes[1]
algo_names = list(results.keys())
anomaly_matrix = np.stack([results[name]['predictions'] for name in algo_names], axis=0)
im = ax.imshow(anomaly_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
ax.set_yticks(range(len(algo_names)))
ax.set_yticklabels(algo_names)
ax.set_xlabel('Window Index (10-min)')
ax.set_ylabel('Algorithm')
ax.set_title('Algorithm Agreement (Red: Anomaly)')

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'ts_10min_anomaly_score.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_10min_anomaly_score.png")

# ----- Figure 3: 변화율 기반 이상 탐지 -----
fig3, axes = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle('Rate of Change Analysis (10-min Windows)', fontsize=14, fontweight='bold')

for i, col in enumerate(pv_cols):
    ax = axes[i // 2, i % 2]

    diff_col = f'{col}_mean_diff'
    x = df_agg['window_10min'].values
    y = df_agg[diff_col].values

    ax.plot(x, y, 'b-', alpha=0.5, linewidth=0.8)
    ax.scatter(x[anomaly_mask], y[anomaly_mask], c='red', s=20, zorder=5, label='Anomaly')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 3시그마 범위
    mean_diff = np.mean(y)
    std_diff = np.std(y)
    ax.axhline(y=mean_diff + 3*std_diff, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=mean_diff - 3*std_diff, color='orange', linestyle='--', alpha=0.7)

    ax.set_title(f'{col.replace("SVM_Z_", "")} - Change Rate')
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Delta')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'ts_10min_change_rate.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_10min_change_rate.png")

# ----- Figure 4: 통계 지표 비교 -----
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle('Normal vs Anomaly - Statistical Features (10-min)', fontsize=14, fontweight='bold')

normal_mask = df_agg['anomaly_if'] == 0
anomaly_mask = df_agg['anomaly_if'] == 1

features_to_compare = [
    ('SVM_Z_CURRENT_std', 'Current Std Dev'),
    ('SVM_Z_PEAK_LOAD_RATIO_cv', 'Peak Load CV'),
    ('SVM_Z_EFFECTIVE_LOAD_RATIO_range', 'Effective Load Range'),
    ('SVM_Z_POSITION_mean', 'Position Mean')
]

for i, (feat, title) in enumerate(features_to_compare):
    ax = axes[i // 2, i % 2]

    normal_data = df_agg.loc[normal_mask, feat]
    anomaly_data = df_agg.loc[anomaly_mask, feat]

    bp = ax.boxplot([normal_data, anomaly_data], labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax.set_title(title)
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig4.savefig(os.path.join(output_dir, 'ts_10min_feature_comparison.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_10min_feature_comparison.png")

# ----- Figure 5: 이상 구간 상세 분석 -----
fig5, axes = plt.subplots(2, 1, figsize=(16, 8))
fig5.suptitle('Anomaly Windows Detail Analysis', fontsize=14, fontweight='bold')

# 이상 윈도우 리스트
anomaly_windows = df_agg.loc[anomaly_mask, 'window_10min'].values
print(f"\n  이상 탐지된 윈도우 수: {len(anomaly_windows)}")

if len(anomaly_windows) > 0:
    # 연속 이상 구간 찾기
    anomaly_groups = []
    current_group = [anomaly_windows[0]]

    for i in range(1, len(anomaly_windows)):
        if anomaly_windows[i] - anomaly_windows[i-1] <= 2:  # 2개 이내면 연속
            current_group.append(anomaly_windows[i])
        else:
            anomaly_groups.append(current_group)
            current_group = [anomaly_windows[i]]
    anomaly_groups.append(current_group)

    print(f"  연속 이상 구간 수: {len(anomaly_groups)}")

    # 상위 5개 이상 구간
    ax = axes[0]
    for col in pv_cols[:2]:
        mean_col = f'{col}_mean'
        ax.plot(df_agg['window_10min'], df_agg[mean_col], label=col.replace('SVM_Z_', ''), alpha=0.7)

    # 이상 구간 음영
    for group in anomaly_groups[:10]:
        ax.axvspan(min(group)-0.5, max(group)+0.5, alpha=0.3, color='red')

    ax.set_xlabel('Window Index (10-min)')
    ax.set_ylabel('Mean Value')
    ax.set_title('Anomaly Regions (Red Shaded)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 이상 구간 크기 분포
    ax = axes[1]
    group_sizes = [len(g) for g in anomaly_groups]
    ax.hist(group_sizes, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Consecutive Anomaly Windows')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Anomaly Region Sizes (Total: {len(anomaly_groups)} regions)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig5.savefig(os.path.join(output_dir, 'ts_10min_anomaly_regions.png'), dpi=150, bbox_inches='tight')
print(f"  저장: ts_10min_anomaly_regions.png")

# =============================================================================
# 결과 요약
# =============================================================================
print("\n" + "=" * 70)
print("결과 요약")
print("=" * 70)

print(f"""
[데이터 집계]
- 원본: {len(df):,} 행 (1초 단위)
- 집계: {len(df_agg):,} 행 (10분 단위)
- 압축률: {len(df)/len(df_agg):.0f}:1

[추출된 특성]
- 기본 통계: mean, std, min, max, median, range
- 분포 특성: cv, skew, kurt, q25, q75, iqr
- 시계열 특성: diff, pct_change, ma_dev
- 총 특성 수: {len(feature_cols)}개

[이상 탐지 결과]
""")

for name, res in results.items():
    print(f"  {name}: {res['n_anomalies']}개 ({res['anomaly_ratio']:.1f}%)")

print(f"  앙상블 (3개 이상 일치): {n_ensemble}개 ({n_ensemble/len(df_agg)*100:.1f}%)")

print(f"""
[시계열 이상 패턴]
- 연속 이상 구간 수: {len(anomaly_groups)}개
- 평균 이상 구간 길이: {np.mean(group_sizes):.1f} 윈도우 ({np.mean(group_sizes)*10:.0f}분)
- 최대 이상 구간 길이: {max(group_sizes)} 윈도우 ({max(group_sizes)*10}분)

[저장된 파일]
- ts_10min_timeseries.png: 10분 집계 시계열
- ts_10min_anomaly_score.png: 이상 점수 시계열
- ts_10min_change_rate.png: 변화율 분석
- ts_10min_feature_comparison.png: 정상/이상 특성 비교
- ts_10min_anomaly_regions.png: 이상 구간 분석
""")

# 집계 데이터 저장
df_agg.to_csv(os.path.join(output_dir, 'aggregated_10min_data.csv'), index=False)
print(f"  집계 데이터 저장: aggregated_10min_data.csv")

print("\n" + "=" * 70)
print("분석 완료!")
print("=" * 70)
