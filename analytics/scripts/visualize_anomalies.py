"""
이상 탐지 결과 시각화 스크립트
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
matplotlib.use('Agg')  # GUI 없이 저장
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("이상 탐지 결과 시각화")
print("=" * 70)

# 1. 데이터 로딩
print("\n[1/4] 데이터 로딩...")
data_path = r"D:\24346_ESWA_PKG_11_1_AL_FORMING_PRESS_SVM.csv"
df = pd.read_csv(data_path)

pv_cols = ['SVM_Z_CURRENT', 'SVM_Z_EFFECTIVE_LOAD_RATIO',
           'SVM_Z_PEAK_LOAD_RATIO', 'SVM_Z_POSITION']

df_clean = df[pv_cols + ['Run']].dropna()
print(f"  데이터 로드 완료: {len(df_clean):,} 행")

# 2. 이상 탐지
print("\n[2/4] 이상 탐지 수행...")
from anomaly_detection.algorithms import IsolationForestDetector

scaler = StandardScaler()
X = scaler.fit_transform(df_clean[pv_cols].values)

# 샘플링 (시각화용)
sample_size = 50000
np.random.seed(42)
if len(X) > sample_size:
    indices = np.random.choice(len(X), sample_size, replace=False)
    indices = np.sort(indices)  # 시간 순서 유지
else:
    indices = np.arange(len(X))

X_sample = X[indices]
df_sample = df_clean.iloc[indices].reset_index(drop=True)

# Isolation Forest
detector = IsolationForestDetector(contamination=0.05, n_estimators=100)
detector.fit(X_sample)
y_pred = detector.predict(X_sample)
y_score = detector.predict_score(X_sample)

n_anomalies = y_pred.sum()
print(f"  탐지된 이상: {n_anomalies:,} ({n_anomalies/len(y_pred)*100:.2f}%)")

# 3. 시각화
print("\n[3/4] 시각화 생성...")
output_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(output_dir, exist_ok=True)

# 색상 설정
colors = np.where(y_pred == 1, 'red', 'blue')
alpha_normal = 0.3
alpha_anomaly = 0.8

# Figure 1: 시계열 플롯 (모든 센서)
fig1, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig1.suptitle('AL FORMING PRESS SVM - Anomaly Detection Results\n(Red: Anomaly, Blue: Normal)',
              fontsize=14, fontweight='bold')

for i, col in enumerate(pv_cols):
    ax = axes[i]

    # 정상 데이터
    normal_mask = y_pred == 0
    ax.scatter(np.where(normal_mask)[0], df_sample.loc[normal_mask, col],
               c='blue', s=1, alpha=alpha_normal, label='Normal')

    # 이상 데이터
    anomaly_mask = y_pred == 1
    ax.scatter(np.where(anomaly_mask)[0], df_sample.loc[anomaly_mask, col],
               c='red', s=5, alpha=alpha_anomaly, label='Anomaly')

    ax.set_ylabel(col.replace('SVM_Z_', ''), fontsize=10)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel('Sample Index', fontsize=12)
plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'anomaly_timeseries.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_timeseries.png")

# Figure 2: 산점도 행렬 (주요 센서 쌍)
fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
fig2.suptitle('Sensor Pair Scatter Plots - Anomaly Detection\n(Red: Anomaly, Blue: Normal)',
              fontsize=14, fontweight='bold')

# Current vs Peak Load
ax = axes[0, 0]
ax.scatter(df_sample.loc[normal_mask, 'SVM_Z_CURRENT'],
           df_sample.loc[normal_mask, 'SVM_Z_PEAK_LOAD_RATIO'],
           c='blue', s=2, alpha=alpha_normal, label='Normal')
ax.scatter(df_sample.loc[anomaly_mask, 'SVM_Z_CURRENT'],
           df_sample.loc[anomaly_mask, 'SVM_Z_PEAK_LOAD_RATIO'],
           c='red', s=10, alpha=alpha_anomaly, label='Anomaly')
ax.set_xlabel('Current')
ax.set_ylabel('Peak Load Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

# Effective Load vs Peak Load
ax = axes[0, 1]
ax.scatter(df_sample.loc[normal_mask, 'SVM_Z_EFFECTIVE_LOAD_RATIO'],
           df_sample.loc[normal_mask, 'SVM_Z_PEAK_LOAD_RATIO'],
           c='blue', s=2, alpha=alpha_normal, label='Normal')
ax.scatter(df_sample.loc[anomaly_mask, 'SVM_Z_EFFECTIVE_LOAD_RATIO'],
           df_sample.loc[anomaly_mask, 'SVM_Z_PEAK_LOAD_RATIO'],
           c='red', s=10, alpha=alpha_anomaly, label='Anomaly')
ax.set_xlabel('Effective Load Ratio')
ax.set_ylabel('Peak Load Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

# Current vs Position
ax = axes[1, 0]
ax.scatter(df_sample.loc[normal_mask, 'SVM_Z_CURRENT'],
           df_sample.loc[normal_mask, 'SVM_Z_POSITION'],
           c='blue', s=2, alpha=alpha_normal, label='Normal')
ax.scatter(df_sample.loc[anomaly_mask, 'SVM_Z_CURRENT'],
           df_sample.loc[anomaly_mask, 'SVM_Z_POSITION'],
           c='red', s=10, alpha=alpha_anomaly, label='Anomaly')
ax.set_xlabel('Current')
ax.set_ylabel('Position')
ax.legend()
ax.grid(True, alpha=0.3)

# Position vs Peak Load
ax = axes[1, 1]
ax.scatter(df_sample.loc[normal_mask, 'SVM_Z_POSITION'],
           df_sample.loc[normal_mask, 'SVM_Z_PEAK_LOAD_RATIO'],
           c='blue', s=2, alpha=alpha_normal, label='Normal')
ax.scatter(df_sample.loc[anomaly_mask, 'SVM_Z_POSITION'],
           df_sample.loc[anomaly_mask, 'SVM_Z_PEAK_LOAD_RATIO'],
           c='red', s=10, alpha=alpha_anomaly, label='Anomaly')
ax.set_xlabel('Position')
ax.set_ylabel('Peak Load Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'anomaly_scatter.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_scatter.png")

# Figure 3: 이상 점수 분포
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Anomaly Score Distribution', fontsize=14, fontweight='bold')

# 히스토그램
ax = axes[0]
ax.hist(y_score[normal_mask], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
ax.hist(y_score[anomaly_mask], bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
ax.axvline(x=np.percentile(y_score, 95), color='green', linestyle='--', label='95th Percentile')
ax.set_xlabel('Anomaly Score')
ax.set_ylabel('Density')
ax.set_title('Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 시간에 따른 이상 점수
ax = axes[1]
ax.scatter(range(len(y_score)), y_score, c=colors, s=2, alpha=0.5)
ax.axhline(y=np.percentile(y_score, 95), color='green', linestyle='--', label='Threshold (95th)')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Anomaly Score')
ax.set_title('Anomaly Score over Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'anomaly_score_distribution.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_score_distribution.png")

# Figure 4: 센서별 박스플롯 비교
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
fig4.suptitle('Normal vs Anomaly - Sensor Value Distribution', fontsize=14, fontweight='bold')

for i, col in enumerate(pv_cols):
    ax = axes[i // 2, i % 2]

    normal_data = df_sample.loc[normal_mask, col]
    anomaly_data = df_sample.loc[anomaly_mask, col]

    bp = ax.boxplot([normal_data, anomaly_data], labels=['Normal', 'Anomaly'],
                    patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax.set_title(col.replace('SVM_Z_', ''))
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig4.savefig(os.path.join(output_dir, 'anomaly_boxplot.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_boxplot.png")

# Figure 5: Run별 이상 비율
fig5, ax = plt.subplots(figsize=(16, 6))

df_sample['is_anomaly'] = y_pred
run_anomaly = df_sample.groupby('Run').agg({
    'is_anomaly': ['sum', 'count']
}).reset_index()
run_anomaly.columns = ['Run', 'anomaly_count', 'total_count']
run_anomaly['anomaly_ratio'] = run_anomaly['anomaly_count'] / run_anomaly['total_count'] * 100

# 상위 30개 Run만 표시
top_runs = run_anomaly.nlargest(30, 'anomaly_ratio')

bars = ax.bar(range(len(top_runs)), top_runs['anomaly_ratio'], color='coral')
ax.set_xticks(range(len(top_runs)))
ax.set_xticklabels(top_runs['Run'], rotation=45, ha='right')
ax.set_xlabel('Run ID')
ax.set_ylabel('Anomaly Ratio (%)')
ax.set_title('Top 30 Runs with Highest Anomaly Ratio')
ax.axhline(y=5, color='red', linestyle='--', label='5% Threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig5.savefig(os.path.join(output_dir, 'anomaly_by_run.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_by_run.png")

# 4. 결과 요약
print("\n[4/4] 결과 요약...")
print(f"\n시각화 파일이 저장된 위치:")
print(f"  {os.path.abspath(output_dir)}")
print(f"\n생성된 파일:")
print(f"  1. anomaly_timeseries.png - 센서별 시계열 이상 탐지")
print(f"  2. anomaly_scatter.png - 센서 쌍별 산점도")
print(f"  3. anomaly_score_distribution.png - 이상 점수 분포")
print(f"  4. anomaly_boxplot.png - 정상/이상 박스플롯 비교")
print(f"  5. anomaly_by_run.png - Run별 이상 비율")

print("\n" + "=" * 70)
print("시각화 완료!")
print("=" * 70)
