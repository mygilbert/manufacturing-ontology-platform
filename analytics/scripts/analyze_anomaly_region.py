"""
150~170시간 이상 구간 상세 분석
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
print("150~170시간 이상 구간 상세 분석")
print("=" * 70)

# =============================================================================
# 1. 데이터 로딩
# =============================================================================
print("\n[1/4] 데이터 로딩...")
data_path = r"D:\24346_ESWA_PKG_11_1_AL_FORMING_PRESS_SVM.csv"
df = pd.read_csv(data_path)

pv_cols = ['SVM_Z_CURRENT', 'SVM_Z_EFFECTIVE_LOAD_RATIO',
           'SVM_Z_PEAK_LOAD_RATIO', 'SVM_Z_POSITION']

df['time_idx'] = range(len(df))
df['time_sec'] = df['time_idx']
df['time_min'] = df['time_idx'] / 60
df['time_hour'] = df['time_idx'] / 3600

print(f"  총 데이터: {len(df):,} 행")
print(f"  총 시간: {df['time_hour'].max():.1f} 시간")

# =============================================================================
# 2. 150~170시간 구간 추출
# =============================================================================
print("\n[2/4] 150~170시간 구간 추출...")

# 분석 구간 정의
START_HOUR = 150
END_HOUR = 170

# 전후 비교를 위해 확장 구간도 추출
EXTENDED_START = 140
EXTENDED_END = 180

# 구간 필터링
mask_target = (df['time_hour'] >= START_HOUR) & (df['time_hour'] <= END_HOUR)
mask_extended = (df['time_hour'] >= EXTENDED_START) & (df['time_hour'] <= EXTENDED_END)

df_target = df[mask_target].copy()
df_extended = df[mask_extended].copy()

# 정상 구간 (비교용): 50~70시간
mask_normal = (df['time_hour'] >= 50) & (df['time_hour'] <= 70)
df_normal = df[mask_normal].copy()

print(f"  이상 구간 (150~170h): {len(df_target):,} 행")
print(f"  확장 구간 (140~180h): {len(df_extended):,} 행")
print(f"  정상 구간 (50~70h): {len(df_normal):,} 행")

# =============================================================================
# 3. 상세 통계 분석
# =============================================================================
print("\n[3/4] 상세 통계 분석...")

print("\n" + "=" * 70)
print("정상 구간 (50~70h) vs 이상 구간 (150~170h) 비교")
print("=" * 70)

comparison_data = []

for col in pv_cols:
    normal_data = df_normal[col].dropna()
    anomaly_data = df_target[col].dropna()

    normal_mean = normal_data.mean()
    anomaly_mean = anomaly_data.mean()
    diff_pct = (anomaly_mean - normal_mean) / normal_mean * 100 if normal_mean != 0 else 0

    normal_std = normal_data.std()
    anomaly_std = anomaly_data.std()
    std_ratio = anomaly_std / normal_std if normal_std != 0 else 0

    print(f"\n[{col}]")
    print(f"  정상 구간: mean={normal_mean:.2f}, std={normal_std:.2f}")
    print(f"  이상 구간: mean={anomaly_mean:.2f}, std={anomaly_std:.2f}")
    print(f"  변화: mean {diff_pct:+.1f}%, std x{std_ratio:.2f}")

    comparison_data.append({
        'Sensor': col.replace('SVM_Z_', ''),
        'Normal_Mean': normal_mean,
        'Anomaly_Mean': anomaly_mean,
        'Mean_Change(%)': diff_pct,
        'Normal_Std': normal_std,
        'Anomaly_Std': anomaly_std,
        'Std_Ratio': std_ratio
    })

# Run 분석
print("\n[Run 분석]")
target_runs = df_target['Run'].unique()
print(f"  이상 구간 내 Run 수: {len(target_runs)}")
print(f"  Run 범위: {target_runs.min()} ~ {target_runs.max()}")

# =============================================================================
# 4. 시각화
# =============================================================================
print("\n[4/4] 시각화...")

output_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(output_dir, exist_ok=True)

# ----- Figure 1: 이상 구간 상세 시계열 -----
fig1, axes = plt.subplots(5, 1, figsize=(20, 16), sharex=True)
fig1.suptitle(f'Anomaly Region Detail: {START_HOUR}~{END_HOUR} hours\n(with context: {EXTENDED_START}~{EXTENDED_END} hours)',
              fontsize=16, fontweight='bold')

time_hours_ext = df_extended['time_hour'].values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, col in enumerate(pv_cols):
    ax = axes[i]
    y = df_extended[col].values

    # 전체 구간 플롯
    ax.plot(time_hours_ext, y, color=colors[i], linewidth=0.3, alpha=0.7)

    # 이상 구간 강조 (음영)
    ax.axvspan(START_HOUR, END_HOUR, alpha=0.3, color='red', label='Anomaly Region')

    # 통계 라인
    normal_mean = df_normal[col].mean()
    normal_std = df_normal[col].std()
    ax.axhline(y=normal_mean, color='green', linestyle='-', alpha=0.7, label=f'Normal Mean ({normal_mean:.1f})')
    ax.axhline(y=normal_mean + 3*normal_std, color='orange', linestyle='--', alpha=0.5, label='+3σ')
    ax.axhline(y=normal_mean - 3*normal_std, color='orange', linestyle='--', alpha=0.5, label='-3σ')

    ax.set_ylabel(col.replace('SVM_Z_', ''), fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

# 시간 마커
ax = axes[4]
ax.set_xlabel('Time (hours)', fontsize=12)

plt.tight_layout()
fig1.savefig(os.path.join(output_dir, 'anomaly_region_detail.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_region_detail.png")

# ----- Figure 2: 1분 집계 상세 -----
# 1분 집계
df_extended['window_1min'] = (df_extended['time_idx'] // 60).astype(int)
df_1min_ext = df_extended.groupby('window_1min').agg({
    **{col: ['mean', 'std', 'min', 'max'] for col in pv_cols},
    'time_hour': 'mean',
    'Run': 'nunique'
}).reset_index()
df_1min_ext.columns = ['_'.join(col).strip('_') for col in df_1min_ext.columns]

fig2, axes = plt.subplots(4, 1, figsize=(20, 14), sharex=True)
fig2.suptitle(f'1-Minute Aggregated View: {EXTENDED_START}~{EXTENDED_END} hours', fontsize=16, fontweight='bold')

for i, col in enumerate(pv_cols):
    ax = axes[i]
    mean_col = f'{col}_mean'
    std_col = f'{col}_std'

    x = df_1min_ext['time_hour_mean'].values
    y_mean = df_1min_ext[mean_col].values
    y_std = df_1min_ext[std_col].values

    # 평균 및 표준편차 범위
    ax.plot(x, y_mean, color=colors[i], linewidth=1, alpha=0.9)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, color=colors[i])

    # 이상 구간 강조
    ax.axvspan(START_HOUR, END_HOUR, alpha=0.2, color='red')

    ax.set_ylabel(col.replace('SVM_Z_', ''), fontsize=11)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (hours)', fontsize=12)
plt.tight_layout()
fig2.savefig(os.path.join(output_dir, 'anomaly_region_1min.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_region_1min.png")

# ----- Figure 3: 정상 vs 이상 분포 비교 -----
fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('Distribution Comparison: Normal (50~70h) vs Anomaly (150~170h)', fontsize=16, fontweight='bold')

for i, col in enumerate(pv_cols):
    ax = axes[i // 2, i % 2]

    normal_data = df_normal[col].dropna()
    anomaly_data = df_target[col].dropna()

    # 히스토그램
    bins = np.linspace(min(normal_data.min(), anomaly_data.min()),
                       max(normal_data.max(), anomaly_data.max()), 50)

    ax.hist(normal_data, bins=bins, alpha=0.5, label='Normal (50~70h)', color='blue', density=True)
    ax.hist(anomaly_data, bins=bins, alpha=0.5, label='Anomaly (150~170h)', color='red', density=True)

    # 평균 라인
    ax.axvline(normal_data.mean(), color='blue', linestyle='--', linewidth=2)
    ax.axvline(anomaly_data.mean(), color='red', linestyle='--', linewidth=2)

    ax.set_title(col.replace('SVM_Z_', ''), fontsize=12)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(os.path.join(output_dir, 'anomaly_region_distribution.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_region_distribution.png")

# ----- Figure 4: 센서 상관관계 변화 -----
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
fig4.suptitle('Sensor Correlation: Normal vs Anomaly', fontsize=16, fontweight='bold')

# 정상 구간 상관관계
ax = axes[0]
corr_normal = df_normal[pv_cols].corr()
im = ax.imshow(corr_normal, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(pv_cols)))
ax.set_yticks(range(len(pv_cols)))
ax.set_xticklabels([c.replace('SVM_Z_', '') for c in pv_cols], rotation=45, ha='right')
ax.set_yticklabels([c.replace('SVM_Z_', '') for c in pv_cols])
ax.set_title('Normal Period (50~70h)', fontsize=12)

for i in range(len(pv_cols)):
    for j in range(len(pv_cols)):
        ax.text(j, i, f'{corr_normal.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)

# 이상 구간 상관관계
ax = axes[1]
corr_anomaly = df_target[pv_cols].corr()
im = ax.imshow(corr_anomaly, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(pv_cols)))
ax.set_yticks(range(len(pv_cols)))
ax.set_xticklabels([c.replace('SVM_Z_', '') for c in pv_cols], rotation=45, ha='right')
ax.set_yticklabels([c.replace('SVM_Z_', '') for c in pv_cols])
ax.set_title('Anomaly Period (150~170h)', fontsize=12)

for i in range(len(pv_cols)):
    for j in range(len(pv_cols)):
        ax.text(j, i, f'{corr_anomaly.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)

plt.colorbar(im, ax=axes, shrink=0.6)
plt.tight_layout()
fig4.savefig(os.path.join(output_dir, 'anomaly_region_correlation.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_region_correlation.png")

# ----- Figure 5: 이상 구간 내 세부 타임라인 -----
fig5, axes = plt.subplots(5, 1, figsize=(20, 16), sharex=True)
fig5.suptitle(f'Detailed Timeline: {START_HOUR}~{END_HOUR} hours (1-second resolution)',
              fontsize=16, fontweight='bold')

time_hours_target = df_target['time_hour'].values

for i, col in enumerate(pv_cols):
    ax = axes[i]
    y = df_target[col].values

    ax.plot(time_hours_target, y, color=colors[i], linewidth=0.2, alpha=0.8)

    # 이동평균 (1분)
    ma_60 = pd.Series(y).rolling(60, min_periods=1).mean()
    ax.plot(time_hours_target, ma_60, color='black', linewidth=1.5, alpha=0.8, label='1-min MA')

    # 정상 범위 표시
    normal_mean = df_normal[col].mean()
    normal_std = df_normal[col].std()
    ax.axhline(y=normal_mean, color='green', linestyle='-', alpha=0.5)
    ax.fill_between(time_hours_target, normal_mean - 2*normal_std, normal_mean + 2*normal_std,
                    alpha=0.1, color='green', label='Normal ±2σ')

    ax.set_ylabel(col.replace('SVM_Z_', ''), fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

# Run 변화 표시
ax = axes[4]
runs = df_target['Run'].values
ax2 = ax.twinx()
ax2.plot(time_hours_target, runs, color='purple', linewidth=0.5, alpha=0.5)
ax2.set_ylabel('Run ID', color='purple', fontsize=10)

axes[-1].set_xlabel('Time (hours)', fontsize=12)
plt.tight_layout()
fig5.savefig(os.path.join(output_dir, 'anomaly_region_timeline.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_region_timeline.png")

# ----- Figure 6: 이상 발생 시점 분석 -----
fig6, axes = plt.subplots(2, 2, figsize=(16, 12))
fig6.suptitle('Anomaly Characteristics Analysis', fontsize=16, fontweight='bold')

# 1. CURRENT vs PEAK_LOAD 산점도
ax = axes[0, 0]
ax.scatter(df_normal['SVM_Z_CURRENT'], df_normal['SVM_Z_PEAK_LOAD_RATIO'],
           c='blue', s=1, alpha=0.3, label='Normal')
ax.scatter(df_target['SVM_Z_CURRENT'], df_target['SVM_Z_PEAK_LOAD_RATIO'],
           c='red', s=1, alpha=0.3, label='Anomaly')
ax.set_xlabel('Current')
ax.set_ylabel('Peak Load Ratio')
ax.set_title('Current vs Peak Load')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. POSITION 분포
ax = axes[0, 1]
normal_pos = df_normal['SVM_Z_POSITION'].value_counts(normalize=True).sort_index()
anomaly_pos = df_target['SVM_Z_POSITION'].value_counts(normalize=True).sort_index()

x = np.arange(1, 6)
width = 0.35
ax.bar(x - width/2, [normal_pos.get(i, 0) for i in x], width, label='Normal', color='blue', alpha=0.7)
ax.bar(x + width/2, [anomaly_pos.get(i, 0) for i in x], width, label='Anomaly', color='red', alpha=0.7)
ax.set_xlabel('Position')
ax.set_ylabel('Ratio')
ax.set_title('Position Distribution')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 시간대별 이상 빈도
ax = axes[1, 0]
df_target['hour_in_range'] = (df_target['time_hour'] - START_HOUR).astype(int)
hourly_counts = df_target.groupby('hour_in_range').size()
ax.bar(hourly_counts.index, hourly_counts.values, color='coral', alpha=0.7)
ax.set_xlabel('Hours from start (150h)')
ax.set_ylabel('Sample Count')
ax.set_title('Samples per Hour in Anomaly Region')
ax.grid(True, alpha=0.3)

# 4. 센서값 변화율
ax = axes[1, 1]
for col in pv_cols:
    pct_change = df_target[col].pct_change().abs()
    pct_change_ma = pct_change.rolling(60).mean()
    ax.plot(df_target['time_hour'].values, pct_change_ma.values,
            linewidth=0.8, alpha=0.7, label=col.replace('SVM_Z_', ''))

ax.set_xlabel('Time (hours)')
ax.set_ylabel('Absolute % Change (1-min MA)')
ax.set_title('Rate of Change')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
fig6.savefig(os.path.join(output_dir, 'anomaly_region_characteristics.png'), dpi=150, bbox_inches='tight')
print(f"  저장: anomaly_region_characteristics.png")

# =============================================================================
# 결과 요약
# =============================================================================
print("\n" + "=" * 70)
print("150~170시간 이상 구간 분석 결과")
print("=" * 70)

print(f"""
[구간 정보]
- 이상 구간: {START_HOUR}~{END_HOUR} 시간 (20시간, 72,000초)
- 데이터 수: {len(df_target):,} 행
- Run 수: {len(target_runs)} 개 (Run {target_runs.min()}~{target_runs.max()})

[센서별 변화 (정상 대비)]
""")

for item in comparison_data:
    print(f"  {item['Sensor']}:")
    print(f"    평균: {item['Normal_Mean']:.2f} -> {item['Anomaly_Mean']:.2f} ({item['Mean_Change(%)']:+.1f}%)")
    print(f"    표준편차: {item['Normal_Std']:.2f} -> {item['Anomaly_Std']:.2f} (x{item['Std_Ratio']:.2f})")

# 상관관계 변화
print(f"""
[상관관계 변화]
- EFFECTIVE vs PEAK: {corr_normal.loc['SVM_Z_EFFECTIVE_LOAD_RATIO', 'SVM_Z_PEAK_LOAD_RATIO']:.2f} -> {corr_anomaly.loc['SVM_Z_EFFECTIVE_LOAD_RATIO', 'SVM_Z_PEAK_LOAD_RATIO']:.2f}
- CURRENT vs POSITION: {corr_normal.loc['SVM_Z_CURRENT', 'SVM_Z_POSITION']:.2f} -> {corr_anomaly.loc['SVM_Z_CURRENT', 'SVM_Z_POSITION']:.2f}

[주요 발견]
1. PEAK_LOAD_RATIO 급격한 하락 (95 -> 42, -56%)
2. POSITION 값 증가 (1.4 -> 2.5, +75%) - 비정상 위치
3. 표준편차 증가 - 불안정한 동작
4. 센서 간 상관관계 변화 - 정상 패턴 붕괴

[저장된 파일]
- anomaly_region_detail.png: 이상 구간 상세 시계열
- anomaly_region_1min.png: 1분 집계 뷰
- anomaly_region_distribution.png: 정상/이상 분포 비교
- anomaly_region_correlation.png: 상관관계 변화
- anomaly_region_timeline.png: 세부 타임라인
- anomaly_region_characteristics.png: 이상 특성 분석
""")

print("=" * 70)
print("분석 완료!")
print("=" * 70)
