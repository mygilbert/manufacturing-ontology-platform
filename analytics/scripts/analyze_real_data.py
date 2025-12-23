"""
실제 FDC 데이터 분석 스크립트 (비지도 학습)
AL FORMING PRESS SVM 데이터 분석
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

print("=" * 70)
print("AL FORMING PRESS SVM 데이터 이상감지 분석")
print("=" * 70)

# 1. 데이터 로딩
print("\n[1/5] 데이터 로딩...")
data_path = r"D:\24346_ESWA_PKG_11_1_AL_FORMING_PRESS_SVM.csv"
df = pd.read_csv(data_path)
print(f"  - 총 행 수: {len(df):,}")
print(f"  - 총 컬럼 수: {len(df.columns)}")

# 2. 데이터 전처리
print("\n[2/5] 데이터 전처리...")

# PV 컬럼 선택
pv_cols = ['SVM_Z_CURRENT', 'SVM_Z_EFFECTIVE_LOAD_RATIO',
           'SVM_Z_PEAK_LOAD_RATIO', 'SVM_Z_POSITION']

# 결측치 제거
df_clean = df[pv_cols].dropna()
print(f"  - 결측치 제거 후: {len(df_clean):,} 행")

# 3. 데이터 특성 분석
print("\n[3/5] 데이터 특성 분석...")

print("\n  [PV 데이터 통계]")
print(df_clean.describe().round(2).to_string())

# 상관관계
print("\n  [상관관계 분석]")
corr = df_clean.corr()
print(corr.round(3).to_string())

# 시계열 특성
print("\n  [시계열 특성]")
for col in pv_cols:
    autocorr = df_clean[col].autocorr(lag=1)
    cv = df_clean[col].std() / abs(df_clean[col].mean()) if df_clean[col].mean() != 0 else 0
    print(f"  - {col}: 자기상관={autocorr:.3f}, 변동계수={cv:.3f}")

# 4. 비지도 이상감지 알고리즘 적용
print("\n[4/5] 비지도 이상감지 알고리즘 적용...")

# 데이터 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(df_clean.values)

# 샘플링 (전체 데이터가 너무 크면)
sample_size = min(100000, len(X))
if len(X) > sample_size:
    np.random.seed(42)
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    print(f"  - 샘플링: {sample_size:,} 샘플 사용")
else:
    X_sample = X
    indices = np.arange(len(X))

# 알고리즘 임포트
from anomaly_detection.algorithms import (
    ZScoreDetector, CUSUMDetector, SPCDetector,
    IsolationForestDetector, LOFDetector, OneClassSVMDetector
)

algorithms = {
    'Z-Score': ZScoreDetector(threshold=3.0),
    'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
    'LOF': LOFDetector(n_neighbors=20, contamination=0.05),
    'One-Class SVM': OneClassSVMDetector(nu=0.05),
}

results = {}

for name, detector in algorithms.items():
    print(f"\n  [{name}] 학습 및 예측 중...")

    try:
        start_time = time.time()

        # 학습 (비지도이므로 y=None)
        detector.fit(X_sample)

        # 예측
        y_pred = detector.predict(X_sample)
        y_score = detector.predict_score(X_sample)

        elapsed = time.time() - start_time

        # 이상 탐지 결과
        n_anomalies = y_pred.sum()
        anomaly_ratio = n_anomalies / len(y_pred) * 100

        results[name] = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': anomaly_ratio,
            'score_mean': y_score.mean(),
            'score_std': y_score.std(),
            'time': elapsed,
            'predictions': y_pred,
            'scores': y_score
        }

        print(f"    - 탐지된 이상: {n_anomalies:,} ({anomaly_ratio:.2f}%)")
        print(f"    - 이상 점수: mean={y_score.mean():.3f}, std={y_score.std():.3f}")
        print(f"    - 소요 시간: {elapsed:.2f}초")

    except Exception as e:
        print(f"    - 오류: {str(e)}")

# 5. 결과 분석 및 추천
print("\n[5/5] 결과 분석 및 추천...")

print("\n  [알고리즘 비교]")
print("-" * 70)
print(f"{'알고리즘':<20} {'탐지 이상 수':>12} {'이상 비율':>10} {'소요시간':>10}")
print("-" * 70)

for name, res in results.items():
    print(f"{name:<20} {res['n_anomalies']:>12,} {res['anomaly_ratio']:>9.2f}% {res['time']:>9.2f}s")

print("-" * 70)

# 앙상블 분석 (여러 알고리즘이 동시에 이상으로 판단한 샘플)
print("\n  [앙상블 분석]")
if len(results) >= 2:
    all_preds = np.stack([res['predictions'] for res in results.values()], axis=1)

    # 2개 이상 알고리즘이 이상으로 판단
    ensemble_2 = (all_preds.sum(axis=1) >= 2).sum()
    # 3개 이상 알고리즘이 이상으로 판단
    ensemble_3 = (all_preds.sum(axis=1) >= 3).sum()
    # 모든 알고리즘이 이상으로 판단
    ensemble_all = (all_preds.sum(axis=1) == len(results)).sum()

    print(f"  - 2개 이상 알고리즘 일치: {ensemble_2:,} ({ensemble_2/len(X_sample)*100:.2f}%)")
    print(f"  - 3개 이상 알고리즘 일치: {ensemble_3:,} ({ensemble_3/len(X_sample)*100:.2f}%)")
    print(f"  - 모든 알고리즘 일치: {ensemble_all:,} ({ensemble_all/len(X_sample)*100:.2f}%)")

# 이상 데이터 샘플 분석
print("\n  [이상 데이터 특성 분석]")
if 'Isolation Forest' in results:
    if_pred = results['Isolation Forest']['predictions']
    if_scores = results['Isolation Forest']['scores']

    # 이상으로 판단된 샘플의 원본 데이터
    anomaly_indices = indices[if_pred == 1]

    if len(anomaly_indices) > 0:
        anomaly_data = df_clean.iloc[anomaly_indices]
        normal_data = df_clean.iloc[indices[if_pred == 0]]

        print("\n  정상 vs 이상 데이터 비교:")
        print("-" * 70)
        print(f"{'센서':<30} {'정상 평균':>15} {'이상 평균':>15} {'차이':>10}")
        print("-" * 70)

        for col in pv_cols:
            normal_mean = normal_data[col].mean()
            anomaly_mean = anomaly_data[col].mean()
            diff_pct = (anomaly_mean - normal_mean) / normal_mean * 100 if normal_mean != 0 else 0
            print(f"{col:<30} {normal_mean:>15.2f} {anomaly_mean:>15.2f} {diff_pct:>9.1f}%")

# 최종 추천
print("\n" + "=" * 70)
print("최종 추천")
print("=" * 70)

print("""
[데이터 특성]
- 설비: AL FORMING PRESS SVM (Z축 서보모터)
- 주요 센서: 전류, 유효부하율, 피크부하율, 위치
- 데이터 크기: 약 93만 샘플

[추천 알고리즘]
1. Isolation Forest (추천)
   - 고차원 데이터에 효과적
   - 빠른 학습/예측 속도
   - contamination 파라미터로 이상 비율 조절 가능

2. LOF (Local Outlier Factor)
   - 지역적 밀도 기반으로 이상치 탐지
   - 군집 구조가 있는 데이터에 효과적

[운영 적용 가이드]
1. contamination 값을 0.01~0.1 범위에서 조절하여 민감도 설정
2. 앙상블 방식으로 여러 알고리즘 결과를 종합하여 오탐 감소
3. 이상 점수 임계값을 조절하여 경보 수준 구분 (주의/경고/위험)
""")

print("=" * 70)
print("분석 완료!")
print("=" * 70)
