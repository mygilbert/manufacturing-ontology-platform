"""
FDC 이상감지 Agent 테스트 스크립트
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

print("=" * 70)
print("FDC 이상감지 알고리즘 추천 Agent 테스트")
print("=" * 70)

# 1. 모듈 임포트 테스트
print("\n[1/6] 모듈 임포트 테스트...")
try:
    from anomaly_detection import DataLoader, FeatureEngineer, ModelEvaluator, AlgorithmRecommender
    from anomaly_detection.algorithms import (
        ZScoreDetector, CUSUMDetector, SPCDetector,
        IsolationForestDetector, LOFDetector, OneClassSVMDetector
    )
    print("  [OK] 모든 모듈 임포트 성공!")
except ImportError as e:
    print(f"  [FAIL] 임포트 오류: {e}")
    sys.exit(1)

# 딥러닝 모듈 (선택적)
try:
    from anomaly_detection.algorithms import AutoEncoderDetector, LSTMAutoEncoderDetector
    if AutoEncoderDetector is not None:
        DL_AVAILABLE = True
        print("  [OK] 딥러닝 모듈 사용 가능")
    else:
        DL_AVAILABLE = False
        print("  [!] 딥러닝 모듈 사용 불가 (TensorFlow 미설치)")
except (ImportError, TypeError):
    DL_AVAILABLE = False
    print("  [!] 딥러닝 모듈 사용 불가 (TensorFlow 미설치)")

# 2. 데이터 로딩 테스트
print("\n[2/6] 데이터 로딩 테스트...")
data_dir = os.path.join(script_dir, '..', 'sample_data')
pv_path = os.path.join(data_dir, 'sample_pv_data.csv')
fault_path = os.path.join(data_dir, 'sample_fault_labels.csv')

loader = DataLoader()
pv_data = loader.load_pv_data(pv_path)
fault_labels = loader.load_fault_labels(fault_path)
merged_data = loader.merge_data()
processed_data = loader.preprocess()

# 데이터 특성 분석
characteristics = loader.analyze_characteristics()
summary = loader.summary()

print("\n  데이터 요약:")
for key, value in summary.items():
    if key != '특성 컬럼':
        print(f"    - {key}: {value}")

# 3. 특성 추출 테스트 (데이터 일부만 사용)
print("\n[3/6] 특성 추출 테스트...")
# 테스트를 위해 데이터 일부만 사용 (10,000개)
test_size = min(10000, len(processed_data))
test_data = processed_data.iloc[:test_size].copy()

feature_cols = loader.get_feature_columns()
feature_engineer = FeatureEngineer()

# 간소화된 특성 추출
enriched_data = feature_engineer.extract_all_features(
    data=test_data,
    feature_cols=feature_cols,
    window_sizes=[5, 10],  # 윈도우 크기 축소
    include_fft=False,  # FFT 제외 (속도)
    include_peaks=False  # 피크 제외 (속도)
)

print(f"  [OK] 특성 추출 완료: {len(feature_engineer.feature_names)}개 특성")

# 특성 중요도
importance_df = feature_engineer.get_feature_importance_ranking(enriched_data)
print(f"\n  상위 5개 중요 특성:")
for _, row in importance_df.head(5).iterrows():
    print(f"    - {row['feature']}: {row['importance']:.4f}")

# 4. 데이터 분할
print("\n[4/6] 데이터 분할...")
train_data, test_data = loader.get_train_test_split(enriched_data, test_ratio=0.2)

exclude_cols = ['timestamp', 'is_anomaly']
feature_columns = [c for c in enriched_data.columns if c not in exclude_cols]

X_train = train_data[feature_columns].values
y_train = train_data['is_anomaly'].values
X_test = test_data[feature_columns].values
y_test = test_data['is_anomaly'].values

print(f"  [OK] 학습: {X_train.shape}, 테스트: {X_test.shape}")
print(f"  [OK] 학습 이상 비율: {y_train.mean()*100:.2f}%")
print(f"  [OK] 테스트 이상 비율: {y_test.mean()*100:.2f}%")

# 5. 알고리즘 학습 및 평가
print("\n[5/6] 알고리즘 학습 및 평가...")
evaluator = ModelEvaluator()

algorithms = {
    'Z-Score': ZScoreDetector(threshold=3.0),
    'CUSUM': CUSUMDetector(threshold=5.0),
    'SPC': SPCDetector(n_sigma=3.0),
    'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=50),
    'LOF': LOFDetector(n_neighbors=20),
    'One-Class SVM': OneClassSVMDetector(nu=0.05),
}

# 딥러닝 모델 추가 (사용 가능한 경우)
if DL_AVAILABLE and len(X_train) >= 5000:
    algorithms['AutoEncoder'] = AutoEncoderDetector(epochs=10, batch_size=128)

results_summary = []

for name, detector in algorithms.items():
    print(f"\n  평가 중: {name}...")

    try:
        # 학습
        start_time = time.time()
        detector.fit(X_train, y_train)
        training_time = time.time() - start_time

        # 예측
        start_time = time.time()
        y_pred = detector.predict(X_test)
        prediction_time = time.time() - start_time

        # 점수
        y_score = detector.predict_score(X_test)

        # 평가
        result = evaluator.evaluate(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            algorithm_name=name,
            training_time=training_time,
            prediction_time=prediction_time
        )

        results_summary.append({
            'Algorithm': name,
            'Precision': f"{result.precision:.4f}",
            'Recall': f"{result.recall:.4f}",
            'F1': f"{result.f1_score:.4f}",
            'AUC-ROC': f"{result.auc_roc:.4f}",
            'Time(s)': f"{training_time:.2f}"
        })

        print(f"    Precision: {result.precision:.4f}, Recall: {result.recall:.4f}, F1: {result.f1_score:.4f}")

    except Exception as e:
        print(f"    오류: {str(e)}")

# 6. 추천 결과
print("\n[6/6] 알고리즘 추천...")
recommender = AlgorithmRecommender()
recommendations = recommender.analyze_and_recommend(
    characteristics=characteristics,
    priorities={
        'speed': 0.2,
        'accuracy': 0.4,
        'interpretability': 0.2,
        'early_detection': 0.2
    }
)

print("\n  데이터 특성 기반 추천 순위:")
for rec in recommendations[:5]:
    print(f"    {rec.rank}. {rec.algorithm_name} (점수: {rec.score:.2f})")
    if rec.reasons:
        print(f"       이유: {rec.reasons[0]}")

# 최종 결과
print("\n" + "=" * 70)
print("테스트 결과 요약")
print("=" * 70)

# 비교 테이블
comparison_df = evaluator.compare_algorithms()
print("\n알고리즘 성능 비교:")
print(comparison_df.to_string(index=False))

# 최고 성능
best_name, best_result = evaluator.get_best_algorithm(metric='f1_score')
print(f"\n최고 성능 알고리즘 (F1 기준): {best_name}")
print(f"  - F1 Score: {best_result.f1_score:.4f}")
print(f"  - AUC-ROC: {best_result.auc_roc:.4f}")

# 추천
print(f"\n데이터 특성 기반 추천: {recommendations[0].algorithm_name}")

print("\n" + "=" * 70)
print("테스트 완료!")
print("=" * 70)
