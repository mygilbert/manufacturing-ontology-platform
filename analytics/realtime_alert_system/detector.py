"""
앙상블 이상 감지 엔진
Z-Score, CUSUM, Isolation Forest, LOF를 결합한 앙상블 감지
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# 기존 알고리즘 모듈 경로 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))

from anomaly_detection.algorithms import (
    ZScoreDetector,
    CUSUMDetector,
    IsolationForestDetector,
    LOFDetector
)

from .config import AlertConfig, AlertLevel
from .models import DetectionResult, SeverityLevel


class EnsembleAnomalyDetector:
    """앙상블 이상 감지 엔진"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False

        # 알고리즘 초기화
        self.algorithms = {
            'zscore': ZScoreDetector(threshold=config.ZSCORE_THRESHOLD),
            'cusum': CUSUMDetector(
                threshold=config.CUSUM_THRESHOLD,
                drift=config.CUSUM_DRIFT
            ),
            'isolation_forest': IsolationForestDetector(
                contamination=config.IF_CONTAMINATION,
                n_estimators=config.IF_N_ESTIMATORS
            ),
            'lof': LOFDetector(
                n_neighbors=config.LOF_N_NEIGHBORS,
                contamination=config.LOF_CONTAMINATION
            ),
        }

        # 실행 상태
        self.is_running = False
        self.total_processed = 0
        self.last_update: Optional[datetime] = None

        # 정규화를 위한 점수 히스토리
        self._score_history: Dict[str, List[float]] = {
            name: [] for name in self.algorithms.keys()
        }
        self._history_max_size = 1000

    def fit(self, X: np.ndarray) -> None:
        """
        초기 학습 (정상 데이터 기반)

        Args:
            X: 학습 데이터 (n_samples, n_features)
        """
        print(f"[Detector] Fitting with {len(X)} samples...")

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 각 알고리즘 학습
        for name, algo in self.algorithms.items():
            print(f"  - Training {name}...")
            algo.fit(X_scaled)

        self.is_fitted = True
        print("[Detector] Model fitting completed!")

    def predict(self, X: np.ndarray) -> DetectionResult:
        """
        앙상블 예측

        Args:
            X: 예측 데이터 (1, n_features) 또는 (n_features,)

        Returns:
            DetectionResult: 감지 결과
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # 입력 차원 처리
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 각 알고리즘 예측
        individual_scores: Dict[str, float] = {}
        individual_preds: Dict[str, int] = {}

        for name, algo in self.algorithms.items():
            score = algo.predict_score(X_scaled)[0]
            pred = algo.predict(X_scaled)[0]

            individual_scores[name] = float(score)
            individual_preds[name] = int(pred)

            # 히스토리 업데이트 (정규화용)
            self._score_history[name].append(score)
            if len(self._score_history[name]) > self._history_max_size:
                self._score_history[name].pop(0)

        # 점수 정규화 (0~1 범위)
        normalized_scores = self._normalize_scores(individual_scores)

        # 앙상블 점수 (정규화된 점수의 평균)
        ensemble_score = float(np.mean(list(normalized_scores.values())))

        # 앙상블 예측 (다수결: 2개 이상이 이상으로 판단)
        ensemble_pred = 1 if sum(individual_preds.values()) >= 2 else 0

        # 심각도 결정
        severity = self._get_severity(ensemble_score)

        # 상태 업데이트
        self.total_processed += 1
        self.last_update = datetime.now()

        return DetectionResult(
            timestamp=datetime.now(),
            window_index=self.total_processed,
            ensemble_score=ensemble_score,
            ensemble_prediction=ensemble_pred,
            individual_scores=normalized_scores,
            individual_predictions=individual_preds,
            severity=severity
        )

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        점수 정규화 (0~1 범위)

        히스토리 기반 Min-Max 정규화
        """
        normalized = {}

        for name, score in scores.items():
            history = self._score_history[name]
            if len(history) < 10:
                # 히스토리가 부족하면 단순 클리핑
                normalized[name] = min(max(score, 0), 1)
            else:
                min_val = np.percentile(history, 5)
                max_val = np.percentile(history, 95)

                if max_val - min_val < 1e-10:
                    normalized[name] = 0.5
                else:
                    norm_score = (score - min_val) / (max_val - min_val)
                    normalized[name] = float(np.clip(norm_score, 0, 1))

        return normalized

    def _get_severity(self, score: float) -> SeverityLevel:
        """앙상블 점수 기반 심각도 결정"""
        if score >= self.config.SCORE_THRESHOLD_EMERGENCY:
            return SeverityLevel.EMERGENCY
        elif score >= self.config.SCORE_THRESHOLD_CRITICAL:
            return SeverityLevel.CRITICAL
        elif score >= self.config.SCORE_THRESHOLD_WARNING:
            return SeverityLevel.WARNING
        return SeverityLevel.NORMAL

    def get_algorithm_names(self) -> List[str]:
        """사용 중인 알고리즘 이름 목록"""
        return list(self.algorithms.keys())

    def reset(self) -> None:
        """상태 초기화"""
        self.is_fitted = False
        self.total_processed = 0
        self.last_update = None
        self._score_history = {name: [] for name in self.algorithms.keys()}

        # 알고리즘 재초기화
        self.algorithms = {
            'zscore': ZScoreDetector(threshold=self.config.ZSCORE_THRESHOLD),
            'cusum': CUSUMDetector(
                threshold=self.config.CUSUM_THRESHOLD,
                drift=self.config.CUSUM_DRIFT
            ),
            'isolation_forest': IsolationForestDetector(
                contamination=self.config.IF_CONTAMINATION,
                n_estimators=self.config.IF_N_ESTIMATORS
            ),
            'lof': LOFDetector(
                n_neighbors=self.config.LOF_N_NEIGHBORS,
                contamination=self.config.LOF_CONTAMINATION
            ),
        }


def aggregate_window(data: np.ndarray, column_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    1분 윈도우 집계

    Args:
        data: 윈도우 내 데이터 (n_samples, n_features)
        column_names: 컬럼 이름 목록

    Returns:
        집계 결과 딕셔너리
    """
    result = {
        'mean': {},
        'std': {},
        'min': {},
        'max': {},
        'range': {}
    }

    for i, col in enumerate(column_names):
        col_data = data[:, i]
        result['mean'][col] = float(np.mean(col_data))
        result['std'][col] = float(np.std(col_data))
        result['min'][col] = float(np.min(col_data))
        result['max'][col] = float(np.max(col_data))
        result['range'][col] = float(np.max(col_data) - np.min(col_data))

    return result


def create_feature_vector(aggregated: Dict[str, Dict[str, float]], column_names: List[str]) -> np.ndarray:
    """
    집계된 데이터에서 특성 벡터 생성

    각 센서의 mean, std, range를 특성으로 사용
    """
    features = []
    for col in column_names:
        features.append(aggregated['mean'].get(col, 0))
        features.append(aggregated['std'].get(col, 0))
        features.append(aggregated['range'].get(col, 0))

    return np.array(features)
