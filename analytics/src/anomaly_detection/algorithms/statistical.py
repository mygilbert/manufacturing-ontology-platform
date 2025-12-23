"""
통계 기반 이상감지 알고리즘

- Z-score 기반 탐지
- CUSUM (Cumulative Sum) 탐지
- SPC (Statistical Process Control) 관리도 기반 탐지
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """이상감지 기본 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측 (0: 정상, 1: 이상)"""
        pass

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수 계산 (높을수록 이상)"""
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """학습 및 예측"""
        self.fit(X, y)
        return self.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """파라미터 반환"""
        return self.params.copy()


class ZScoreDetector(BaseDetector):
    """Z-score 기반 이상감지"""

    def __init__(self, threshold: float = 3.0, window_size: Optional[int] = None):
        """
        Args:
            threshold: Z-score 임계값 (기본값: 3.0)
            window_size: 이동 윈도우 크기 (None이면 전체 데이터 사용)
        """
        super().__init__("Z-Score Detector")
        self.threshold = threshold
        self.window_size = window_size
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.params = {
            'threshold': threshold,
            'window_size': window_size
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """학습 데이터에서 평균과 표준편차 계산"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-10  # 0으로 나누기 방지

        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Z-score 계산"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.window_size is not None:
            # 이동 윈도우 기반 Z-score
            scores = np.zeros(len(X))
            for i in range(len(X)):
                start_idx = max(0, i - self.window_size + 1)
                window = X[start_idx:i+1]
                window_mean = np.mean(window, axis=0)
                window_std = np.std(window, axis=0)
                window_std[window_std == 0] = 1e-10
                z = np.abs((X[i] - window_mean) / window_std)
                scores[i] = np.max(z)
        else:
            z_scores = np.abs((X - self.mean_) / self.std_)
            scores = np.max(z_scores, axis=1)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)


class CUSUMDetector(BaseDetector):
    """CUSUM (Cumulative Sum) 기반 이상감지"""

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
        direction: str = 'both'
    ):
        """
        Args:
            threshold: CUSUM 임계값
            drift: 허용 드리프트 (k 값)
            direction: 탐지 방향 ('positive', 'negative', 'both')
        """
        super().__init__("CUSUM Detector")
        self.threshold = threshold
        self.drift = drift
        self.direction = direction
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.params = {
            'threshold': threshold,
            'drift': drift,
            'direction': direction
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """학습 데이터에서 기준 통계량 계산"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-10

        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """CUSUM 점수 계산"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 정규화
        X_norm = (X - self.mean_) / self.std_

        n_samples, n_features = X_norm.shape
        scores = np.zeros(n_samples)

        for f in range(n_features):
            s_pos = 0
            s_neg = 0

            for i in range(n_samples):
                # 양의 방향 CUSUM
                s_pos = max(0, s_pos + X_norm[i, f] - self.drift)
                # 음의 방향 CUSUM
                s_neg = max(0, s_neg - X_norm[i, f] - self.drift)

                if self.direction == 'positive':
                    scores[i] = max(scores[i], s_pos)
                elif self.direction == 'negative':
                    scores[i] = max(scores[i], s_neg)
                else:  # both
                    scores[i] = max(scores[i], s_pos, s_neg)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)


class SPCDetector(BaseDetector):
    """SPC (Statistical Process Control) 관리도 기반 이상감지"""

    def __init__(
        self,
        n_sigma: float = 3.0,
        use_western_electric_rules: bool = True
    ):
        """
        Args:
            n_sigma: 관리 한계 시그마 수 (기본값: 3.0)
            use_western_electric_rules: Western Electric 규칙 사용 여부
        """
        super().__init__("SPC Control Chart Detector")
        self.n_sigma = n_sigma
        self.use_western_electric_rules = use_western_electric_rules
        self.center_line_: Optional[np.ndarray] = None
        self.ucl_: Optional[np.ndarray] = None  # Upper Control Limit
        self.lcl_: Optional[np.ndarray] = None  # Lower Control Limit
        self.std_: Optional[np.ndarray] = None
        self.params = {
            'n_sigma': n_sigma,
            'use_western_electric_rules': use_western_electric_rules
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """관리 한계 계산"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.center_line_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1e-10

        self.ucl_ = self.center_line_ + self.n_sigma * self.std_
        self.lcl_ = self.center_line_ - self.n_sigma * self.std_

        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """관리도 기반 점수 계산"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 정규화된 거리 (시그마 단위)
        z_scores = np.abs(X - self.center_line_) / self.std_
        scores = np.max(z_scores, axis=1)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = len(X)
        anomalies = np.zeros(n_samples, dtype=int)

        # Rule 1: 관리 한계 초과
        for f in range(X.shape[1]):
            out_of_limits = (X[:, f] > self.ucl_[f]) | (X[:, f] < self.lcl_[f])
            anomalies[out_of_limits] = 1

        if self.use_western_electric_rules:
            anomalies = self._apply_western_electric_rules(X, anomalies)

        return anomalies

    def _apply_western_electric_rules(
        self,
        X: np.ndarray,
        anomalies: np.ndarray
    ) -> np.ndarray:
        """Western Electric 규칙 적용"""
        n_samples = len(X)

        for f in range(X.shape[1]):
            sigma_1 = self.std_[f]
            sigma_2 = 2 * self.std_[f]
            cl = self.center_line_[f]

            for i in range(n_samples):
                # Rule 2: 연속 9개 점이 중심선의 같은 쪽
                if i >= 8:
                    window = X[i-8:i+1, f]
                    if np.all(window > cl) or np.all(window < cl):
                        anomalies[i] = 1

                # Rule 3: 연속 6개 점이 증가 또는 감소
                if i >= 5:
                    window = X[i-5:i+1, f]
                    diffs = np.diff(window)
                    if np.all(diffs > 0) or np.all(diffs < 0):
                        anomalies[i] = 1

                # Rule 4: 연속 3개 중 2개가 2시그마 초과
                if i >= 2:
                    window = X[i-2:i+1, f]
                    beyond_2sigma = np.abs(window - cl) > sigma_2
                    if np.sum(beyond_2sigma) >= 2:
                        anomalies[i] = 1

        return anomalies

    def get_control_limits(self) -> Dict[str, np.ndarray]:
        """관리 한계 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        return {
            'center_line': self.center_line_,
            'upper_control_limit': self.ucl_,
            'lower_control_limit': self.lcl_
        }
