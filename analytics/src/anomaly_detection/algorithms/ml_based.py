"""
ML 기반 이상감지 알고리즘

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
"""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class BaseMLDetector:
    """ML 이상감지 기본 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.model = None
        self.params: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측 (0: 정상, 1: 이상)"""
        raise NotImplementedError

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수 계산"""
        raise NotImplementedError

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """학습 및 예측"""
        self.fit(X, y)
        return self.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """파라미터 반환"""
        return self.params.copy()


class IsolationForestDetector(BaseMLDetector):
    """Isolation Forest 기반 이상감지"""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: int = 42
    ):
        """
        Args:
            contamination: 예상 이상 비율
            n_estimators: 트리 개수
            max_samples: 샘플링 크기
            random_state: 랜덤 시드
        """
        super().__init__("Isolation Forest")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )

        self.params = {
            'contamination': contamination,
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'random_state': random_state
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 정상 데이터만 사용 (라벨이 있는 경우)
        if y is not None:
            y = np.asarray(y)
            normal_mask = y == 0
            if np.sum(normal_mask) > 0:
                X_scaled = X_scaled[normal_mask]

        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수 계산 (높을수록 이상)"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)

        # Isolation Forest의 decision_function은 낮을수록 이상
        # 부호를 반전하여 높을수록 이상하도록 변환
        scores = -self.model.decision_function(X_scaled)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)

        # -1: 이상, 1: 정상 → 0: 정상, 1: 이상으로 변환
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)


class LOFDetector(BaseMLDetector):
    """Local Outlier Factor 기반 이상감지"""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        novelty: bool = True
    ):
        """
        Args:
            n_neighbors: 이웃 개수
            contamination: 예상 이상 비율
            novelty: True면 새로운 데이터 예측 가능
        """
        super().__init__("Local Outlier Factor")
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty,
            n_jobs=-1
        )

        self.params = {
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'novelty': novelty
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(X)

        # 정상 데이터만 사용
        if y is not None:
            y = np.asarray(y)
            normal_mask = y == 0
            if np.sum(normal_mask) > 0:
                X_scaled = X_scaled[normal_mask]

        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수 계산"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)

        # LOF 점수 (낮을수록 이상) → 부호 반전
        scores = -self.model.decision_function(X_scaled)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)


class OneClassSVMDetector(BaseMLDetector):
    """One-Class SVM 기반 이상감지"""

    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: str = 'scale',
        nu: float = 0.05
    ):
        """
        Args:
            kernel: 커널 유형 ('rbf', 'linear', 'poly')
            gamma: 커널 계수
            nu: 이상치 비율 상한
        """
        super().__init__("One-Class SVM")
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu

        self.model = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu
        )

        self.params = {
            'kernel': kernel,
            'gamma': gamma,
            'nu': nu
        }

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.fit_transform(X)

        # 정상 데이터만 사용
        if y is not None:
            y = np.asarray(y)
            normal_mask = y == 0
            if np.sum(normal_mask) > 0:
                X_scaled = X_scaled[normal_mask]

        # 데이터가 너무 크면 샘플링
        max_samples = 10000
        if len(X_scaled) > max_samples:
            indices = np.random.choice(len(X_scaled), max_samples, replace=False)
            X_scaled = X_scaled[indices]

        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수 계산"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)

        # SVM 점수 (낮을수록 이상) → 부호 반전
        scores = -self.model.decision_function(X_scaled)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)
