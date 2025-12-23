"""
Isolation Forest 기반 이상 감지

다변량 센서 데이터에서 이상치를 탐지합니다.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('/app/src')

from config import config

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest 이상 감지기"""

    def __init__(
        self,
        equipment_id: str,
        feature_names: List[str],
        contamination: float = None,
        n_estimators: int = None,
    ):
        self.equipment_id = equipment_id
        self.feature_names = feature_names
        self.contamination = contamination or config.anomaly.isolation_forest_contamination
        self.n_estimators = n_estimators or config.anomaly.isolation_forest_n_estimators

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        self.training_timestamp: Optional[datetime] = None
        self.training_samples: int = 0

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        모델 학습

        Args:
            data: 학습 데이터 (feature_names 컬럼 포함)

        Returns:
            학습 결과 메트릭
        """
        logger.info(f"Training Isolation Forest for equipment {self.equipment_id}")

        # 피처 검증
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # 피처 추출
        X = data[self.feature_names].values

        # 결측치 처리
        X = np.nan_to_num(X, nan=0.0)

        # 스케일링
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 모델 학습
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=config.anomaly.isolation_forest_max_samples,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        # 학습 메타데이터
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        self.training_samples = len(X)

        # 학습 데이터에 대한 예측 (baseline)
        scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)

        anomaly_ratio = (predictions == -1).sum() / len(predictions)

        metrics = {
            "equipment_id": self.equipment_id,
            "training_samples": self.training_samples,
            "training_timestamp": self.training_timestamp.isoformat(),
            "contamination": self.contamination,
            "anomaly_ratio": float(anomaly_ratio),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
        }

        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상 탐지 예측

        Args:
            data: 예측 데이터

        Returns:
            (predictions, anomaly_scores)
            predictions: 1 = 정상, -1 = 이상
            anomaly_scores: 이상 점수 (낮을수록 이상)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # 피처 추출
        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 예측
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        return predictions, scores

    def detect_anomalies(
        self,
        data: pd.DataFrame,
        return_details: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        이상 데이터 감지 및 상세 정보 반환

        Args:
            data: 입력 데이터
            return_details: 상세 정보 포함 여부

        Returns:
            이상으로 감지된 레코드 목록
        """
        predictions, scores = self.predict(data)

        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # 이상
                anomaly = {
                    "index": i,
                    "anomaly_score": float(score),
                    "severity": self._calculate_severity(score),
                    "detected_at": datetime.utcnow().isoformat(),
                }

                if return_details:
                    # 피처 값 포함
                    anomaly["features"] = {
                        name: float(data.iloc[i][name])
                        for name in self.feature_names
                    }

                    # 기타 메타데이터
                    if "measurement_id" in data.columns:
                        anomaly["measurement_id"] = data.iloc[i]["measurement_id"]
                    if "timestamp" in data.columns:
                        anomaly["timestamp"] = str(data.iloc[i]["timestamp"])

                anomalies.append(anomaly)

        return anomalies

    def _calculate_severity(self, score: float) -> str:
        """
        이상 점수에 따른 심각도 계산

        Args:
            score: 이상 점수 (낮을수록 심각)

        Returns:
            심각도 레벨
        """
        if score < -0.5:
            return "CRITICAL"
        elif score < -0.3:
            return "MAJOR"
        elif score < -0.1:
            return "MINOR"
        else:
            return "WARNING"

    def save_model(self, path: Optional[str] = None) -> str:
        """모델 저장"""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        if path is None:
            path = Path(config.prediction.model_save_path) / f"isolation_forest_{self.equipment_id}.pkl"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "equipment_id": self.equipment_id,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "model": self.model,
            "scaler": self.scaler,
            "training_timestamp": self.training_timestamp,
            "training_samples": self.training_samples,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")
        return str(path)

    @classmethod
    def load_model(cls, path: str) -> "IsolationForestDetector":
        """모델 로드"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        detector = cls(
            equipment_id=model_data["equipment_id"],
            feature_names=model_data["feature_names"],
            contamination=model_data["contamination"],
            n_estimators=model_data["n_estimators"],
        )

        detector.model = model_data["model"]
        detector.scaler = model_data["scaler"]
        detector.training_timestamp = model_data["training_timestamp"]
        detector.training_samples = model_data["training_samples"]
        detector.is_trained = True

        logger.info(f"Model loaded from {path}")
        return detector


class MultiEquipmentIsolationForest:
    """다중 설비 Isolation Forest 관리자"""

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.detectors: Dict[str, IsolationForestDetector] = {}

    def get_or_create_detector(self, equipment_id: str) -> IsolationForestDetector:
        """설비별 감지기 조회 또는 생성"""
        if equipment_id not in self.detectors:
            self.detectors[equipment_id] = IsolationForestDetector(
                equipment_id=equipment_id,
                feature_names=self.feature_names,
            )
        return self.detectors[equipment_id]

    def train_all(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        모든 설비에 대해 학습

        Args:
            data: 전체 데이터 (equipment_id 컬럼 포함)

        Returns:
            설비별 학습 결과
        """
        results = {}

        for equipment_id, group in data.groupby("equipment_id"):
            if len(group) < config.anomaly.min_samples_for_training:
                logger.warning(
                    f"Not enough samples for {equipment_id}: "
                    f"{len(group)} < {config.anomaly.min_samples_for_training}"
                )
                continue

            detector = self.get_or_create_detector(equipment_id)
            metrics = detector.train(group)
            results[equipment_id] = metrics

        return results

    def detect_all(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        모든 설비에 대해 이상 감지

        Args:
            data: 전체 데이터

        Returns:
            감지된 모든 이상
        """
        all_anomalies = []

        for equipment_id, group in data.groupby("equipment_id"):
            detector = self.detectors.get(equipment_id)

            if detector is None or not detector.is_trained:
                logger.warning(f"No trained model for equipment {equipment_id}")
                continue

            anomalies = detector.detect_anomalies(group)
            for anomaly in anomalies:
                anomaly["equipment_id"] = equipment_id

            all_anomalies.extend(anomalies)

        return all_anomalies

    def save_all_models(self, base_path: str = None) -> Dict[str, str]:
        """모든 모델 저장"""
        base_path = base_path or config.prediction.model_save_path
        paths = {}

        for equipment_id, detector in self.detectors.items():
            if detector.is_trained:
                path = detector.save_model(
                    Path(base_path) / f"isolation_forest_{equipment_id}.pkl"
                )
                paths[equipment_id] = path

        return paths

    def load_all_models(self, base_path: str = None) -> int:
        """모든 모델 로드"""
        base_path = Path(base_path or config.prediction.model_save_path)
        loaded_count = 0

        for model_file in base_path.glob("isolation_forest_*.pkl"):
            try:
                detector = IsolationForestDetector.load_model(str(model_file))
                self.detectors[detector.equipment_id] = detector
                loaded_count += 1
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")

        return loaded_count
