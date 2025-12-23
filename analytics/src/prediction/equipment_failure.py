"""
설비 고장 예측 모델

센서 데이터를 기반으로 설비 고장을 사전에 예측합니다.
"""
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

import sys
sys.path.append('/app/src')

from config import config

logger = logging.getLogger(__name__)


class EquipmentFailurePredictor:
    """설비 고장 예측기"""

    def __init__(
        self,
        equipment_id: str,
        prediction_horizon_hours: int = None,
        feature_names: List[str] = None,
        model_type: str = "random_forest",
    ):
        self.equipment_id = equipment_id
        self.prediction_horizon = prediction_horizon_hours or config.prediction.failure_prediction_horizon_hours
        self.feature_names = feature_names or config.prediction.failure_prediction_features
        self.model_type = model_type

        # 모델
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None

        # 메타데이터
        self.is_trained = False
        self.training_timestamp: Optional[datetime] = None
        self.feature_importance: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}

    def _create_model(self):
        """모델 생성"""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_features(
        self,
        data: pd.DataFrame,
        window_size: int = 60,
    ) -> pd.DataFrame:
        """
        시계열 데이터로부터 피처 생성

        Args:
            data: 원본 데이터 (timestamp, feature columns)
            window_size: 윈도우 크기 (분)

        Returns:
            피처 DataFrame
        """
        features = pd.DataFrame()

        for col in self.feature_names:
            if col not in data.columns:
                continue

            values = data[col].values

            # 기본 통계
            features[f"{col}_mean"] = pd.Series(values).rolling(window_size).mean()
            features[f"{col}_std"] = pd.Series(values).rolling(window_size).std()
            features[f"{col}_min"] = pd.Series(values).rolling(window_size).min()
            features[f"{col}_max"] = pd.Series(values).rolling(window_size).max()

            # 범위
            features[f"{col}_range"] = features[f"{col}_max"] - features[f"{col}_min"]

            # 변화율
            features[f"{col}_diff"] = pd.Series(values).diff()
            features[f"{col}_pct_change"] = pd.Series(values).pct_change()

            # 이동 평균 대비
            ma = pd.Series(values).rolling(window_size).mean()
            features[f"{col}_ma_ratio"] = values / ma.replace(0, np.nan)

            # 트렌드 (기울기)
            features[f"{col}_trend"] = (
                pd.Series(values)
                .rolling(window_size)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            )

        # 결측치 제거
        features = features.dropna()

        return features

    def create_labels(
        self,
        data: pd.DataFrame,
        failure_column: str = "failure",
        timestamp_column: str = "timestamp",
    ) -> np.ndarray:
        """
        예측 라벨 생성 (향후 N시간 내 고장 여부)

        Args:
            data: 데이터
            failure_column: 고장 여부 컬럼
            timestamp_column: 타임스탬프 컬럼

        Returns:
            라벨 배열
        """
        labels = np.zeros(len(data))

        failure_times = data[data[failure_column] == 1][timestamp_column].values

        for i, row in data.iterrows():
            current_time = row[timestamp_column]

            # 향후 horizon 내에 고장이 있는지 확인
            for failure_time in failure_times:
                time_to_failure = (failure_time - current_time) / np.timedelta64(1, 'h')

                if 0 < time_to_failure <= self.prediction_horizon:
                    labels[i] = 1
                    break

        return labels

    def train(
        self,
        data: pd.DataFrame,
        failure_column: str = "failure",
        timestamp_column: str = "timestamp",
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        모델 학습

        Args:
            data: 학습 데이터
            failure_column: 고장 컬럼
            timestamp_column: 타임스탬프 컬럼
            test_size: 테스트 데이터 비율

        Returns:
            학습 결과 메트릭
        """
        logger.info(f"Training failure prediction model for {self.equipment_id}")

        # 피처 준비
        features = self.prepare_features(data)

        # 데이터 정렬
        data = data.iloc[-len(features):]

        # 라벨 생성
        labels = self.create_labels(data, failure_column, timestamp_column)
        labels = labels[-len(features):]

        # 피처 컬럼 저장
        self.feature_columns = features.columns.tolist()

        # 스케일링
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)
        y = labels

        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if y.sum() > 1 else None
        )

        # 모델 학습
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # 평가
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else y_pred

        # 메트릭 계산
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(auc),
        }

        # 피처 중요도
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_.tolist()
            ))

        # 메타데이터 업데이트
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()

        result = {
            "equipment_id": self.equipment_id,
            "training_samples": len(X),
            "positive_samples": int(y.sum()),
            "negative_samples": int(len(y) - y.sum()),
            "training_timestamp": self.training_timestamp.isoformat(),
            "metrics": self.metrics,
            "top_features": dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
        }

        logger.info(f"Training completed. AUC: {self.metrics['auc']:.4f}")
        return result

    def predict(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        고장 예측

        Args:
            data: 예측 데이터
            threshold: 분류 임계값

        Returns:
            (predictions, probabilities)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # 피처 준비
        features = self.prepare_features(data)

        # 스케일링
        X = self.scaler.transform(features[self.feature_columns].values)

        # 예측
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        return predictions, probabilities

    def predict_with_details(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        상세 예측 결과 반환
        """
        predictions, probabilities = self.predict(data, threshold)

        features = self.prepare_features(data)
        data = data.iloc[-len(features):]

        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred == 1:
                result = {
                    "index": i,
                    "failure_predicted": True,
                    "probability": float(prob),
                    "predicted_at": datetime.utcnow().isoformat(),
                    "prediction_horizon_hours": self.prediction_horizon,
                    "risk_level": self._calculate_risk_level(prob),
                }

                # 주요 기여 피처
                if self.feature_importance:
                    result["top_contributing_features"] = list(
                        sorted(self.feature_importance.keys(),
                               key=lambda x: self.feature_importance[x],
                               reverse=True)[:5]
                    )

                if "timestamp" in data.columns:
                    result["timestamp"] = str(data.iloc[i]["timestamp"])

                results.append(result)

        return results

    def _calculate_risk_level(self, probability: float) -> str:
        """확률 기반 위험도 계산"""
        if probability >= 0.8:
            return "CRITICAL"
        elif probability >= 0.6:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def save_model(self, path: Optional[str] = None) -> str:
        """모델 저장"""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        if path is None:
            path = Path(config.prediction.model_save_path) / f"failure_{self.equipment_id}.pkl"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "equipment_id": self.equipment_id,
            "prediction_horizon": self.prediction_horizon,
            "feature_names": self.feature_names,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "feature_importance": self.feature_importance,
            "metrics": self.metrics,
            "training_timestamp": self.training_timestamp,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")
        return str(path)

    @classmethod
    def load_model(cls, path: str) -> "EquipmentFailurePredictor":
        """모델 로드"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        predictor = cls(
            equipment_id=model_data["equipment_id"],
            prediction_horizon_hours=model_data["prediction_horizon"],
            feature_names=model_data["feature_names"],
            model_type=model_data["model_type"],
        )

        predictor.feature_columns = model_data["feature_columns"]
        predictor.model = model_data["model"]
        predictor.scaler = model_data["scaler"]
        predictor.feature_importance = model_data["feature_importance"]
        predictor.metrics = model_data["metrics"]
        predictor.training_timestamp = model_data["training_timestamp"]
        predictor.is_trained = True

        logger.info(f"Model loaded from {path}")
        return predictor


class RemainingUsefulLifePredictor:
    """잔여 수명 예측 (RUL)"""

    def __init__(self, equipment_id: str):
        self.equipment_id = equipment_id
        self.degradation_model = None
        self.failure_threshold = None

    def fit_degradation_curve(
        self,
        data: pd.DataFrame,
        health_indicator: str,
        timestamp_column: str = "timestamp",
    ) -> Dict[str, Any]:
        """
        열화 곡선 피팅

        Args:
            data: 시계열 데이터
            health_indicator: 건강 지표 컬럼
            timestamp_column: 타임스탬프 컬럼

        Returns:
            피팅 결과
        """
        # 시간 인덱스 계산 (운전 시간)
        times = (data[timestamp_column] - data[timestamp_column].min()).dt.total_seconds() / 3600
        values = data[health_indicator].values

        # 선형 회귀 (단순화)
        from scipy import optimize

        def degradation_func(t, a, b, c):
            return a * np.exp(b * t) + c

        try:
            popt, _ = optimize.curve_fit(
                degradation_func, times, values,
                p0=[0.1, 0.001, values.min()],
                maxfev=5000
            )
            self.degradation_model = lambda t: degradation_func(t, *popt)

            return {
                "equipment_id": self.equipment_id,
                "model_params": popt.tolist(),
                "r_squared": self._calculate_r_squared(times, values, self.degradation_model),
            }
        except Exception as e:
            logger.error(f"Failed to fit degradation curve: {e}")
            return {"error": str(e)}

    def _calculate_r_squared(self, x, y, model):
        """R² 계산"""
        y_pred = model(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def predict_rul(
        self,
        current_time: float,
        current_value: float,
    ) -> Dict[str, Any]:
        """
        잔여 수명 예측

        Args:
            current_time: 현재 운전 시간 (hours)
            current_value: 현재 건강 지표 값

        Returns:
            RUL 예측 결과
        """
        if self.degradation_model is None:
            raise RuntimeError("Degradation model not fitted.")

        if self.failure_threshold is None:
            raise RuntimeError("Failure threshold not set.")

        # 이진 탐색으로 고장 시점 찾기
        t_failure = current_time
        max_time = current_time + 10000  # 최대 예측 범위

        while t_failure < max_time:
            if self.degradation_model(t_failure) >= self.failure_threshold:
                break
            t_failure += 1

        rul_hours = t_failure - current_time

        return {
            "equipment_id": self.equipment_id,
            "current_time_hours": current_time,
            "current_health_value": current_value,
            "predicted_failure_time_hours": t_failure,
            "remaining_useful_life_hours": rul_hours,
            "remaining_useful_life_days": rul_hours / 24,
            "confidence": 0.8 if rul_hours < 1000 else 0.6,  # 단순 신뢰도
        }
