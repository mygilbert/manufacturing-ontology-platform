"""
품질 예측 모델

공정 파라미터를 기반으로 품질(수율, 불량률 등)을 예측합니다.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append('/app/src')

from config import config

logger = logging.getLogger(__name__)


class QualityPredictor:
    """품질 예측기"""

    def __init__(
        self,
        process_id: str,
        target_name: str = None,
        feature_names: List[str] = None,
        model_type: str = None,
    ):
        self.process_id = process_id
        self.target_name = target_name or config.prediction.quality_prediction_target
        self.feature_names = feature_names or []
        self.model_type = model_type or config.prediction.quality_model_type

        # 모델
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.poly_features: Optional[PolynomialFeatures] = None

        # 메타데이터
        self.is_trained = False
        self.training_timestamp: Optional[datetime] = None
        self.feature_importance: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}

    def _create_model(self):
        """모델 생성"""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "linear":
            return Ridge(alpha=1.0)
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def train(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        use_polynomial: bool = False,
        poly_degree: int = 2,
    ) -> Dict[str, Any]:
        """
        모델 학습

        Args:
            data: 학습 데이터 (feature_names + target_name 컬럼)
            test_size: 테스트 비율
            use_polynomial: 다항 피처 사용 여부
            poly_degree: 다항 차수

        Returns:
            학습 결과 메트릭
        """
        logger.info(f"Training quality prediction model for {self.process_id}")

        # 피처/타겟 검증
        if self.target_name not in data.columns:
            raise ValueError(f"Target column '{self.target_name}' not found")

        # 피처 자동 감지 (지정되지 않은 경우)
        if not self.feature_names:
            self.feature_names = [c for c in data.columns
                                  if c != self.target_name and data[c].dtype in ['int64', 'float64']]

        # 피처/타겟 분리
        X = data[self.feature_names].values
        y = data[self.target_name].values

        # 결측치 처리
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=np.nanmean(y))

        # 스케일링
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 다항 피처 (선택)
        if use_polynomial:
            self.poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_scaled = self.poly_features.fit_transform(X_scaled)

        # 학습/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )

        # 모델 학습
        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        # 평가
        y_pred = self.model.predict(X_test)

        self.metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100),
        }

        # 교차 검증
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        self.metrics["cv_r2_mean"] = float(cv_scores.mean())
        self.metrics["cv_r2_std"] = float(cv_scores.std())

        # 피처 중요도
        if hasattr(self.model, 'feature_importances_'):
            if self.poly_features:
                feature_names = self.poly_features.get_feature_names_out(self.feature_names)
            else:
                feature_names = self.feature_names

            self.feature_importance = dict(zip(
                feature_names,
                self.model.feature_importances_.tolist()
            ))
        elif hasattr(self.model, 'coef_'):
            if self.poly_features:
                feature_names = self.poly_features.get_feature_names_out(self.feature_names)
            else:
                feature_names = self.feature_names

            self.feature_importance = dict(zip(
                feature_names,
                np.abs(self.model.coef_).tolist()
            ))

        # 메타데이터 업데이트
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()

        result = {
            "process_id": self.process_id,
            "target": self.target_name,
            "training_samples": len(X),
            "feature_count": X_scaled.shape[1],
            "training_timestamp": self.training_timestamp.isoformat(),
            "metrics": self.metrics,
            "top_features": dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]) if self.feature_importance else {},
        }

        logger.info(f"Training completed. R²: {self.metrics['r2']:.4f}")
        return result

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        품질 예측

        Args:
            data: 예측 데이터

        Returns:
            예측값 배열
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)

        X_scaled = self.scaler.transform(X)

        if self.poly_features:
            X_scaled = self.poly_features.transform(X_scaled)

        return self.model.predict(X_scaled)

    def predict_with_interval(
        self,
        data: pd.DataFrame,
        confidence: float = 0.95,
    ) -> List[Dict[str, Any]]:
        """
        신뢰 구간과 함께 예측

        Args:
            data: 예측 데이터
            confidence: 신뢰 수준

        Returns:
            예측 결과 (값, 하한, 상한)
        """
        predictions = self.predict(data)

        # 간단한 신뢰 구간 (학습 RMSE 기반)
        rmse = self.metrics.get("rmse", 0)
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

        results = []
        for i, pred in enumerate(predictions):
            result = {
                "index": i,
                "predicted_value": float(pred),
                "lower_bound": float(pred - z_score * rmse),
                "upper_bound": float(pred + z_score * rmse),
                "confidence": confidence,
            }

            # 메타데이터
            if "lot_id" in data.columns:
                result["lot_id"] = data.iloc[i]["lot_id"]
            if "wafer_id" in data.columns:
                result["wafer_id"] = data.iloc[i]["wafer_id"]

            results.append(result)

        return results

    def analyze_feature_impact(
        self,
        data: pd.DataFrame,
        feature_name: str,
        n_points: int = 20,
    ) -> Dict[str, Any]:
        """
        특정 피처의 품질 영향도 분석 (부분 의존성)

        Args:
            data: 기준 데이터
            feature_name: 분석할 피처
            n_points: 분석 포인트 수

        Returns:
            피처 값에 따른 예측 변화
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not in model features")

        # 피처 범위
        feature_values = data[feature_name].values
        min_val, max_val = feature_values.min(), feature_values.max()
        analysis_values = np.linspace(min_val, max_val, n_points)

        # 부분 의존성 계산
        predictions = []
        base_data = data.copy()

        for val in analysis_values:
            base_data[feature_name] = val
            pred = self.predict(base_data).mean()
            predictions.append(pred)

        return {
            "feature_name": feature_name,
            "feature_values": analysis_values.tolist(),
            "predictions": predictions,
            "impact_range": float(max(predictions) - min(predictions)),
            "correlation": float(np.corrcoef(analysis_values, predictions)[0, 1]),
        }

    def find_optimal_conditions(
        self,
        data: pd.DataFrame,
        target_value: float,
        variable_features: List[str] = None,
    ) -> Dict[str, Any]:
        """
        목표 품질을 달성하기 위한 최적 조건 탐색

        Args:
            data: 기준 데이터
            target_value: 목표 품질값
            variable_features: 조정 가능한 피처

        Returns:
            최적 조건
        """
        from scipy.optimize import minimize

        if variable_features is None:
            variable_features = self.feature_names[:5]  # 상위 5개

        # 기준점
        base_values = data[self.feature_names].mean().to_dict()

        def objective(x):
            test_data = pd.DataFrame([base_values])
            for i, feat in enumerate(variable_features):
                test_data[feat] = x[i]

            pred = self.predict(test_data)[0]
            return (pred - target_value) ** 2

        # 초기값 및 범위
        x0 = [data[f].mean() for f in variable_features]
        bounds = [(data[f].min(), data[f].max()) for f in variable_features]

        # 최적화
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        optimal_conditions = dict(zip(variable_features, result.x.tolist()))

        # 최적 조건에서의 예측
        test_data = pd.DataFrame([base_values])
        for feat, val in optimal_conditions.items():
            test_data[feat] = val
        predicted_value = self.predict(test_data)[0]

        return {
            "target_value": target_value,
            "predicted_value": float(predicted_value),
            "optimal_conditions": optimal_conditions,
            "optimization_success": result.success,
            "gap": float(abs(predicted_value - target_value)),
        }

    def save_model(self, path: Optional[str] = None) -> str:
        """모델 저장"""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        if path is None:
            path = Path(config.prediction.model_save_path) / f"quality_{self.process_id}.pkl"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "process_id": self.process_id,
            "target_name": self.target_name,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler,
            "poly_features": self.poly_features,
            "feature_importance": self.feature_importance,
            "metrics": self.metrics,
            "training_timestamp": self.training_timestamp,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")
        return str(path)

    @classmethod
    def load_model(cls, path: str) -> "QualityPredictor":
        """모델 로드"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        predictor = cls(
            process_id=model_data["process_id"],
            target_name=model_data["target_name"],
            feature_names=model_data["feature_names"],
            model_type=model_data["model_type"],
        )

        predictor.model = model_data["model"]
        predictor.scaler = model_data["scaler"]
        predictor.poly_features = model_data["poly_features"]
        predictor.feature_importance = model_data["feature_importance"]
        predictor.metrics = model_data["metrics"]
        predictor.training_timestamp = model_data["training_timestamp"]
        predictor.is_trained = True

        logger.info(f"Model loaded from {path}")
        return predictor


class YieldOptimizer:
    """수율 최적화"""

    def __init__(self, quality_predictor: QualityPredictor):
        self.predictor = quality_predictor

    def optimize_for_yield(
        self,
        current_conditions: Dict[str, float],
        constraints: Dict[str, Tuple[float, float]] = None,
        target_yield: float = 0.95,
    ) -> Dict[str, Any]:
        """
        수율 최대화를 위한 조건 최적화

        Args:
            current_conditions: 현재 공정 조건
            constraints: 피처별 조정 범위 {feature: (min, max)}
            target_yield: 목표 수율

        Returns:
            최적화된 조건
        """
        from scipy.optimize import minimize

        features = list(current_conditions.keys())

        # 범위 설정
        if constraints is None:
            constraints = {f: (v * 0.8, v * 1.2) for f, v in current_conditions.items()}

        def objective(x):
            conditions = dict(zip(features, x))
            data = pd.DataFrame([conditions])
            pred = self.predictor.predict(data)[0]
            return -pred  # 최대화를 위해 음수

        x0 = [current_conditions[f] for f in features]
        bounds = [constraints.get(f, (v * 0.8, v * 1.2)) for f, v in current_conditions.items()]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        optimized_conditions = dict(zip(features, result.x.tolist()))

        # 개선 정도 계산
        current_yield = self.predictor.predict(pd.DataFrame([current_conditions]))[0]
        optimized_yield = -result.fun

        return {
            "current_yield": float(current_yield),
            "optimized_yield": float(optimized_yield),
            "improvement": float(optimized_yield - current_yield),
            "improvement_pct": float((optimized_yield - current_yield) / current_yield * 100),
            "current_conditions": current_conditions,
            "optimized_conditions": optimized_conditions,
            "changes": {
                f: float(optimized_conditions[f] - current_conditions[f])
                for f in features
            },
        }
