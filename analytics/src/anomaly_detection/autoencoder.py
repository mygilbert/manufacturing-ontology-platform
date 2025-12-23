"""
Autoencoder 기반 이상 감지

딥러닝 Autoencoder를 사용하여 비정상 패턴을 탐지합니다.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('/app/src')

from config import config

logger = logging.getLogger(__name__)

# GPU 사용 가능 여부
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoencoderNetwork(nn.Module):
    """Autoencoder 신경망 모델"""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # 인코더 구성
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:len(hidden_dims)//2]:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # 디코더 구성
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims[len(hidden_dims)//2:]:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파"""
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """인코딩"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """디코딩"""
        return self.decoder(z)


class AutoencoderDetector:
    """Autoencoder 이상 감지기"""

    def __init__(
        self,
        equipment_id: str,
        feature_names: List[str],
        hidden_dims: List[int] = None,
        latent_dim: int = None,
        learning_rate: float = None,
        epochs: int = None,
        batch_size: int = None,
        threshold_percentile: float = None,
    ):
        self.equipment_id = equipment_id
        self.feature_names = feature_names

        # 하이퍼파라미터
        self.hidden_dims = hidden_dims or config.anomaly.autoencoder_hidden_dims
        self.latent_dim = latent_dim or config.anomaly.autoencoder_latent_dim
        self.learning_rate = learning_rate or config.anomaly.autoencoder_learning_rate
        self.epochs = epochs or config.anomaly.autoencoder_epochs
        self.batch_size = batch_size or config.anomaly.autoencoder_batch_size
        self.threshold_percentile = threshold_percentile or config.anomaly.autoencoder_threshold_percentile

        # 모델 및 스케일러
        self.model: Optional[AutoencoderNetwork] = None
        self.scaler: Optional[StandardScaler] = None

        # 임계값 (학습 후 설정)
        self.threshold: Optional[float] = None

        # 메타데이터
        self.is_trained = False
        self.training_timestamp: Optional[datetime] = None
        self.training_samples: int = 0
        self.training_loss_history: List[float] = []

    def train(
        self,
        data: pd.DataFrame,
        validation_split: float = 0.2,
    ) -> Dict[str, Any]:
        """
        모델 학습

        Args:
            data: 학습 데이터
            validation_split: 검증 데이터 비율

        Returns:
            학습 결과 메트릭
        """
        logger.info(f"Training Autoencoder for equipment {self.equipment_id}")

        # 피처 추출
        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)

        # 스케일링
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 학습/검증 분할
        n_samples = len(X_scaled)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        X_train = X_scaled[indices[n_val:]]
        X_val = X_scaled[indices[:n_val]]

        # 텐서 변환
        train_tensor = torch.FloatTensor(X_train)
        val_tensor = torch.FloatTensor(X_val)

        train_dataset = TensorDataset(train_tensor, train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # 모델 초기화
        input_dim = len(self.feature_names)
        self.model = AutoencoderNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
        ).to(device)

        # 옵티마이저 및 손실 함수
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # 학습
        self.training_loss_history = []
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_x, _ in train_loader:
                batch_x = batch_x.to(device)

                optimizer.zero_grad()
                x_reconstructed, _ = self.model(batch_x)
                loss = criterion(x_reconstructed, batch_x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 검증
            self.model.eval()
            with torch.no_grad():
                val_tensor_device = val_tensor.to(device)
                val_reconstructed, _ = self.model(val_tensor_device)
                val_loss = criterion(val_reconstructed, val_tensor_device).item()

            self.training_loss_history.append(train_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        # 임계값 계산 (학습 데이터의 재구성 오차 기반)
        self.model.eval()
        with torch.no_grad():
            all_data = torch.FloatTensor(X_scaled).to(device)
            reconstructed, _ = self.model(all_data)
            reconstruction_errors = torch.mean((reconstructed - all_data) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()

        self.threshold = np.percentile(reconstruction_errors, self.threshold_percentile)

        # 메타데이터 업데이트
        self.is_trained = True
        self.training_timestamp = datetime.utcnow()
        self.training_samples = len(X)

        metrics = {
            "equipment_id": self.equipment_id,
            "training_samples": self.training_samples,
            "training_timestamp": self.training_timestamp.isoformat(),
            "final_train_loss": float(train_loss),
            "best_val_loss": float(best_val_loss),
            "threshold": float(self.threshold),
            "threshold_percentile": self.threshold_percentile,
        }

        logger.info(f"Training completed. Metrics: {metrics}")
        return metrics

    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상 탐지 예측

        Args:
            data: 예측 데이터

        Returns:
            (is_anomaly, reconstruction_errors)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # 피처 추출 및 스케일링
        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        # 예측
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            X_reconstructed, _ = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_reconstructed - X_tensor) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()

        is_anomaly = reconstruction_errors > self.threshold

        return is_anomaly, reconstruction_errors

    def detect_anomalies(
        self,
        data: pd.DataFrame,
        return_details: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        이상 데이터 감지 및 상세 정보 반환
        """
        is_anomaly, reconstruction_errors = self.predict(data)

        # 피처별 재구성 오차 계산
        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            X_reconstructed, latent = self.model(X_tensor)
            feature_errors = (X_reconstructed - X_tensor).abs().cpu().numpy()

        anomalies = []
        for i, (is_anom, error) in enumerate(zip(is_anomaly, reconstruction_errors)):
            if is_anom:
                anomaly = {
                    "index": i,
                    "reconstruction_error": float(error),
                    "threshold": float(self.threshold),
                    "severity": self._calculate_severity(error),
                    "detected_at": datetime.utcnow().isoformat(),
                }

                if return_details:
                    # 피처별 기여도
                    anomaly["feature_contributions"] = {
                        name: float(feature_errors[i][j])
                        for j, name in enumerate(self.feature_names)
                    }

                    # 가장 영향 큰 피처
                    top_features = sorted(
                        anomaly["feature_contributions"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    anomaly["top_contributing_features"] = [f[0] for f in top_features]

                    # 원본 값
                    anomaly["features"] = {
                        name: float(data.iloc[i][name])
                        for name in self.feature_names
                    }

                    if "measurement_id" in data.columns:
                        anomaly["measurement_id"] = data.iloc[i]["measurement_id"]
                    if "timestamp" in data.columns:
                        anomaly["timestamp"] = str(data.iloc[i]["timestamp"])

                anomalies.append(anomaly)

        return anomalies

    def _calculate_severity(self, error: float) -> str:
        """재구성 오차 기반 심각도 계산"""
        ratio = error / self.threshold

        if ratio > 3.0:
            return "CRITICAL"
        elif ratio > 2.0:
            return "MAJOR"
        elif ratio > 1.5:
            return "MINOR"
        else:
            return "WARNING"

    def get_latent_representation(self, data: pd.DataFrame) -> np.ndarray:
        """잠재 공간 표현 추출 (시각화 및 클러스터링용)"""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        X = data[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            latent = self.model.encode(X_tensor)
            return latent.cpu().numpy()

    def save_model(self, path: Optional[str] = None) -> str:
        """모델 저장"""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        if path is None:
            path = Path(config.prediction.model_save_path) / f"autoencoder_{self.equipment_id}.pkl"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "equipment_id": self.equipment_id,
            "feature_names": self.feature_names,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "model_state_dict": self.model.state_dict(),
            "scaler": self.scaler,
            "threshold": self.threshold,
            "training_timestamp": self.training_timestamp,
            "training_samples": self.training_samples,
            "training_loss_history": self.training_loss_history,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")
        return str(path)

    @classmethod
    def load_model(cls, path: str) -> "AutoencoderDetector":
        """모델 로드"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        detector = cls(
            equipment_id=model_data["equipment_id"],
            feature_names=model_data["feature_names"],
            hidden_dims=model_data["hidden_dims"],
            latent_dim=model_data["latent_dim"],
        )

        # 네트워크 재구성
        input_dim = len(model_data["feature_names"])
        detector.model = AutoencoderNetwork(
            input_dim=input_dim,
            hidden_dims=model_data["hidden_dims"],
            latent_dim=model_data["latent_dim"],
        ).to(device)
        detector.model.load_state_dict(model_data["model_state_dict"])
        detector.model.eval()

        detector.scaler = model_data["scaler"]
        detector.threshold = model_data["threshold"]
        detector.training_timestamp = model_data["training_timestamp"]
        detector.training_samples = model_data["training_samples"]
        detector.training_loss_history = model_data["training_loss_history"]
        detector.is_trained = True

        logger.info(f"Model loaded from {path}")
        return detector
