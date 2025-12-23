"""
딥러닝 기반 이상감지 알고리즘

- AutoEncoder
- LSTM-AutoEncoder (시계열 특화)
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# TensorFlow 임포트
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # TensorFlow가 없으면 이 모듈 사용 불가
    raise ImportError("TensorFlow is required for deep learning models. Install with: pip install tensorflow")


class BaseDeepLearningDetector:
    """딥러닝 이상감지 기본 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.params: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
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


class AutoEncoderDetector(BaseDeepLearningDetector):
    """AutoEncoder 기반 이상감지"""

    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_layers: list = [32, 16],
        epochs: int = 50,
        batch_size: int = 64,
        threshold_percentile: float = 95.0,
        learning_rate: float = 0.001
    ):
        """
        Args:
            encoding_dim: 잠재 공간 차원
            hidden_layers: 은닉층 뉴런 수 리스트
            epochs: 학습 에포크
            batch_size: 배치 크기
            threshold_percentile: 이상 판정 임계값 백분위
            learning_rate: 학습률
        """
        super().__init__("AutoEncoder")

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow가 설치되어 있지 않습니다.")

        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.learning_rate = learning_rate
        self.input_dim = None

        self.params = {
            'encoding_dim': encoding_dim,
            'hidden_layers': hidden_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'threshold_percentile': threshold_percentile,
            'learning_rate': learning_rate
        }

    def _build_model(self, input_dim: int):
        """AutoEncoder 모델 구축"""
        self.input_dim = input_dim

        # 인코더
        inputs = layers.Input(shape=(input_dim,))
        x = inputs

        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # 잠재 공간
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoded')(x)

        # 디코더
        x = encoded
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # 출력
        outputs = layers.Dense(input_dim, activation='linear')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 정상 데이터만 사용
        if y is not None:
            y = np.asarray(y)
            normal_mask = y == 0
            if np.sum(normal_mask) > 0:
                X_train = X_scaled[normal_mask]
            else:
                X_train = X_scaled
        else:
            X_train = X_scaled

        # 모델 구축
        self.model = self._build_model(X_scaled.shape[1])

        # 학습
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.model.fit(
            X_train, X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )

        # 임계값 설정 (학습 데이터 기반)
        train_scores = self._calculate_reconstruction_error(X_train)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)

        self.is_fitted = True
        return self

    def _calculate_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """재구성 오차 계산"""
        X_pred = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_pred), axis=1)
        return mse

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수 계산 (재구성 오차)"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)
        scores = self._calculate_reconstruction_error(X_scaled)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)


class LSTMAutoEncoderDetector(BaseDeepLearningDetector):
    """LSTM AutoEncoder 기반 시계열 이상감지"""

    def __init__(
        self,
        sequence_length: int = 30,
        lstm_units: list = [64, 32],
        epochs: int = 50,
        batch_size: int = 64,
        threshold_percentile: float = 95.0,
        learning_rate: float = 0.001
    ):
        """
        Args:
            sequence_length: 시퀀스 길이
            lstm_units: LSTM 유닛 수 리스트
            epochs: 학습 에포크
            batch_size: 배치 크기
            threshold_percentile: 이상 판정 임계값 백분위
            learning_rate: 학습률
        """
        super().__init__("LSTM AutoEncoder")

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow가 설치되어 있지 않습니다.")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self.learning_rate = learning_rate
        self.n_features = None

        self.params = {
            'sequence_length': sequence_length,
            'lstm_units': lstm_units,
            'epochs': epochs,
            'batch_size': batch_size,
            'threshold_percentile': threshold_percentile,
            'learning_rate': learning_rate
        }

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """시계열 데이터를 시퀀스로 변환"""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)

    def _build_model(self, n_features: int):
        """LSTM AutoEncoder 모델 구축"""
        self.n_features = n_features

        # 인코더
        inputs = layers.Input(shape=(self.sequence_length, n_features))
        x = inputs

        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(units, activation='relu', return_sequences=return_sequences)(x)

        # 디코더
        x = layers.RepeatVector(self.sequence_length)(x)

        for units in reversed(self.lstm_units):
            x = layers.LSTM(units, activation='relu', return_sequences=True)(x)

        outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """모델 학습"""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 정상 데이터만 사용
        if y is not None:
            y = np.asarray(y)
            normal_mask = y == 0
            if np.sum(normal_mask) > 0:
                X_train = X_scaled[normal_mask]
            else:
                X_train = X_scaled
        else:
            X_train = X_scaled

        # 시퀀스 생성
        X_seq = self._create_sequences(X_train)

        if len(X_seq) == 0:
            raise ValueError(f"데이터가 너무 짧습니다. 최소 {self.sequence_length}개 이상 필요합니다.")

        # 모델 구축
        self.model = self._build_model(X_scaled.shape[1])

        # 학습
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        self.model.fit(
            X_seq, X_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )

        # 임계값 설정
        train_scores = self._calculate_sequence_error(X_seq)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)

        self.is_fitted = True
        return self

    def _calculate_sequence_error(self, X_seq: np.ndarray) -> np.ndarray:
        """시퀀스 재구성 오차 계산"""
        X_pred = self.model.predict(X_seq, verbose=0)
        mse = np.mean(np.square(X_seq - X_pred), axis=(1, 2))
        return mse

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """이상 점수 계산"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.zeros(len(X))

        # 시퀀스 점수 계산
        seq_scores = self._calculate_sequence_error(X_seq)

        # 원래 길이로 확장 (각 시점에 해당하는 시퀀스 점수의 평균)
        scores = np.zeros(len(X))
        counts = np.zeros(len(X))

        for i, score in enumerate(seq_scores):
            end_idx = i + self.sequence_length
            scores[i:end_idx] += score
            counts[i:end_idx] += 1

        scores = scores / np.maximum(counts, 1)

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """이상 여부 예측"""
        scores = self.predict_score(X)
        return (scores > self.threshold).astype(int)
