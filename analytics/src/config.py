"""
Analytics Engine Configuration
"""
import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DatabaseConfig:
    """PostgreSQL (AGE) 연결 설정"""
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "ontology")
    password: str = os.getenv("POSTGRES_PASSWORD", "ontology123")
    database: str = os.getenv("POSTGRES_DB", "manufacturing")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class TimescaleConfig:
    """TimescaleDB 연결 설정"""
    host: str = os.getenv("TIMESCALE_HOST", "timescaledb")
    port: int = int(os.getenv("TIMESCALE_PORT", "5432"))
    user: str = os.getenv("TIMESCALE_USER", "timescale")
    password: str = os.getenv("TIMESCALE_PASSWORD", "timescale123")
    database: str = os.getenv("TIMESCALE_DB", "measurements")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis 연결 설정"""
    host: str = os.getenv("REDIS_HOST", "redis")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD", "redis123")
    db: int = int(os.getenv("REDIS_DB", "0"))


@dataclass
class KafkaConfig:
    """Kafka 연결 설정"""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")

    # 소비 토픽
    enriched_measurement_topic: str = "flink.measurements.enriched"
    spc_analyzed_topic: str = "flink.spc.analyzed"
    cep_alerts_topic: str = "flink.cep.alerts"

    # 발행 토픽
    anomaly_alerts_topic: str = "analytics.anomaly.alerts"
    prediction_results_topic: str = "analytics.prediction.results"


@dataclass
class AnomalyDetectionConfig:
    """이상 감지 설정"""
    # Isolation Forest
    isolation_forest_n_estimators: int = 100
    isolation_forest_contamination: float = float(os.getenv("IF_CONTAMINATION", "0.1"))
    isolation_forest_max_samples: str = "auto"

    # Autoencoder
    autoencoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16, 32, 64])
    autoencoder_latent_dim: int = 8
    autoencoder_learning_rate: float = 0.001
    autoencoder_epochs: int = 100
    autoencoder_batch_size: int = 32
    autoencoder_threshold_percentile: float = 95.0

    # 공통
    min_samples_for_training: int = 1000
    retraining_interval_hours: int = 24
    feature_window_size: int = 60  # 분


@dataclass
class SPCConfig:
    """SPC 분석 설정"""
    # 관리도 타입
    chart_types: List[str] = field(default_factory=lambda: ["xbar", "r", "s", "p", "np", "c", "u"])

    # 시그마 레벨
    sigma_level: float = 3.0

    # 히스토리 크기
    history_size: int = 25

    # Cp/Cpk 경고 수준
    cp_warning_threshold: float = 1.33
    cp_critical_threshold: float = 1.0
    cpk_warning_threshold: float = 1.33
    cpk_critical_threshold: float = 1.0


@dataclass
class PredictionConfig:
    """예측 모델 설정"""
    # 설비 고장 예측
    failure_prediction_horizon_hours: int = 24
    failure_prediction_features: List[str] = field(default_factory=lambda: [
        "temperature", "pressure", "vibration", "power", "runtime_hours"
    ])

    # 품질 예측
    quality_prediction_target: str = "yield"
    quality_model_type: str = "random_forest"  # random_forest, gradient_boosting, neural_network

    # 모델 저장
    model_save_path: str = os.getenv("MODEL_SAVE_PATH", "/app/models")


@dataclass
class AnalyticsConfig:
    """전체 분석 설정"""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    timescale: TimescaleConfig = field(default_factory=TimescaleConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    anomaly: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    spc: SPCConfig = field(default_factory=SPCConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)


# 전역 설정 인스턴스
config = AnalyticsConfig()
