"""
Flink 스트림 처리 설정
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KafkaConfig:
    """Kafka 연결 설정"""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    group_id: str = os.getenv("KAFKA_GROUP_ID", "flink-stream-processor")

    # 소스 토픽
    fdc_measurement_topic: str = "fdc.FDC_PARAM_VALUE"
    fdc_alarm_topic: str = "fdc.FDC_ALARM_HISTORY"
    fdc_equipment_topic: str = "fdc.FDC_EQUIPMENT_MASTER"
    spc_measurement_topic: str = "spc.SPC_MEASUREMENT"
    mes_lot_topic: str = "mes.MES_LOT_MASTER"
    mes_tracking_topic: str = "mes.MES_TRACK_IN_OUT"

    # 싱크 토픽
    enriched_measurement_topic: str = "flink.measurements.enriched"
    spc_analyzed_topic: str = "flink.spc.analyzed"
    cep_alerts_topic: str = "flink.cep.alerts"
    aggregated_stats_topic: str = "flink.stats.aggregated"


@dataclass
class PostgresConfig:
    """PostgreSQL (AGE) 연결 설정"""
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "ontology")
    password: str = os.getenv("POSTGRES_PASSWORD", "ontology123")
    database: str = os.getenv("POSTGRES_DB", "manufacturing")

    @property
    def jdbc_url(self) -> str:
        return f"jdbc:postgresql://{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis 연결 설정 (상태 캐시)"""
    host: str = os.getenv("REDIS_HOST", "redis")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD", "redis123")
    db: int = 0


@dataclass
class FlinkConfig:
    """Flink 실행 설정"""
    parallelism: int = int(os.getenv("FLINK_PARALLELISM", "2"))
    checkpoint_interval: int = int(os.getenv("FLINK_CHECKPOINT_INTERVAL", "60000"))  # 1분
    checkpoint_timeout: int = int(os.getenv("FLINK_CHECKPOINT_TIMEOUT", "120000"))  # 2분
    state_backend: str = os.getenv("FLINK_STATE_BACKEND", "rocksdb")  # filesystem, rocksdb

    # 워터마크
    watermark_interval: int = int(os.getenv("FLINK_WATERMARK_INTERVAL", "1000"))  # 1초
    max_out_of_orderness: int = int(os.getenv("FLINK_MAX_OUT_OF_ORDERNESS", "5000"))  # 5초

    # 윈도우
    tumbling_window_size: int = int(os.getenv("FLINK_TUMBLING_WINDOW_SIZE", "300000"))  # 5분
    sliding_window_size: int = int(os.getenv("FLINK_SLIDING_WINDOW_SIZE", "60000"))  # 1분
    sliding_window_slide: int = int(os.getenv("FLINK_SLIDING_WINDOW_SLIDE", "10000"))  # 10초


@dataclass
class CEPConfig:
    """CEP 규칙 설정"""
    # 임계값 초과 패턴
    threshold_violation_count: int = int(os.getenv("CEP_THRESHOLD_VIOLATION_COUNT", "3"))
    threshold_violation_window: int = int(os.getenv("CEP_THRESHOLD_VIOLATION_WINDOW", "180000"))  # 3분

    # 연속 알람 패턴
    consecutive_alarm_count: int = int(os.getenv("CEP_CONSECUTIVE_ALARM_COUNT", "5"))
    consecutive_alarm_window: int = int(os.getenv("CEP_CONSECUTIVE_ALARM_WINDOW", "300000"))  # 5분

    # 드리프트 감지
    drift_detection_window: int = int(os.getenv("CEP_DRIFT_DETECTION_WINDOW", "600000"))  # 10분
    drift_threshold_sigma: float = float(os.getenv("CEP_DRIFT_THRESHOLD_SIGMA", "2.0"))


@dataclass
class SPCConfig:
    """SPC 분석 설정"""
    # Western Electric Rules
    rule1_enabled: bool = True  # 1점 관리한계 초과
    rule2_enabled: bool = True  # 연속 9점 한쪽
    rule3_enabled: bool = True  # 연속 6점 증가/감소
    rule4_enabled: bool = True  # 연속 14점 교대
    rule5_enabled: bool = True  # 3점 중 2점 2시그마 초과
    rule6_enabled: bool = True  # 5점 중 4점 1시그마 초과
    rule7_enabled: bool = True  # 15점 1시그마 이내
    rule8_enabled: bool = True  # 8점 1시그마 밖

    # 히스토리 윈도우
    history_window_size: int = 25  # 관리도용 데이터 포인트 수


# 전역 설정 인스턴스
kafka_config = KafkaConfig()
postgres_config = PostgresConfig()
redis_config = RedisConfig()
flink_config = FlinkConfig()
cep_config = CEPConfig()
spc_config = SPCConfig()
