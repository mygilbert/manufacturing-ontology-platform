"""
CDC 트랜스포머 설정
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class KafkaConfig:
    """Kafka 연결 설정"""
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "ontology-transformer")
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 500
    session_timeout_ms: int = 30000


@dataclass
class PostgresConfig:
    """PostgreSQL (AGE) 연결 설정"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "ontology")
    password: str = os.getenv("POSTGRES_PASSWORD", "ontology123")
    database: str = os.getenv("POSTGRES_DB", "manufacturing")

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class TimescaleConfig:
    """TimescaleDB 연결 설정"""
    host: str = os.getenv("TIMESCALE_HOST", "localhost")
    port: int = int(os.getenv("TIMESCALE_PORT", "5433"))
    user: str = os.getenv("TIMESCALE_USER", "timescale")
    password: str = os.getenv("TIMESCALE_PASSWORD", "timescale123")
    database: str = os.getenv("TIMESCALE_DB", "measurements")

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis 연결 설정"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD", "redis123")
    db: int = 0


# 토픽 매핑
TOPIC_MAPPING = {
    # FDC 소스 토픽 → 처리 토픽
    "fdc.FDC_PARAM_VALUE": "fdc.measurements.processed",
    "fdc.FDC_ALARM_HISTORY": "fdc.alarms.enriched",
    "fdc.FDC_EQUIPMENT_MASTER": "ontology.objects.equipment",
    "fdc.FDC_RECIPE_MASTER": "ontology.objects.recipe",

    # SPC 소스 토픽 → 처리 토픽
    "spc.SPC_MEASUREMENT": "spc.measurements.analyzed",
    "spc.SPC_ALARM_LOG": "alerts.realtime",
    "spc.SPC_PROCESS_SPEC": "ontology.objects.process",

    # MES 소스 토픽 → 처리 토픽
    "mes.MES_LOT_MASTER": "ontology.objects.lot",
    "mes.MES_WAFER_MASTER": "ontology.objects.wafer",
    "mes.MES_EQUIPMENT": "ontology.objects.equipment",
    "mes.MES_TRACK_IN_OUT": "mes.tracking.events",
}

# 컬럼 매핑 (소스 DB 컬럼 → 온톨로지 속성)
COLUMN_MAPPING = {
    "FDC_PARAM_VALUE": {
        "VALUE_ID": "measurement_id",
        "EQUIP_ID": "equipment_id",
        "CHAMBER_ID": "chamber_id",
        "RECIPE_ID": "recipe_id",
        "LOT_ID": "lot_id",
        "WAFER_ID": "wafer_id",
        "SLOT_NO": "slot_no",
        "PARAM_ID": "param_id",
        "PARAM_NAME": "param_name",
        "PARAM_VALUE": "value",
        "PARAM_UNIT": "unit",
        "COLLECT_TIME": "timestamp",
        "USL": "usl",
        "LSL": "lsl",
        "TARGET": "target",
    },
    "FDC_ALARM_HISTORY": {
        "ALARM_ID": "alarm_id",
        "ALARM_CODE": "alarm_code",
        "ALARM_NAME": "alarm_name",
        "ALARM_LEVEL": "severity",
        "EQUIP_ID": "equipment_id",
        "CHAMBER_ID": "chamber_id",
        "LOT_ID": "lot_id",
        "WAFER_ID": "wafer_id",
        "ALARM_TIME": "occurred_at",
        "ALARM_MSG": "message",
        "ALARM_VALUE": "triggered_value",
        "THRESHOLD": "threshold_value",
    },
    "SPC_MEASUREMENT": {
        "MEAS_ID": "measurement_id",
        "EQUIP_ID": "equipment_id",
        "PROCESS_ID": "process_id",
        "LOT_ID": "lot_id",
        "WAFER_ID": "wafer_id",
        "ITEM_ID": "item_id",
        "ITEM_NAME": "item_name",
        "MEAS_VALUE": "value",
        "UNIT": "unit",
        "MEAS_TIME": "timestamp",
        "USL": "usl",
        "LSL": "lsl",
        "UCL": "ucl",
        "LCL": "lcl",
        "TARGET": "target",
    },
    "MES_LOT_MASTER": {
        "LOT_ID": "lot_id",
        "PRODUCT_ID": "product_code",
        "PRODUCT_NAME": "product_name",
        "LOT_QTY": "quantity",
        "PRIORITY_CODE": "priority",
        "LOT_STATUS": "status",
        "CURR_STEP_ID": "current_step",
        "LOT_START_TIME": "start_time",
        "LOT_END_TIME": "end_time",
        "FAB_ID": "fab_id",
        "ROUTE_ID": "route_id",
    },
    "MES_WAFER_MASTER": {
        "WAFER_ID": "wafer_id",
        "LOT_ID": "lot_id",
        "SLOT_NO": "slot_no",
        "WAFER_STATUS": "status",
        "CREATE_TIME": "created_at",
    },
}
