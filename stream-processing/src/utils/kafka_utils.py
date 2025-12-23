"""
Kafka 유틸리티
"""
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    KafkaSource,
    KafkaSink,
    KafkaRecordSerializationSchema,
    KafkaOffsetsInitializer,
    DeliveryGuarantee,
)
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common import WatermarkStrategy, Duration

from config import kafka_config, flink_config


def create_kafka_source(
    topics: list[str],
    group_id: Optional[str] = None,
) -> KafkaSource:
    """Kafka 소스 생성"""
    return (
        KafkaSource.builder()
        .set_bootstrap_servers(kafka_config.bootstrap_servers)
        .set_topics(*topics)
        .set_group_id(group_id or kafka_config.group_id)
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )


def create_kafka_sink(topic: str) -> KafkaSink:
    """Kafka 싱크 생성"""
    return (
        KafkaSink.builder()
        .set_bootstrap_servers(kafka_config.bootstrap_servers)
        .set_record_serializer(
            KafkaRecordSerializationSchema.builder()
            .set_topic(topic)
            .set_value_serialization_schema(SimpleStringSchema())
            .build()
        )
        .set_delivery_guarantee(DeliveryGuarantee.AT_LEAST_ONCE)
        .build()
    )


def get_watermark_strategy() -> WatermarkStrategy:
    """워터마크 전략 생성"""
    return (
        WatermarkStrategy
        .for_bounded_out_of_orderness(
            Duration.of_millis(flink_config.max_out_of_orderness)
        )
        .with_idleness(Duration.of_minutes(1))
    )


def parse_json_message(message: str) -> Optional[Dict[str, Any]]:
    """JSON 메시지 파싱"""
    try:
        return json.loads(message)
    except json.JSONDecodeError:
        return None


def serialize_to_json(obj: Any) -> str:
    """JSON 직렬화"""
    def default_serializer(o):
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return o.__dict__
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    return json.dumps(obj, default=default_serializer)


def parse_timestamp(value: Any) -> Optional[datetime]:
    """타임스탬프 파싱"""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        try:
            # ISO 형식
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass

    if isinstance(value, (int, float)):
        # Unix timestamp (밀리초)
        if value > 1e12:
            return datetime.fromtimestamp(value / 1000)
        else:
            return datetime.fromtimestamp(value)

    return None
