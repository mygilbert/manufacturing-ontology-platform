"""
CDC 데이터 변환기 베이스 클래스
"""
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient

from config import KafkaConfig, COLUMN_MAPPING

logger = logging.getLogger(__name__)


class BaseTransformer(ABC):
    """CDC 데이터 변환기 베이스 클래스"""

    def __init__(
        self,
        source_topics: list[str],
        target_topic: str,
        kafka_config: Optional[KafkaConfig] = None,
    ):
        self.source_topics = source_topics
        self.target_topic = target_topic
        self.kafka_config = kafka_config or KafkaConfig()
        self.running = False

        # Consumer 설정
        self.consumer_config = {
            "bootstrap.servers": self.kafka_config.bootstrap_servers,
            "group.id": self.kafka_config.consumer_group,
            "auto.offset.reset": self.kafka_config.auto_offset_reset,
            "enable.auto.commit": self.kafka_config.enable_auto_commit,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": self.kafka_config.session_timeout_ms,
        }

        # Producer 설정
        self.producer_config = {
            "bootstrap.servers": self.kafka_config.bootstrap_servers,
            "acks": "all",
            "retries": 3,
            "linger.ms": 5,
            "batch.size": 16384,
        }

        self.consumer: Optional[Consumer] = None
        self.producer: Optional[Producer] = None

    def start(self):
        """트랜스포머 시작"""
        logger.info(f"Starting transformer: {self.__class__.__name__}")
        logger.info(f"Source topics: {self.source_topics}")
        logger.info(f"Target topic: {self.target_topic}")

        self.consumer = Consumer(self.consumer_config)
        self.producer = Producer(self.producer_config)

        self.consumer.subscribe(self.source_topics)
        self.running = True

        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """트랜스포머 중지"""
        logger.info("Stopping transformer...")
        self.running = False

        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()

    def _run_loop(self):
        """메인 처리 루프"""
        while self.running:
            msg = self.consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(f"End of partition: {msg.topic()}[{msg.partition()}]")
                else:
                    raise KafkaException(msg.error())
                continue

            try:
                # 메시지 파싱
                key = msg.key().decode("utf-8") if msg.key() else None
                value = json.loads(msg.value().decode("utf-8"))
                topic = msg.topic()

                # 테이블 이름 추출
                table_name = self._extract_table_name(topic)

                # 컬럼 매핑 적용
                mapped_value = self._apply_column_mapping(value, table_name)

                # 변환 수행
                transformed = self.transform(mapped_value, table_name)

                if transformed:
                    # 결과 발행
                    self._produce(key, transformed)

                # 오프셋 커밋
                self.consumer.commit(asynchronous=False)

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                # DLQ로 전송하거나 재시도 로직 추가 가능

    def _extract_table_name(self, topic: str) -> str:
        """토픽에서 테이블 이름 추출"""
        # 예: "fdc.FDC_PARAM_VALUE" → "FDC_PARAM_VALUE"
        parts = topic.split(".")
        return parts[-1] if parts else topic

    def _apply_column_mapping(self, value: dict, table_name: str) -> dict:
        """컬럼 매핑 적용"""
        mapping = COLUMN_MAPPING.get(table_name, {})
        if not mapping:
            return value

        result = {}
        for source_col, target_col in mapping.items():
            if source_col in value:
                result[target_col] = value[source_col]

        # 매핑되지 않은 필드 유지 (source_system 등)
        for key, val in value.items():
            if key not in mapping and key not in result:
                result[key] = val

        return result

    def _produce(self, key: Optional[str], value: dict):
        """변환된 메시지 발행"""
        self.producer.produce(
            topic=self.target_topic,
            key=key.encode("utf-8") if key else None,
            value=json.dumps(value, default=self._json_serializer).encode("utf-8"),
            callback=self._delivery_callback,
        )
        self.producer.poll(0)

    def _delivery_callback(self, err, msg):
        """메시지 발행 콜백"""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()}[{msg.partition()}]")

    def _json_serializer(self, obj):
        """JSON 직렬화 헬퍼"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @abstractmethod
    def transform(self, record: dict, table_name: str) -> Optional[dict]:
        """
        레코드 변환 (서브클래스에서 구현)

        Args:
            record: 입력 레코드 (컬럼 매핑 적용됨)
            table_name: 소스 테이블 이름

        Returns:
            변환된 레코드 또는 None (필터링)
        """
        pass
