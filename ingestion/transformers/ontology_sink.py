"""
온톨로지 싱크 (Kafka → PostgreSQL AGE)

변환된 데이터를 그래프 DB에 적재
"""
import json
import logging
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from confluent_kafka import Consumer, KafkaError, KafkaException

from config import KafkaConfig, PostgresConfig, TimescaleConfig

logger = logging.getLogger(__name__)


class OntologySink:
    """온톨로지 DB 싱크"""

    def __init__(
        self,
        source_topics: list[str],
        kafka_config: Optional[KafkaConfig] = None,
        postgres_config: Optional[PostgresConfig] = None,
        timescale_config: Optional[TimescaleConfig] = None,
    ):
        self.source_topics = source_topics
        self.kafka_config = kafka_config or KafkaConfig()
        self.postgres_config = postgres_config or PostgresConfig()
        self.timescale_config = timescale_config or TimescaleConfig()

        self.consumer_config = {
            "bootstrap.servers": self.kafka_config.bootstrap_servers,
            "group.id": f"{self.kafka_config.consumer_group}-sink",
            "auto.offset.reset": self.kafka_config.auto_offset_reset,
            "enable.auto.commit": False,
        }

        self.consumer: Optional[Consumer] = None
        self.pg_conn = None
        self.ts_conn = None
        self.running = False

        # 배치 설정
        self.batch_size = 100
        self.batch_timeout_ms = 5000

    def start(self):
        """싱크 시작"""
        logger.info(f"Starting ontology sink")
        logger.info(f"Source topics: {self.source_topics}")

        # DB 연결
        self.pg_conn = psycopg2.connect(self.postgres_config.dsn)
        self.pg_conn.autocommit = False

        self.ts_conn = psycopg2.connect(self.timescale_config.dsn)
        self.ts_conn.autocommit = False

        # AGE 설정
        with self.pg_conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
        self.pg_conn.commit()

        # Kafka Consumer
        self.consumer = Consumer(self.consumer_config)
        self.consumer.subscribe(self.source_topics)
        self.running = True

        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """싱크 중지"""
        logger.info("Stopping ontology sink...")
        self.running = False

        if self.consumer:
            self.consumer.close()
        if self.pg_conn:
            self.pg_conn.close()
        if self.ts_conn:
            self.ts_conn.close()

    def _run_loop(self):
        """메인 처리 루프"""
        batch = []
        last_flush = datetime.utcnow()

        while self.running:
            msg = self.consumer.poll(timeout=1.0)

            if msg is None:
                # 타임아웃 체크
                elapsed = (datetime.utcnow() - last_flush).total_seconds() * 1000
                if batch and elapsed > self.batch_timeout_ms:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = datetime.utcnow()
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            try:
                topic = msg.topic()
                value = json.loads(msg.value().decode("utf-8"))
                batch.append((topic, value))

                if len(batch) >= self.batch_size:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = datetime.utcnow()

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

    def _flush_batch(self, batch: list[tuple[str, dict]]):
        """배치 적재"""
        if not batch:
            return

        logger.info(f"Flushing batch of {len(batch)} records")

        try:
            # 토픽별로 그룹화
            by_topic = {}
            for topic, value in batch:
                if topic not in by_topic:
                    by_topic[topic] = []
                by_topic[topic].append(value)

            # 토픽별 처리
            for topic, records in by_topic.items():
                if "measurement" in topic:
                    self._sink_measurements(records)
                elif "alarm" in topic:
                    self._sink_alarms(records)
                elif "objects.equipment" in topic:
                    self._sink_equipment(records)
                elif "objects.lot" in topic:
                    self._sink_lot(records)
                elif "objects.wafer" in topic:
                    self._sink_wafer(records)

            # 커밋
            self.pg_conn.commit()
            self.ts_conn.commit()
            self.consumer.commit(asynchronous=False)

        except Exception as e:
            logger.error(f"Error flushing batch: {e}", exc_info=True)
            self.pg_conn.rollback()
            self.ts_conn.rollback()

    def _sink_measurements(self, records: list[dict]):
        """측정 데이터 적재 (TimescaleDB)"""
        if not records:
            return

        # FDC 측정값
        fdc_records = [r for r in records if r.get("source_system") == "FDC"]
        if fdc_records:
            with self.ts_conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO fdc_measurements (
                        time, measurement_id, equipment_id, chamber_id, recipe_id,
                        lot_id, wafer_id, slot_no, param_id, param_name,
                        value, unit, usl, lsl, target, status
                    ) VALUES %s
                    ON CONFLICT (time, measurement_id) DO UPDATE SET
                        value = EXCLUDED.value,
                        status = EXCLUDED.status
                    """,
                    [
                        (
                            r.get("timestamp"),
                            r.get("measurement_id"),
                            r.get("equipment_id"),
                            r.get("chamber_id"),
                            r.get("recipe_id"),
                            r.get("lot_id"),
                            r.get("wafer_id"),
                            r.get("slot_no"),
                            r.get("param_id"),
                            r.get("param_name"),
                            r.get("value"),
                            r.get("unit"),
                            r.get("usl"),
                            r.get("lsl"),
                            r.get("target"),
                            r.get("status"),
                        )
                        for r in fdc_records
                    ],
                )

        # SPC 측정값
        spc_records = [r for r in records if r.get("source_system") == "SPC"]
        if spc_records:
            with self.ts_conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO spc_measurements (
                        time, measurement_id, equipment_id, process_id,
                        lot_id, wafer_id, item_id, item_name,
                        value, unit, usl, lsl, ucl, lcl, target,
                        subgroup_id, subgroup_size,
                        x_bar, range_val, std_dev, cp, cpk,
                        status, rule_violations
                    ) VALUES %s
                    ON CONFLICT (time, measurement_id) DO UPDATE SET
                        value = EXCLUDED.value,
                        status = EXCLUDED.status,
                        rule_violations = EXCLUDED.rule_violations
                    """,
                    [
                        (
                            r.get("timestamp"),
                            r.get("measurement_id"),
                            r.get("equipment_id"),
                            r.get("process_id"),
                            r.get("lot_id"),
                            r.get("wafer_id"),
                            r.get("item_id"),
                            r.get("item_name"),
                            r.get("value"),
                            r.get("unit"),
                            r.get("usl"),
                            r.get("lsl"),
                            r.get("ucl"),
                            r.get("lcl"),
                            r.get("target"),
                            r.get("subgroup_id"),
                            r.get("subgroup_size", 1),
                            r.get("statistics", {}).get("x_bar"),
                            r.get("statistics", {}).get("range_val"),
                            r.get("statistics", {}).get("std_dev"),
                            r.get("statistics", {}).get("cp"),
                            r.get("statistics", {}).get("cpk"),
                            r.get("status"),
                            r.get("rule_violations", []),
                        )
                        for r in spc_records
                    ],
                )

        logger.info(f"Sunk {len(records)} measurements to TimescaleDB")

    def _sink_alarms(self, records: list[dict]):
        """알람 데이터 적재"""
        if not records:
            return

        # TimescaleDB에 이력 저장
        with self.ts_conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO alarm_history (
                    time, alarm_id, alarm_code, alarm_name,
                    severity, category, source_system,
                    equipment_id, chamber_id, process_id, lot_id, wafer_id,
                    message, triggered_value, threshold_value, status
                ) VALUES %s
                ON CONFLICT (time, alarm_id) DO UPDATE SET
                    status = EXCLUDED.status
                """,
                [
                    (
                        r.get("occurred_at"),
                        r.get("alarm_id"),
                        r.get("alarm_code"),
                        r.get("alarm_name"),
                        r.get("severity"),
                        r.get("category"),
                        r.get("source_system"),
                        r.get("equipment_id"),
                        r.get("chamber_id"),
                        r.get("process_id"),
                        r.get("lot_id"),
                        r.get("wafer_id"),
                        r.get("message"),
                        r.get("triggered_value"),
                        r.get("threshold_value"),
                        r.get("status"),
                    )
                    for r in records
                ],
            )

        # AGE 그래프에도 저장
        with self.pg_conn.cursor() as cur:
            for r in records:
                props = json.dumps({
                    "alarm_id": r.get("alarm_id"),
                    "alarm_code": r.get("alarm_code"),
                    "alarm_name": r.get("alarm_name"),
                    "severity": r.get("severity"),
                    "category": r.get("category"),
                    "source_system": r.get("source_system"),
                    "equipment_id": r.get("equipment_id"),
                    "message": r.get("message"),
                    "status": r.get("status"),
                    "occurred_at": r.get("occurred_at"),
                })

                cur.execute(f"""
                    SELECT * FROM cypher('manufacturing', $$
                        MERGE (a:Alarm {{alarm_id: '{r.get("alarm_id")}'}})
                        SET a += {props}
                        RETURN a
                    $$) AS (a agtype)
                """)

        logger.info(f"Sunk {len(records)} alarms")

    def _sink_equipment(self, records: list[dict]):
        """설비 데이터 적재 (AGE)"""
        with self.pg_conn.cursor() as cur:
            for r in records:
                props = json.dumps({
                    "equipment_id": r.get("equipment_id"),
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "status": r.get("status"),
                    "location": r.get("location"),
                })

                cur.execute(f"""
                    SELECT * FROM cypher('manufacturing', $$
                        MERGE (e:Equipment {{equipment_id: '{r.get("equipment_id")}'}})
                        SET e += {props}
                        RETURN e
                    $$) AS (e agtype)
                """)

        logger.info(f"Sunk {len(records)} equipment records")

    def _sink_lot(self, records: list[dict]):
        """Lot 데이터 적재 (AGE)"""
        with self.pg_conn.cursor() as cur:
            for r in records:
                props = json.dumps({
                    "lot_id": r.get("lot_id"),
                    "product_code": r.get("product_code"),
                    "product_name": r.get("product_name"),
                    "quantity": r.get("quantity"),
                    "priority": r.get("priority"),
                    "status": r.get("status"),
                    "current_step": r.get("current_step"),
                    "fab_id": r.get("fab_id"),
                    "route_id": r.get("route_id"),
                })

                cur.execute(f"""
                    SELECT * FROM cypher('manufacturing', $$
                        MERGE (l:Lot {{lot_id: '{r.get("lot_id")}'}})
                        SET l += {props}
                        RETURN l
                    $$) AS (l agtype)
                """)

        logger.info(f"Sunk {len(records)} lot records")

    def _sink_wafer(self, records: list[dict]):
        """Wafer 데이터 적재 (AGE)"""
        with self.pg_conn.cursor() as cur:
            for r in records:
                props = json.dumps({
                    "wafer_id": r.get("wafer_id"),
                    "lot_id": r.get("lot_id"),
                    "slot_no": r.get("slot_no"),
                    "status": r.get("status"),
                })

                cur.execute(f"""
                    SELECT * FROM cypher('manufacturing', $$
                        MERGE (w:Wafer {{wafer_id: '{r.get("wafer_id")}'}})
                        SET w += {props}
                        RETURN w
                    $$) AS (w agtype)
                """)

                # Lot과의 관계 생성
                if r.get("lot_id"):
                    cur.execute(f"""
                        SELECT * FROM cypher('manufacturing', $$
                            MATCH (w:Wafer {{wafer_id: '{r.get("wafer_id")}'}}),
                                  (l:Lot {{lot_id: '{r.get("lot_id")}'}})
                            MERGE (w)-[r:BELONGS_TO]->(l)
                            SET r.slot_no = {r.get("slot_no", 0)}
                            RETURN r
                        $$) AS (r agtype)
                    """)

        logger.info(f"Sunk {len(records)} wafer records")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    sink = OntologySink(
        source_topics=[
            "fdc.measurements.processed",
            "spc.measurements.analyzed",
            "fdc.alarms.enriched",
            "ontology.objects.equipment",
            "ontology.objects.lot",
            "ontology.objects.wafer",
        ]
    )
    sink.start()
