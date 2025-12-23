"""
FDC 데이터 변환기

FDC 시스템에서 수집된 공정 파라미터 데이터를 온톨로지 형식으로 변환
"""
import logging
from datetime import datetime
from typing import Optional

from base_transformer import BaseTransformer
from config import KafkaConfig

logger = logging.getLogger(__name__)


class FdcMeasurementTransformer(BaseTransformer):
    """FDC 측정 데이터 변환기"""

    def __init__(self, kafka_config: Optional[KafkaConfig] = None):
        super().__init__(
            source_topics=["fdc.FDC_PARAM_VALUE"],
            target_topic="fdc.measurements.processed",
            kafka_config=kafka_config,
        )
        # 상태 판정을 위한 캐시 (실제로는 Redis 사용)
        self._spec_cache = {}

    def transform(self, record: dict, table_name: str) -> Optional[dict]:
        """FDC 측정값 변환"""
        try:
            # 필수 필드 검증
            if not record.get("measurement_id") or not record.get("equipment_id"):
                logger.warning(f"Missing required fields: {record}")
                return None

            # 타임스탬프 정규화
            timestamp = record.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000)

            # 상태 판정
            status = self._determine_status(
                value=record.get("value"),
                usl=record.get("usl"),
                lsl=record.get("lsl"),
            )

            # 변환된 레코드
            transformed = {
                "measurement_id": record["measurement_id"],
                "equipment_id": record["equipment_id"],
                "chamber_id": record.get("chamber_id"),
                "recipe_id": record.get("recipe_id"),
                "lot_id": record.get("lot_id"),
                "wafer_id": record.get("wafer_id"),
                "slot_no": record.get("slot_no"),
                "param_id": record.get("param_id"),
                "param_name": record.get("param_name"),
                "value": float(record["value"]) if record.get("value") is not None else None,
                "unit": record.get("unit"),
                "timestamp": timestamp.isoformat() if timestamp else None,
                "usl": float(record["usl"]) if record.get("usl") is not None else None,
                "lsl": float(record["lsl"]) if record.get("lsl") is not None else None,
                "target": float(record["target"]) if record.get("target") is not None else None,
                "status": status,
                "source_system": "FDC",
                "event_time": datetime.utcnow().isoformat(),
                "processing_time": datetime.utcnow().isoformat(),
            }

            return transformed

        except Exception as e:
            logger.error(f"Error transforming FDC measurement: {e}", exc_info=True)
            return None

    def _determine_status(
        self,
        value: Optional[float],
        usl: Optional[float],
        lsl: Optional[float],
    ) -> str:
        """측정값 상태 판정"""
        if value is None:
            return "NORMAL"

        # 규격 한계 초과 (ALARM)
        if usl is not None and value > usl:
            return "ALARM"
        if lsl is not None and value < lsl:
            return "ALARM"

        # 경고 범위 (UCL/LCL 또는 규격의 80%)
        if usl is not None and lsl is not None:
            range_val = usl - lsl
            warning_upper = usl - (range_val * 0.1)
            warning_lower = lsl + (range_val * 0.1)

            if value > warning_upper or value < warning_lower:
                return "WARNING"

        return "NORMAL"


class FdcAlarmTransformer(BaseTransformer):
    """FDC 알람 데이터 변환기"""

    def __init__(self, kafka_config: Optional[KafkaConfig] = None):
        super().__init__(
            source_topics=["fdc.FDC_ALARM_HISTORY"],
            target_topic="fdc.alarms.enriched",
            kafka_config=kafka_config,
        )

    def transform(self, record: dict, table_name: str) -> Optional[dict]:
        """FDC 알람 변환"""
        try:
            # 심각도 매핑
            severity_map = {
                "1": "CRITICAL",
                "2": "MAJOR",
                "3": "MINOR",
                "4": "WARNING",
                "5": "INFO",
                "CRITICAL": "CRITICAL",
                "MAJOR": "MAJOR",
                "MINOR": "MINOR",
                "WARNING": "WARNING",
                "INFO": "INFO",
            }

            severity = severity_map.get(
                str(record.get("severity", "")).upper(),
                "WARNING"
            )

            # 타임스탬프 정규화
            occurred_at = record.get("occurred_at")
            if isinstance(occurred_at, str):
                occurred_at = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            elif isinstance(occurred_at, (int, float)):
                occurred_at = datetime.fromtimestamp(occurred_at / 1000)

            # 변환된 레코드
            transformed = {
                "alarm_id": record.get("alarm_id"),
                "alarm_code": record.get("alarm_code"),
                "alarm_name": record.get("alarm_name"),
                "source_system": "FDC",
                "severity": severity,
                "category": "FAULT",  # FDC 알람은 기본적으로 FAULT
                "equipment_id": record.get("equipment_id"),
                "chamber_id": record.get("chamber_id"),
                "lot_id": record.get("lot_id"),
                "wafer_id": record.get("wafer_id"),
                "occurred_at": occurred_at.isoformat() if occurred_at else None,
                "message": record.get("message"),
                "triggered_value": float(record["triggered_value"]) if record.get("triggered_value") is not None else None,
                "threshold_value": float(record["threshold_value"]) if record.get("threshold_value") is not None else None,
                "status": "ACTIVE",
                "affected_lots": [record["lot_id"]] if record.get("lot_id") else [],
                "context": {
                    "source_table": table_name,
                    "original_severity": str(record.get("severity")),
                },
            }

            return transformed

        except Exception as e:
            logger.error(f"Error transforming FDC alarm: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    transformer_type = sys.argv[1] if len(sys.argv) > 1 else "measurement"

    if transformer_type == "measurement":
        transformer = FdcMeasurementTransformer()
    elif transformer_type == "alarm":
        transformer = FdcAlarmTransformer()
    else:
        print(f"Unknown transformer type: {transformer_type}")
        sys.exit(1)

    transformer.start()
