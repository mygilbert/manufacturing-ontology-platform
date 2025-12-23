"""
FDC 데이터 보강 잡

FDC 측정 데이터에 설비/레시피/Lot 정보를 조인하여 보강합니다.

입력: fdc.FDC_PARAM_VALUE
출력: flink.measurements.enriched
"""
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

from pyflink.datastream import StreamExecutionEnvironment, RuntimeContext
from pyflink.datastream.functions import MapFunction, RichMapFunction
from pyflink.common import WatermarkStrategy

import sys
sys.path.append('/app/src')

from config import kafka_config, flink_config
from utils.kafka_utils import (
    create_kafka_source,
    create_kafka_sink,
    parse_json_message,
    serialize_to_json,
    parse_timestamp,
)
from utils.state_utils import state_manager
from models import FdcMeasurement, EnrichedMeasurement, MeasurementStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FdcEnrichmentFunction(RichMapFunction):
    """FDC 데이터 보강 함수"""

    def open(self, runtime_context: RuntimeContext):
        """초기화"""
        logger.info("FDC Enrichment Function initialized")

    def map(self, value: str) -> Optional[str]:
        """측정 데이터 보강"""
        try:
            # JSON 파싱
            data = parse_json_message(value)
            if not data:
                return None

            # 기본 측정값 추출
            measurement = self._parse_measurement(data)
            if not measurement:
                return None

            # 상태 판정
            measurement.status = self._determine_status(
                value=measurement.value,
                usl=measurement.usl,
                lsl=measurement.lsl,
            )

            # 보강 정보 조회 및 적용
            enriched = self._enrich_measurement(measurement)

            # JSON 출력
            return serialize_to_json(enriched.to_dict())

        except Exception as e:
            logger.error(f"Error enriching measurement: {e}")
            return None

    def _parse_measurement(self, data: Dict[str, Any]) -> Optional[FdcMeasurement]:
        """측정 데이터 파싱"""
        try:
            # 필수 필드 확인
            measurement_id = data.get("VALUE_ID") or data.get("measurement_id")
            equipment_id = data.get("EQUIP_ID") or data.get("equipment_id")
            param_id = data.get("PARAM_ID") or data.get("param_id")
            value = data.get("PARAM_VALUE") or data.get("value")
            timestamp_raw = data.get("COLLECT_TIME") or data.get("timestamp")

            if not all([measurement_id, equipment_id, param_id, value is not None]):
                return None

            timestamp = parse_timestamp(timestamp_raw) or datetime.utcnow()

            return FdcMeasurement(
                measurement_id=str(measurement_id),
                equipment_id=str(equipment_id),
                timestamp=timestamp,
                param_id=str(param_id),
                value=float(value),
                chamber_id=data.get("CHAMBER_ID") or data.get("chamber_id"),
                recipe_id=data.get("RECIPE_ID") or data.get("recipe_id"),
                lot_id=data.get("LOT_ID") or data.get("lot_id"),
                wafer_id=data.get("WAFER_ID") or data.get("wafer_id"),
                slot_no=data.get("SLOT_NO") or data.get("slot_no"),
                param_name=data.get("PARAM_NAME") or data.get("param_name"),
                unit=data.get("PARAM_UNIT") or data.get("unit"),
                usl=float(data["USL"]) if data.get("USL") or data.get("usl") else None,
                lsl=float(data["LSL"]) if data.get("LSL") or data.get("lsl") else None,
                target=float(data["TARGET"]) if data.get("TARGET") or data.get("target") else None,
            )

        except Exception as e:
            logger.error(f"Error parsing measurement: {e}")
            return None

    def _determine_status(
        self,
        value: float,
        usl: Optional[float],
        lsl: Optional[float],
    ) -> MeasurementStatus:
        """측정값 상태 판정"""
        # 규격 한계 초과
        if usl is not None and value > usl:
            return MeasurementStatus.ALARM
        if lsl is not None and value < lsl:
            return MeasurementStatus.ALARM

        # 경고 범위 (규격의 90-100%)
        if usl is not None and lsl is not None:
            range_val = usl - lsl
            warning_upper = usl - (range_val * 0.1)
            warning_lower = lsl + (range_val * 0.1)

            if value > warning_upper or value < warning_lower:
                return MeasurementStatus.WARNING

        return MeasurementStatus.NORMAL

    def _enrich_measurement(self, measurement: FdcMeasurement) -> EnrichedMeasurement:
        """측정 데이터에 추가 정보 보강"""
        # 설비 정보 조회
        equipment_info = state_manager.get_equipment(measurement.equipment_id)

        # 레시피 정보 조회
        recipe_info = None
        if measurement.recipe_id:
            recipe_info = state_manager.get_recipe(measurement.recipe_id)

        # Lot 정보 조회
        lot_info = None
        if measurement.lot_id:
            lot_info = state_manager.get_lot(measurement.lot_id)

        # EnrichedMeasurement 생성
        return EnrichedMeasurement(
            # 기본 필드
            measurement_id=measurement.measurement_id,
            equipment_id=measurement.equipment_id,
            timestamp=measurement.timestamp,
            param_id=measurement.param_id,
            value=measurement.value,
            chamber_id=measurement.chamber_id,
            recipe_id=measurement.recipe_id,
            lot_id=measurement.lot_id,
            wafer_id=measurement.wafer_id,
            slot_no=measurement.slot_no,
            param_name=measurement.param_name,
            unit=measurement.unit,
            usl=measurement.usl,
            lsl=measurement.lsl,
            target=measurement.target,
            status=measurement.status,
            # 보강 필드
            equipment_name=equipment_info.get("name") if equipment_info else None,
            equipment_type=equipment_info.get("type") if equipment_info else None,
            equipment_location=equipment_info.get("location") if equipment_info else None,
            recipe_name=recipe_info.get("recipe_name") if recipe_info else None,
            recipe_version=recipe_info.get("version") if recipe_info else None,
            process_id=lot_info.get("current_step") if lot_info else None,
            product_code=lot_info.get("product_code") if lot_info else None,
        )


class EquipmentUpdateFunction(RichMapFunction):
    """설비 마스터 업데이트 함수"""

    def map(self, value: str) -> Optional[str]:
        """설비 정보 캐시 업데이트"""
        try:
            data = parse_json_message(value)
            if not data:
                return None

            equipment_id = data.get("EQUIP_ID") or data.get("equipment_id")
            if not equipment_id:
                return None

            # 캐시 업데이트
            equipment_data = {
                "equipment_id": equipment_id,
                "name": data.get("EQUIP_NAME") or data.get("name"),
                "type": data.get("EQUIP_TYPE") or data.get("type"),
                "status": data.get("EQUIP_STATUS") or data.get("status"),
                "location": data.get("EQUIP_LOCATION") or data.get("location"),
            }

            state_manager.set_equipment(equipment_id, equipment_data)
            logger.debug(f"Updated equipment cache: {equipment_id}")

            return value  # 패스스루

        except Exception as e:
            logger.error(f"Error updating equipment: {e}")
            return None


def run_fdc_enrichment_job():
    """FDC 보강 잡 실행"""
    logger.info("Starting FDC Enrichment Job")

    # 실행 환경 설정
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(flink_config.parallelism)

    # 체크포인트 설정
    env.enable_checkpointing(flink_config.checkpoint_interval)

    # =========================================================
    # 1. 설비 마스터 스트림 (캐시 업데이트용)
    # =========================================================
    equipment_source = create_kafka_source(
        topics=[kafka_config.fdc_equipment_topic],
        group_id=f"{kafka_config.group_id}-equipment",
    )

    env.from_source(
        equipment_source,
        WatermarkStrategy.no_watermarks(),
        "Equipment Source",
    ).map(EquipmentUpdateFunction()).name("Update Equipment Cache")

    # =========================================================
    # 2. FDC 측정 스트림 (메인 처리)
    # =========================================================
    measurement_source = create_kafka_source(
        topics=[kafka_config.fdc_measurement_topic],
        group_id=f"{kafka_config.group_id}-fdc",
    )

    measurement_sink = create_kafka_sink(kafka_config.enriched_measurement_topic)

    (
        env.from_source(
            measurement_source,
            WatermarkStrategy.no_watermarks(),
            "FDC Measurement Source",
        )
        .map(FdcEnrichmentFunction())
        .name("Enrich FDC Measurements")
        .filter(lambda x: x is not None)
        .name("Filter Nulls")
        .sink_to(measurement_sink)
        .name("Enriched Measurement Sink")
    )

    # 실행
    env.execute("FDC Enrichment Job")


if __name__ == "__main__":
    run_fdc_enrichment_job()
