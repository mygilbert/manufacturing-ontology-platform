"""
SPC 실시간 관리도 잡

SPC 측정 데이터에 Western Electric Rules를 적용하여 OOC/OOS를 실시간 감지합니다.

입력: spc.SPC_MEASUREMENT
출력: flink.spc.analyzed
"""
import json
import logging
import statistics
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from pyflink.datastream import StreamExecutionEnvironment, RuntimeContext
from pyflink.datastream.functions import RichMapFunction
from pyflink.common import WatermarkStrategy

import sys
sys.path.append('/app/src')

from config import kafka_config, flink_config, spc_config
from utils.kafka_utils import (
    create_kafka_source,
    create_kafka_sink,
    parse_json_message,
    serialize_to_json,
    parse_timestamp,
)
from utils.state_utils import state_manager
from models import SpcMeasurement, SpcAnalyzedMeasurement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpcAnalysisFunction(RichMapFunction):
    """SPC 분석 함수 (Western Electric Rules)"""

    def open(self, runtime_context: RuntimeContext):
        """초기화"""
        logger.info("SPC Analysis Function initialized")

    def map(self, value: str) -> Optional[str]:
        """SPC 측정 데이터 분석"""
        try:
            # JSON 파싱
            data = parse_json_message(value)
            if not data:
                return None

            # 측정값 추출
            measurement = self._parse_measurement(data)
            if not measurement:
                return None

            # 히스토리에 추가
            state_manager.add_spc_value(
                equipment_id=measurement.equipment_id,
                item_id=measurement.item_id,
                value=measurement.value,
                timestamp=measurement.timestamp,
                max_size=spc_config.history_window_size,
            )

            # 히스토리 조회
            history = state_manager.get_spc_values(
                equipment_id=measurement.equipment_id,
                item_id=measurement.item_id,
                count=spc_config.history_window_size,
            )

            # SPC 규칙 체크
            violations = self._check_western_electric_rules(
                values=history,
                current_value=measurement.value,
                ucl=measurement.ucl,
                lcl=measurement.lcl,
                target=measurement.target,
            )

            # 상태 결정
            status = self._determine_status(
                value=measurement.value,
                usl=measurement.usl,
                lsl=measurement.lsl,
                ucl=measurement.ucl,
                lcl=measurement.lcl,
                violations=violations,
            )

            # 통계 계산
            stats = self._calculate_statistics(
                values=history,
                usl=measurement.usl,
                lsl=measurement.lsl,
            )

            # 분석 결과 생성
            analyzed = SpcAnalyzedMeasurement(
                # 기본 필드
                measurement_id=measurement.measurement_id,
                equipment_id=measurement.equipment_id,
                process_id=measurement.process_id,
                item_id=measurement.item_id,
                value=measurement.value,
                timestamp=measurement.timestamp,
                lot_id=measurement.lot_id,
                wafer_id=measurement.wafer_id,
                item_name=measurement.item_name,
                unit=measurement.unit,
                usl=measurement.usl,
                lsl=measurement.lsl,
                ucl=measurement.ucl,
                lcl=measurement.lcl,
                target=measurement.target,
                subgroup_id=measurement.subgroup_id,
                subgroup_size=measurement.subgroup_size,
                # 분석 결과
                status=status,
                rule_violations=violations,
                x_bar=stats.get("x_bar"),
                range_val=stats.get("range_val"),
                std_dev=stats.get("std_dev"),
                cp=stats.get("cp"),
                cpk=stats.get("cpk"),
            )

            return serialize_to_json(analyzed.to_dict())

        except Exception as e:
            logger.error(f"Error analyzing SPC measurement: {e}")
            return None

    def _parse_measurement(self, data: Dict[str, Any]) -> Optional[SpcMeasurement]:
        """SPC 측정 데이터 파싱"""
        try:
            measurement_id = data.get("MEAS_ID") or data.get("measurement_id")
            equipment_id = data.get("EQUIP_ID") or data.get("equipment_id")
            process_id = data.get("PROCESS_ID") or data.get("process_id")
            item_id = data.get("ITEM_ID") or data.get("item_id")
            value = data.get("MEAS_VALUE") or data.get("value")
            timestamp_raw = data.get("MEAS_TIME") or data.get("timestamp")

            if not all([measurement_id, equipment_id, process_id, item_id, value is not None]):
                return None

            timestamp = parse_timestamp(timestamp_raw) or datetime.utcnow()

            def safe_float(v):
                return float(v) if v is not None else None

            return SpcMeasurement(
                measurement_id=str(measurement_id),
                equipment_id=str(equipment_id),
                process_id=str(process_id),
                item_id=str(item_id),
                value=float(value),
                timestamp=timestamp,
                lot_id=data.get("LOT_ID") or data.get("lot_id"),
                wafer_id=data.get("WAFER_ID") or data.get("wafer_id"),
                item_name=data.get("ITEM_NAME") or data.get("item_name"),
                unit=data.get("UNIT") or data.get("unit"),
                usl=safe_float(data.get("USL") or data.get("usl")),
                lsl=safe_float(data.get("LSL") or data.get("lsl")),
                ucl=safe_float(data.get("UCL") or data.get("ucl")),
                lcl=safe_float(data.get("LCL") or data.get("lcl")),
                target=safe_float(data.get("TARGET") or data.get("target")),
                subgroup_id=data.get("SUBGROUP_ID") or data.get("subgroup_id"),
                subgroup_size=data.get("SUBGROUP_SIZE") or data.get("subgroup_size") or 1,
            )

        except Exception as e:
            logger.error(f"Error parsing SPC measurement: {e}")
            return None

    def _check_western_electric_rules(
        self,
        values: List[float],
        current_value: float,
        ucl: Optional[float],
        lcl: Optional[float],
        target: Optional[float],
    ) -> List[str]:
        """Western Electric SPC 규칙 체크"""
        violations = []

        if not values or ucl is None or lcl is None:
            return violations

        # 센터라인 계산
        if target is not None:
            center = target
        else:
            center = (ucl + lcl) / 2

        # 시그마 계산
        sigma = (ucl - center) / 3
        if sigma <= 0:
            return violations

        one_sigma_upper = center + sigma
        one_sigma_lower = center - sigma
        two_sigma_upper = center + 2 * sigma
        two_sigma_lower = center - 2 * sigma

        # Rule 1: 1점이 관리한계 초과
        if spc_config.rule1_enabled:
            if current_value > ucl or current_value < lcl:
                violations.append("RULE1_OOC")

        # Rule 2: 연속 9점이 중심선 한쪽
        if spc_config.rule2_enabled and len(values) >= 9:
            last_9 = values[-9:]
            if all(v > center for v in last_9):
                violations.append("RULE2_RUN_ABOVE")
            elif all(v < center for v in last_9):
                violations.append("RULE2_RUN_BELOW")

        # Rule 3: 연속 6점 증가 또는 감소
        if spc_config.rule3_enabled and len(values) >= 6:
            last_6 = values[-6:]
            increasing = all(last_6[i] < last_6[i + 1] for i in range(5))
            decreasing = all(last_6[i] > last_6[i + 1] for i in range(5))
            if increasing:
                violations.append("RULE3_TREND_UP")
            elif decreasing:
                violations.append("RULE3_TREND_DOWN")

        # Rule 4: 연속 14점 교대로 상승/하락
        if spc_config.rule4_enabled and len(values) >= 14:
            last_14 = values[-14:]
            try:
                alternating = all(
                    (last_14[i] < last_14[i + 1]) != (last_14[i + 1] < last_14[i + 2])
                    for i in range(12)
                )
                if alternating:
                    violations.append("RULE4_ALTERNATING")
            except IndexError:
                pass

        # Rule 5: 3점 중 2점이 2시그마 초과 (같은 방향)
        if spc_config.rule5_enabled and len(values) >= 3:
            last_3 = values[-3:]
            above_2sigma = sum(1 for v in last_3 if v > two_sigma_upper)
            below_2sigma = sum(1 for v in last_3 if v < two_sigma_lower)
            if above_2sigma >= 2:
                violations.append("RULE5_2OF3_ABOVE_2SIGMA")
            elif below_2sigma >= 2:
                violations.append("RULE5_2OF3_BELOW_2SIGMA")

        # Rule 6: 5점 중 4점이 1시그마 초과 (같은 방향)
        if spc_config.rule6_enabled and len(values) >= 5:
            last_5 = values[-5:]
            above_1sigma = sum(1 for v in last_5 if v > one_sigma_upper)
            below_1sigma = sum(1 for v in last_5 if v < one_sigma_lower)
            if above_1sigma >= 4:
                violations.append("RULE6_4OF5_ABOVE_1SIGMA")
            elif below_1sigma >= 4:
                violations.append("RULE6_4OF5_BELOW_1SIGMA")

        # Rule 7: 15점이 1시그마 이내 (층화)
        if spc_config.rule7_enabled and len(values) >= 15:
            last_15 = values[-15:]
            within_1sigma = all(
                one_sigma_lower <= v <= one_sigma_upper
                for v in last_15
            )
            if within_1sigma:
                violations.append("RULE7_STRATIFICATION")

        # Rule 8: 8점 연속으로 1시그마 밖 (양쪽 교대, 혼합)
        if spc_config.rule8_enabled and len(values) >= 8:
            last_8 = values[-8:]
            outside_1sigma = all(
                v > one_sigma_upper or v < one_sigma_lower
                for v in last_8
            )
            if outside_1sigma:
                violations.append("RULE8_MIXTURE")

        return violations

    def _determine_status(
        self,
        value: float,
        usl: Optional[float],
        lsl: Optional[float],
        ucl: Optional[float],
        lcl: Optional[float],
        violations: List[str],
    ) -> str:
        """상태 결정"""
        # 규격 한계 초과 (OOS)
        if usl is not None and value > usl:
            return "OOS"
        if lsl is not None and value < lsl:
            return "OOS"

        # 관리 한계 초과 (OOC)
        if "RULE1_OOC" in violations:
            return "OOC"

        # 트렌드
        if any(v.startswith("RULE3_TREND") for v in violations):
            return "TREND"

        # 시프트
        if any(v.startswith("RULE2_RUN") for v in violations):
            return "SHIFT"

        # 기타 규칙 위반
        if violations:
            return "OOC"

        return "NORMAL"

    def _calculate_statistics(
        self,
        values: List[float],
        usl: Optional[float],
        lsl: Optional[float],
    ) -> Dict[str, Optional[float]]:
        """통계 계산"""
        result: Dict[str, Optional[float]] = {
            "x_bar": None,
            "range_val": None,
            "std_dev": None,
            "cp": None,
            "cpk": None,
        }

        if not values or len(values) < 2:
            return result

        try:
            n = len(values)
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values)

            result["x_bar"] = mean
            result["std_dev"] = std_dev

            # Range (최근 2개 값의 차이)
            if n >= 2:
                result["range_val"] = abs(values[-1] - values[-2])

            # Cp, Cpk 계산
            if usl is not None and lsl is not None and std_dev > 0:
                result["cp"] = (usl - lsl) / (6 * std_dev)

                cpu = (usl - mean) / (3 * std_dev)
                cpl = (mean - lsl) / (3 * std_dev)
                result["cpk"] = min(cpu, cpl)

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")

        return result


def run_spc_control_chart_job():
    """SPC 관리도 잡 실행"""
    logger.info("Starting SPC Control Chart Job")

    # 실행 환경 설정
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(flink_config.parallelism)

    # 체크포인트 설정
    env.enable_checkpointing(flink_config.checkpoint_interval)

    # =========================================================
    # SPC 측정 스트림
    # =========================================================
    spc_source = create_kafka_source(
        topics=[kafka_config.spc_measurement_topic],
        group_id=f"{kafka_config.group_id}-spc",
    )

    spc_sink = create_kafka_sink(kafka_config.spc_analyzed_topic)

    (
        env.from_source(
            spc_source,
            WatermarkStrategy.no_watermarks(),
            "SPC Measurement Source",
        )
        .map(SpcAnalysisFunction())
        .name("Analyze SPC Measurements")
        .filter(lambda x: x is not None)
        .name("Filter Nulls")
        .sink_to(spc_sink)
        .name("SPC Analyzed Sink")
    )

    # 실행
    env.execute("SPC Control Chart Job")


if __name__ == "__main__":
    run_spc_control_chart_job()
