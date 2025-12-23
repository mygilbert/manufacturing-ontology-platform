"""
SPC 데이터 변환기

SPC 시스템에서 수집된 품질 측정 데이터를 온톨로지 형식으로 변환하고
SPC 규칙 위반을 감지
"""
import logging
from collections import deque
from datetime import datetime
from typing import Optional

from base_transformer import BaseTransformer
from config import KafkaConfig

logger = logging.getLogger(__name__)


class SpcMeasurementTransformer(BaseTransformer):
    """SPC 측정 데이터 변환기"""

    # Western Electric Rules 체크를 위한 윈도우 크기
    WINDOW_SIZE = 9

    def __init__(self, kafka_config: Optional[KafkaConfig] = None):
        super().__init__(
            source_topics=["spc.SPC_MEASUREMENT"],
            target_topic="spc.measurements.analyzed",
            kafka_config=kafka_config,
        )
        # 항목별 이력 저장 (실제로는 Redis 사용)
        self._history: dict[str, deque] = {}

    def transform(self, record: dict, table_name: str) -> Optional[dict]:
        """SPC 측정값 변환 및 규칙 체크"""
        try:
            # 필수 필드 검증
            if not record.get("measurement_id") or not record.get("item_id"):
                logger.warning(f"Missing required fields: {record}")
                return None

            value = float(record["value"]) if record.get("value") is not None else None

            # 관리 한계 추출
            ucl = float(record["ucl"]) if record.get("ucl") is not None else None
            lcl = float(record["lcl"]) if record.get("lcl") is not None else None
            target = float(record["target"]) if record.get("target") is not None else None

            # 센터라인 계산 (target 또는 ucl/lcl 중간값)
            if target is not None:
                center = target
            elif ucl is not None and lcl is not None:
                center = (ucl + lcl) / 2
            else:
                center = None

            # 이력에 값 추가
            history_key = f"{record.get('equipment_id')}:{record.get('item_id')}"
            if history_key not in self._history:
                self._history[history_key] = deque(maxlen=self.WINDOW_SIZE)
            self._history[history_key].append(value)

            # SPC 규칙 위반 체크
            violations = []
            if value is not None and ucl is not None and lcl is not None and center is not None:
                violations = self._check_western_electric_rules(
                    values=list(self._history[history_key]),
                    ucl=ucl,
                    lcl=lcl,
                    center=center,
                )

            # 상태 결정
            status = self._determine_status(value, ucl, lcl, violations)

            # 통계 계산
            statistics = self._calculate_statistics(
                values=list(self._history[history_key]),
                usl=float(record["usl"]) if record.get("usl") else None,
                lsl=float(record["lsl"]) if record.get("lsl") else None,
            )

            # 타임스탬프 정규화
            timestamp = record.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            elif isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp / 1000)

            # 변환된 레코드
            transformed = {
                "measurement_id": record["measurement_id"],
                "equipment_id": record.get("equipment_id"),
                "process_id": record.get("process_id"),
                "lot_id": record.get("lot_id"),
                "wafer_id": record.get("wafer_id"),
                "item_id": record.get("item_id"),
                "item_name": record.get("item_name"),
                "value": value,
                "unit": record.get("unit"),
                "timestamp": timestamp.isoformat() if timestamp else None,
                "usl": float(record["usl"]) if record.get("usl") is not None else None,
                "lsl": float(record["lsl"]) if record.get("lsl") is not None else None,
                "ucl": ucl,
                "lcl": lcl,
                "target": target,
                "subgroup_id": record.get("subgroup_id"),
                "subgroup_size": record.get("subgroup_size", 1),
                "statistics": statistics,
                "status": status,
                "rule_violations": violations,
                "source_system": "SPC",
            }

            return transformed

        except Exception as e:
            logger.error(f"Error transforming SPC measurement: {e}", exc_info=True)
            return None

    def _check_western_electric_rules(
        self,
        values: list[float],
        ucl: float,
        lcl: float,
        center: float,
    ) -> list[str]:
        """Western Electric SPC 규칙 체크"""
        violations = []

        if not values:
            return violations

        current_value = values[-1]
        sigma = (ucl - center) / 3
        one_sigma_upper = center + sigma
        one_sigma_lower = center - sigma
        two_sigma_upper = center + 2 * sigma
        two_sigma_lower = center - 2 * sigma

        # Rule 1: 1점이 관리한계 초과
        if current_value > ucl or current_value < lcl:
            violations.append("RULE1_OOC")

        # Rule 2: 연속 9점이 중심선 한쪽
        if len(values) >= 9:
            last_9 = values[-9:]
            if all(v > center for v in last_9):
                violations.append("RULE2_RUN_ABOVE")
            elif all(v < center for v in last_9):
                violations.append("RULE2_RUN_BELOW")

        # Rule 3: 연속 6점 증가 또는 감소
        if len(values) >= 6:
            last_6 = values[-6:]
            increasing = all(last_6[i] < last_6[i+1] for i in range(5))
            decreasing = all(last_6[i] > last_6[i+1] for i in range(5))
            if increasing:
                violations.append("RULE3_TREND_UP")
            elif decreasing:
                violations.append("RULE3_TREND_DOWN")

        # Rule 4: 연속 14점 교대로 상승/하락
        if len(values) >= 14:
            last_14 = values[-14:]
            alternating = all(
                (last_14[i] < last_14[i+1]) != (last_14[i+1] < last_14[i+2])
                for i in range(12)
            )
            if alternating:
                violations.append("RULE4_ALTERNATING")

        # Rule 5: 3점 중 2점이 2시그마 초과 (같은 방향)
        if len(values) >= 3:
            last_3 = values[-3:]
            above_2sigma = sum(1 for v in last_3 if v > two_sigma_upper)
            below_2sigma = sum(1 for v in last_3 if v < two_sigma_lower)
            if above_2sigma >= 2:
                violations.append("RULE5_2OF3_ABOVE")
            elif below_2sigma >= 2:
                violations.append("RULE5_2OF3_BELOW")

        # Rule 6: 5점 중 4점이 1시그마 초과 (같은 방향)
        if len(values) >= 5:
            last_5 = values[-5:]
            above_1sigma = sum(1 for v in last_5 if v > one_sigma_upper)
            below_1sigma = sum(1 for v in last_5 if v < one_sigma_lower)
            if above_1sigma >= 4:
                violations.append("RULE6_4OF5_ABOVE")
            elif below_1sigma >= 4:
                violations.append("RULE6_4OF5_BELOW")

        # Rule 7: 15점이 1시그마 이내
        if len(values) >= 15:
            last_15 = values[-15:]
            within_1sigma = all(
                one_sigma_lower <= v <= one_sigma_upper
                for v in last_15
            )
            if within_1sigma:
                violations.append("RULE7_STRATIFICATION")

        # Rule 8: 8점 연속으로 1시그마 밖 (양쪽 교대)
        if len(values) >= 8:
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
        value: Optional[float],
        ucl: Optional[float],
        lcl: Optional[float],
        violations: list[str],
    ) -> str:
        """상태 결정"""
        if "RULE1_OOC" in violations:
            return "OOC"  # Out of Control
        elif any(v.startswith("RULE3_TREND") for v in violations):
            return "TREND"
        elif any(v.startswith("RULE2_RUN") for v in violations):
            return "SHIFT"
        elif violations:
            return "OOC"
        return "NORMAL"

    def _calculate_statistics(
        self,
        values: list[float],
        usl: Optional[float],
        lsl: Optional[float],
    ) -> dict:
        """통계 계산"""
        if not values or len(values) < 2:
            return {}

        import statistics

        n = len(values)
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if n > 1 else 0

        result = {
            "x_bar": mean,
            "std_dev": std_dev,
        }

        # Range (마지막 2개 값의 차이)
        if n >= 2:
            result["range_val"] = abs(values[-1] - values[-2])

        # Cp, Cpk 계산
        if usl is not None and lsl is not None and std_dev > 0:
            result["cp"] = (usl - lsl) / (6 * std_dev)

            cpu = (usl - mean) / (3 * std_dev)
            cpl = (mean - lsl) / (3 * std_dev)
            result["cpk"] = min(cpu, cpl)

        return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    transformer = SpcMeasurementTransformer()
    transformer.start()
