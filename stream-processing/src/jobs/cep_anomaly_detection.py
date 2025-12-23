"""
CEP 이상 패턴 감지 잡

Complex Event Processing을 사용하여 복합 이상 패턴을 실시간 감지합니다.

패턴:
1. 연속 임계값 초과: 동일 설비에서 N회 연속 임계값 초과
2. 다중 알람: 동일 설비에서 짧은 시간 내 여러 종류의 알람
3. 드리프트 감지: 값이 점진적으로 한 방향으로 이동
4. 설비 간 상관: 연관 설비에서 동시에 이상 발생

입력: flink.measurements.enriched, fdc.FDC_ALARM_HISTORY
출력: flink.cep.alerts
"""
import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import defaultdict

from pyflink.datastream import StreamExecutionEnvironment, RuntimeContext
from pyflink.datastream.functions import RichMapFunction, RichFlatMapFunction
from pyflink.common import WatermarkStrategy

import sys
sys.path.append('/app/src')

from config import kafka_config, flink_config, cep_config
from utils.kafka_utils import (
    create_kafka_source,
    create_kafka_sink,
    parse_json_message,
    serialize_to_json,
    parse_timestamp,
)
from utils.state_utils import state_manager
from models import CepAlert, AlarmSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThresholdViolationPatternDetector(RichFlatMapFunction):
    """
    패턴 1: 연속 임계값 초과 감지

    동일 설비-파라미터에서 지정된 시간 윈도우 내에
    N회 이상 임계값 초과가 발생하면 알람 생성
    """

    def open(self, runtime_context: RuntimeContext):
        logger.info("Threshold Violation Pattern Detector initialized")

    def flat_map(self, value: str):
        """임계값 초과 패턴 감지"""
        try:
            data = parse_json_message(value)
            if not data:
                return

            # ALARM 상태인 경우만 처리
            status = data.get("status")
            if status not in ["ALARM", "WARNING"]:
                return

            equipment_id = data.get("equipment_id")
            param_id = data.get("param_id")
            measurement_id = data.get("measurement_id")

            if not equipment_id or not param_id:
                return

            # 패턴 키
            pattern_key = f"threshold:{equipment_id}:{param_id}"

            # 이벤트 추가
            event = {
                "measurement_id": measurement_id,
                "equipment_id": equipment_id,
                "param_id": param_id,
                "value": data.get("value"),
                "status": status,
                "timestamp": data.get("timestamp"),
            }

            state_manager.add_cep_event(
                pattern_key=pattern_key,
                event=event,
                window_ms=cep_config.threshold_violation_window,
            )

            # 윈도우 내 이벤트 수 확인
            event_count = state_manager.count_cep_events(
                pattern_key=pattern_key,
                window_ms=cep_config.threshold_violation_window,
            )

            # 임계값 초과 시 알람 생성
            if event_count >= cep_config.threshold_violation_count:
                alert = CepAlert(
                    alert_id=f"CEP-THR-{uuid.uuid4().hex[:8]}",
                    pattern_id="THRESHOLD_VIOLATION",
                    pattern_name="Consecutive Threshold Violations",
                    equipment_id=equipment_id,
                    severity=AlarmSeverity.MAJOR,
                    message=f"Parameter {param_id} exceeded threshold {event_count} times within {cep_config.threshold_violation_window // 1000}s",
                    detected_at=datetime.utcnow(),
                    lot_id=data.get("lot_id"),
                    process_id=data.get("process_id"),
                    affected_measurements=[e["measurement_id"] for e in
                                           state_manager.get_cep_events(pattern_key, cep_config.threshold_violation_window)],
                    context={
                        "param_id": param_id,
                        "violation_count": event_count,
                        "window_ms": cep_config.threshold_violation_window,
                        "last_value": data.get("value"),
                    },
                    confidence_score=min(1.0, event_count / cep_config.threshold_violation_count),
                )

                yield serialize_to_json(alert.to_dict())

        except Exception as e:
            logger.error(f"Error in threshold violation detection: {e}")


class ConsecutiveAlarmPatternDetector(RichFlatMapFunction):
    """
    패턴 2: 연속 알람 감지

    동일 설비에서 짧은 시간 내 여러 종류의 알람이 발생하면
    복합 이상으로 판단
    """

    def open(self, runtime_context: RuntimeContext):
        logger.info("Consecutive Alarm Pattern Detector initialized")

    def flat_map(self, value: str):
        """연속 알람 패턴 감지"""
        try:
            data = parse_json_message(value)
            if not data:
                return

            equipment_id = data.get("EQUIP_ID") or data.get("equipment_id")
            alarm_code = data.get("ALARM_CODE") or data.get("alarm_code")
            alarm_id = data.get("ALARM_ID") or data.get("alarm_id")

            if not equipment_id or not alarm_code:
                return

            # 패턴 키
            pattern_key = f"consecutive_alarm:{equipment_id}"

            # 이벤트 추가
            event = {
                "alarm_id": alarm_id,
                "alarm_code": alarm_code,
                "equipment_id": equipment_id,
                "timestamp": data.get("ALARM_TIME") or data.get("occurred_at"),
            }

            state_manager.add_cep_event(
                pattern_key=pattern_key,
                event=event,
                window_ms=cep_config.consecutive_alarm_window,
            )

            # 윈도우 내 이벤트 조회
            events = state_manager.get_cep_events(
                pattern_key=pattern_key,
                window_ms=cep_config.consecutive_alarm_window,
            )

            # 서로 다른 알람 코드 수 확인
            unique_alarm_codes = set(e["alarm_code"] for e in events)

            # 임계값 초과 시 알람 생성
            if len(events) >= cep_config.consecutive_alarm_count:
                alert = CepAlert(
                    alert_id=f"CEP-CONS-{uuid.uuid4().hex[:8]}",
                    pattern_id="CONSECUTIVE_ALARMS",
                    pattern_name="Multiple Alarms in Short Period",
                    equipment_id=equipment_id,
                    severity=AlarmSeverity.CRITICAL if len(unique_alarm_codes) >= 3 else AlarmSeverity.MAJOR,
                    message=f"Equipment {equipment_id} triggered {len(events)} alarms ({len(unique_alarm_codes)} different types) within {cep_config.consecutive_alarm_window // 1000}s",
                    detected_at=datetime.utcnow(),
                    lot_id=data.get("LOT_ID") or data.get("lot_id"),
                    affected_measurements=[e["alarm_id"] for e in events],
                    context={
                        "alarm_count": len(events),
                        "unique_alarm_types": len(unique_alarm_codes),
                        "alarm_codes": list(unique_alarm_codes),
                        "window_ms": cep_config.consecutive_alarm_window,
                    },
                    confidence_score=min(1.0, len(events) / cep_config.consecutive_alarm_count),
                )

                yield serialize_to_json(alert.to_dict())

        except Exception as e:
            logger.error(f"Error in consecutive alarm detection: {e}")


class DriftPatternDetector(RichFlatMapFunction):
    """
    패턴 3: 드리프트 감지

    값이 점진적으로 한 방향으로 이동하는 패턴 감지
    (이동 평균 기반)
    """

    def open(self, runtime_context: RuntimeContext):
        logger.info("Drift Pattern Detector initialized")

    def flat_map(self, value: str):
        """드리프트 패턴 감지"""
        try:
            data = parse_json_message(value)
            if not data:
                return

            equipment_id = data.get("equipment_id")
            param_id = data.get("param_id")
            current_value = data.get("value")
            target = data.get("target")

            if not equipment_id or not param_id or current_value is None:
                return

            # 패턴 키
            pattern_key = f"drift:{equipment_id}:{param_id}"

            # 히스토리 조회
            events = state_manager.get_cep_events(
                pattern_key=pattern_key,
                window_ms=cep_config.drift_detection_window,
            )

            # 현재 이벤트 추가
            event = {
                "value": current_value,
                "timestamp": data.get("timestamp"),
            }

            state_manager.add_cep_event(
                pattern_key=pattern_key,
                event=event,
                window_ms=cep_config.drift_detection_window,
            )

            # 드리프트 분석 (최소 10개 포인트 필요)
            if len(events) < 10:
                return

            values = [e["value"] for e in events] + [current_value]

            # 전반부와 후반부 평균 비교
            mid_point = len(values) // 2
            first_half_avg = sum(values[:mid_point]) / mid_point
            second_half_avg = sum(values[mid_point:]) / (len(values) - mid_point)

            # 표준편차 계산
            import statistics
            try:
                std_dev = statistics.stdev(values)
            except statistics.StatisticsError:
                return

            if std_dev == 0:
                return

            # 드리프트 정도 계산 (시그마 단위)
            drift_sigma = abs(second_half_avg - first_half_avg) / std_dev

            # 드리프트 방향
            drift_direction = "UP" if second_half_avg > first_half_avg else "DOWN"

            # 임계값 초과 시 알람 생성
            if drift_sigma >= cep_config.drift_threshold_sigma:
                alert = CepAlert(
                    alert_id=f"CEP-DFT-{uuid.uuid4().hex[:8]}",
                    pattern_id="DRIFT_DETECTED",
                    pattern_name=f"Value Drift {drift_direction}",
                    equipment_id=equipment_id,
                    severity=AlarmSeverity.WARNING if drift_sigma < 3.0 else AlarmSeverity.MAJOR,
                    message=f"Parameter {param_id} showing {drift_direction} drift of {drift_sigma:.2f} sigma over {cep_config.drift_detection_window // 1000}s",
                    detected_at=datetime.utcnow(),
                    lot_id=data.get("lot_id"),
                    process_id=data.get("process_id"),
                    context={
                        "param_id": param_id,
                        "drift_sigma": round(drift_sigma, 2),
                        "drift_direction": drift_direction,
                        "first_half_avg": round(first_half_avg, 4),
                        "second_half_avg": round(second_half_avg, 4),
                        "std_dev": round(std_dev, 4),
                        "target": target,
                        "sample_count": len(values),
                    },
                    confidence_score=min(1.0, drift_sigma / 4.0),
                )

                yield serialize_to_json(alert.to_dict())

        except Exception as e:
            logger.error(f"Error in drift detection: {e}")


def run_cep_anomaly_detection_job():
    """CEP 이상 패턴 감지 잡 실행"""
    logger.info("Starting CEP Anomaly Detection Job")

    # 실행 환경 설정
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(flink_config.parallelism)

    # 체크포인트 설정
    env.enable_checkpointing(flink_config.checkpoint_interval)

    # 출력 싱크
    alert_sink = create_kafka_sink(kafka_config.cep_alerts_topic)

    # =========================================================
    # 1. 측정 데이터 기반 패턴 (임계값 초과, 드리프트)
    # =========================================================
    measurement_source = create_kafka_source(
        topics=[kafka_config.enriched_measurement_topic],
        group_id=f"{kafka_config.group_id}-cep-measurement",
    )

    measurement_stream = env.from_source(
        measurement_source,
        WatermarkStrategy.no_watermarks(),
        "Enriched Measurement Source",
    )

    # 임계값 초과 패턴
    threshold_alerts = (
        measurement_stream
        .flat_map(ThresholdViolationPatternDetector())
        .name("Threshold Violation Detection")
    )

    # 드리프트 패턴
    drift_alerts = (
        measurement_stream
        .flat_map(DriftPatternDetector())
        .name("Drift Detection")
    )

    # =========================================================
    # 2. 알람 데이터 기반 패턴 (연속 알람)
    # =========================================================
    alarm_source = create_kafka_source(
        topics=[kafka_config.fdc_alarm_topic],
        group_id=f"{kafka_config.group_id}-cep-alarm",
    )

    consecutive_alerts = (
        env.from_source(
            alarm_source,
            WatermarkStrategy.no_watermarks(),
            "FDC Alarm Source",
        )
        .flat_map(ConsecutiveAlarmPatternDetector())
        .name("Consecutive Alarm Detection")
    )

    # =========================================================
    # 3. 모든 알람 통합 출력
    # =========================================================
    (
        threshold_alerts
        .union(drift_alerts, consecutive_alerts)
        .filter(lambda x: x is not None)
        .name("Filter Nulls")
        .sink_to(alert_sink)
        .name("CEP Alerts Sink")
    )

    # 실행
    env.execute("CEP Anomaly Detection Job")


if __name__ == "__main__":
    run_cep_anomaly_detection_job()
