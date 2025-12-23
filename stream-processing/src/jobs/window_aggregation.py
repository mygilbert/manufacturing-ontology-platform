"""
윈도우 집계 잡

측정 데이터를 시간 윈도우 단위로 집계하여 통계를 생성합니다.

윈도우:
- 5분 텀블링 윈도우: 설비-파라미터별 통계
- 1시간 텀블링 윈도우: 공정별 SPC 통계

입력: flink.measurements.enriched
출력: flink.stats.aggregated
"""
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Iterable

from pyflink.datastream import StreamExecutionEnvironment, RuntimeContext
from pyflink.datastream.functions import (
    RichMapFunction,
    AggregateFunction,
    ProcessWindowFunction,
)
from pyflink.datastream.window import TumblingProcessingTimeWindows, Time
from pyflink.common import WatermarkStrategy, Types
from pyflink.datastream.state import ValueStateDescriptor

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
from models import AggregatedStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeasurementAccumulator:
    """측정값 누적기"""

    def __init__(self):
        self.count: int = 0
        self.sum_value: float = 0.0
        self.sum_squared: float = 0.0
        self.min_value: Optional[float] = None
        self.max_value: Optional[float] = None
        self.alarm_count: int = 0
        self.warning_count: int = 0
        self.values: List[float] = []  # 표준편차 계산용


class MeasurementAggregateFunction(AggregateFunction):
    """측정값 집계 함수"""

    def create_accumulator(self) -> MeasurementAccumulator:
        return MeasurementAccumulator()

    def add(self, value: Tuple, accumulator: MeasurementAccumulator) -> MeasurementAccumulator:
        """값 추가"""
        # value = (equipment_id, param_id, value, status, timestamp)
        _, _, val, status, _ = value

        if val is not None:
            accumulator.count += 1
            accumulator.sum_value += val
            accumulator.sum_squared += val * val
            accumulator.values.append(val)

            if accumulator.min_value is None or val < accumulator.min_value:
                accumulator.min_value = val
            if accumulator.max_value is None or val > accumulator.max_value:
                accumulator.max_value = val

        if status == "ALARM":
            accumulator.alarm_count += 1
        elif status == "WARNING":
            accumulator.warning_count += 1

        return accumulator

    def get_result(self, accumulator: MeasurementAccumulator) -> MeasurementAccumulator:
        return accumulator

    def merge(self, a: MeasurementAccumulator, b: MeasurementAccumulator) -> MeasurementAccumulator:
        """두 누적기 병합"""
        result = MeasurementAccumulator()
        result.count = a.count + b.count
        result.sum_value = a.sum_value + b.sum_value
        result.sum_squared = a.sum_squared + b.sum_squared
        result.values = a.values + b.values
        result.alarm_count = a.alarm_count + b.alarm_count
        result.warning_count = a.warning_count + b.warning_count

        if a.min_value is not None and b.min_value is not None:
            result.min_value = min(a.min_value, b.min_value)
        else:
            result.min_value = a.min_value or b.min_value

        if a.max_value is not None and b.max_value is not None:
            result.max_value = max(a.max_value, b.max_value)
        else:
            result.max_value = a.max_value or b.max_value

        return result


class StatsProcessWindowFunction(ProcessWindowFunction):
    """윈도우 통계 처리 함수"""

    def process(
        self,
        key: Tuple[str, str],
        context: ProcessWindowFunction.Context,
        elements: Iterable[MeasurementAccumulator],
    ) -> Iterable[str]:
        """윈도우 결과 처리"""
        equipment_id, param_id = key
        window = context.window()

        # 누적기 결과 가져오기
        accumulator = list(elements)[0]

        if accumulator.count == 0:
            return

        # 통계 계산
        avg_value = accumulator.sum_value / accumulator.count

        std_value = None
        if len(accumulator.values) >= 2:
            try:
                std_value = statistics.stdev(accumulator.values)
            except statistics.StatisticsError:
                pass

        # 결과 생성
        stats = AggregatedStats(
            window_start=datetime.fromtimestamp(window.start / 1000),
            window_end=datetime.fromtimestamp(window.end / 1000),
            equipment_id=equipment_id,
            param_id=param_id,
            count=accumulator.count,
            sum_value=accumulator.sum_value,
            min_value=accumulator.min_value,
            max_value=accumulator.max_value,
            avg_value=avg_value,
            std_value=std_value,
            alarm_count=accumulator.alarm_count,
            warning_count=accumulator.warning_count,
        )

        yield serialize_to_json(stats.to_dict())


class MeasurementParser(RichMapFunction):
    """측정 데이터 파서"""

    def map(self, value: str) -> Optional[Tuple[str, str, float, str, int]]:
        """
        JSON → (equipment_id, param_id, value, status, timestamp_ms)
        """
        try:
            data = parse_json_message(value)
            if not data:
                return None

            equipment_id = data.get("equipment_id")
            param_id = data.get("param_id")
            val = data.get("value")
            status = data.get("status", "NORMAL")
            timestamp = data.get("timestamp")

            if not equipment_id or not param_id or val is None:
                return None

            # 타임스탬프 파싱
            ts = parse_timestamp(timestamp)
            timestamp_ms = int(ts.timestamp() * 1000) if ts else int(datetime.utcnow().timestamp() * 1000)

            return (equipment_id, param_id, float(val), status, timestamp_ms)

        except Exception as e:
            logger.error(f"Error parsing measurement: {e}")
            return None


def run_window_aggregation_job():
    """윈도우 집계 잡 실행"""
    logger.info("Starting Window Aggregation Job")

    # 실행 환경 설정
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(flink_config.parallelism)

    # 체크포인트 설정
    env.enable_checkpointing(flink_config.checkpoint_interval)

    # 소스/싱크
    measurement_source = create_kafka_source(
        topics=[kafka_config.enriched_measurement_topic],
        group_id=f"{kafka_config.group_id}-aggregation",
    )

    stats_sink = create_kafka_sink(kafka_config.aggregated_stats_topic)

    # =========================================================
    # 측정 데이터 스트림
    # =========================================================
    measurement_stream = (
        env.from_source(
            measurement_source,
            WatermarkStrategy.no_watermarks(),
            "Enriched Measurement Source",
        )
        .map(MeasurementParser())
        .name("Parse Measurements")
        .filter(lambda x: x is not None)
        .name("Filter Nulls")
    )

    # =========================================================
    # 5분 텀블링 윈도우 집계
    # =========================================================
    five_min_stats = (
        measurement_stream
        .key_by(lambda x: (x[0], x[1]))  # (equipment_id, param_id)
        .window(TumblingProcessingTimeWindows.of(Time.minutes(5)))
        .aggregate(
            MeasurementAggregateFunction(),
            StatsProcessWindowFunction(),
        )
        .name("5-Minute Window Aggregation")
    )

    # =========================================================
    # 1시간 텀블링 윈도우 집계
    # =========================================================
    one_hour_stats = (
        measurement_stream
        .key_by(lambda x: (x[0], x[1]))  # (equipment_id, param_id)
        .window(TumblingProcessingTimeWindows.of(Time.hours(1)))
        .aggregate(
            MeasurementAggregateFunction(),
            StatsProcessWindowFunction(),
        )
        .name("1-Hour Window Aggregation")
    )

    # =========================================================
    # 통합 출력
    # =========================================================
    (
        five_min_stats
        .union(one_hour_stats)
        .sink_to(stats_sink)
        .name("Aggregated Stats Sink")
    )

    # 실행
    env.execute("Window Aggregation Job")


if __name__ == "__main__":
    run_window_aggregation_job()
