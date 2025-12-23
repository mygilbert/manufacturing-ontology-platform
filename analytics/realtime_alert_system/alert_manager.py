"""
경보 관리자 - 콘솔 출력, 로그 파일 기록
"""

import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from .config import AlertConfig, AlertLevel, ALERT_COLORS, RESET_COLOR
from .models import Alert, DetectionResult, SeverityLevel


class AlertManager:
    """경보 관리자"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.alert_history: List[Alert] = []
        self.alerts_by_severity: Dict[str, int] = {
            "WARNING": 0,
            "CRITICAL": 0,
            "EMERGENCY": 0
        }

        # 로그 디렉토리 생성
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            config.LOG_DIR
        )
        os.makedirs(log_dir, exist_ok=True)

        # 로거 설정
        self.logger = self._setup_logger(log_dir)

    def _setup_logger(self, log_dir: str) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("AlertManager")
        logger.setLevel(logging.INFO)

        # 기존 핸들러 제거
        logger.handlers = []

        # 파일 핸들러 (JSON 형식)
        log_path = os.path.join(log_dir, self.config.LOG_FILE)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=self.config.LOG_MAX_BYTES,
            backupCount=self.config.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)

        # JSON 포맷터
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                if hasattr(record, 'alert_data'):
                    return json.dumps(record.alert_data, ensure_ascii=False, default=str)
                return super().format(record)

        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

        return logger

    def process_detection(
        self,
        detection: DetectionResult,
        sensor_values: Dict[str, float]
    ) -> Optional[Alert]:
        """
        이상 감지 결과 처리 및 경보 발생

        Args:
            detection: 감지 결과
            sensor_values: 센서 값

        Returns:
            Alert 또는 None (정상인 경우)
        """
        if detection.severity == SeverityLevel.NORMAL:
            return None

        # 경보 생성
        alert = Alert(
            id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            timestamp=detection.timestamp,
            severity=detection.severity,
            ensemble_score=detection.ensemble_score,
            algorithm_votes=sum(detection.individual_predictions.values()),
            sensor_values=sensor_values,
            individual_scores=detection.individual_scores,
            message=self._generate_message(detection, sensor_values),
            window_index=detection.window_index
        )

        # 콘솔 출력
        self._console_alert(alert)

        # 로그 기록
        self._log_alert(alert)

        # 히스토리 저장
        self.alert_history.append(alert)
        self.alerts_by_severity[detection.severity.value] += 1

        # 히스토리 크기 제한 (최근 10000개만 유지)
        if len(self.alert_history) > 10000:
            self.alert_history = self.alert_history[-10000:]

        return alert

    def _generate_message(
        self,
        detection: DetectionResult,
        sensor_values: Dict[str, float]
    ) -> str:
        """경보 메시지 생성"""
        # 가장 이상 점수가 높은 알고리즘 찾기
        max_algo = max(detection.individual_scores, key=detection.individual_scores.get)
        max_score = detection.individual_scores[max_algo]

        # 가장 이상한 센서 찾기 (평균에서 가장 벗어난)
        abnormal_sensors = []
        for sensor, value in sensor_values.items():
            abnormal_sensors.append((sensor, value))

        # 메시지 구성
        severity_text = {
            SeverityLevel.WARNING: "주의 필요",
            SeverityLevel.CRITICAL: "즉시 확인 필요",
            SeverityLevel.EMERGENCY: "긴급 대응 필요"
        }

        votes = sum(detection.individual_predictions.values())
        message = f"{severity_text.get(detection.severity, '')} - "
        message += f"{votes}/4 알고리즘 이상 판정, "
        message += f"최고점수: {max_algo}({max_score:.2f})"

        return message

    def _console_alert(self, alert: Alert) -> None:
        """콘솔 경보 출력"""
        color = ALERT_COLORS.get(alert.severity.value, "")
        reset = RESET_COLOR

        # 경보 헤더
        print(f"\n{color}{'='*60}")
        print(f"[{alert.severity.value}] {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}{reset}")

        # 점수 정보
        print(f"  Ensemble Score: {alert.ensemble_score:.3f}")
        print(f"  Algorithm Votes: {alert.algorithm_votes}/4")

        # 알고리즘별 점수
        print(f"  Individual Scores:")
        for algo, score in alert.individual_scores.items():
            print(f"    - {algo}: {score:.3f}")

        # 센서 값
        print(f"  Sensor Values:")
        for sensor, value in alert.sensor_values.items():
            short_name = sensor.replace('SVM_Z_', '')
            print(f"    - {short_name}: {value:.2f}")

        # 메시지
        print(f"\n  {color}Message: {alert.message}{reset}")
        print(f"{color}{'='*60}{reset}\n")

    def _log_alert(self, alert: Alert) -> None:
        """로그 파일에 경보 기록"""
        alert_data = {
            "id": alert.id,
            "timestamp": alert.timestamp.isoformat(),
            "severity": alert.severity.value,
            "ensemble_score": alert.ensemble_score,
            "algorithm_votes": alert.algorithm_votes,
            "individual_scores": alert.individual_scores,
            "sensor_values": alert.sensor_values,
            "message": alert.message,
            "window_index": alert.window_index
        }

        record = logging.LogRecord(
            name="AlertManager",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None
        )
        record.alert_data = alert_data
        self.logger.handle(record)

    def get_history(
        self,
        limit: int = 100,
        severity: Optional[SeverityLevel] = None
    ) -> List[Alert]:
        """경보 이력 조회"""
        history = self.alert_history

        if severity:
            history = [a for a in history if a.severity == severity]

        return history[-limit:]

    def get_statistics(self) -> Dict:
        """경보 통계"""
        return {
            "total_alerts": len(self.alert_history),
            "by_severity": self.alerts_by_severity.copy(),
            "recent_10": [
                {
                    "id": a.id,
                    "timestamp": a.timestamp.isoformat(),
                    "severity": a.severity.value,
                    "score": a.ensemble_score
                }
                for a in self.alert_history[-10:]
            ]
        }

    def reset(self) -> None:
        """상태 초기화"""
        self.alert_history = []
        self.alerts_by_severity = {
            "WARNING": 0,
            "CRITICAL": 0,
            "EMERGENCY": 0
        }
