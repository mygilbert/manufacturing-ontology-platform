"""
설정 모듈 - 경보 임계값, 알고리즘 파라미터, 시스템 설정
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AlertConfig:
    """경보 시스템 설정"""

    # 경보 임계값 (앙상블 점수 기준)
    SCORE_THRESHOLD_WARNING: float = 0.6      # 주의 (노랑)
    SCORE_THRESHOLD_CRITICAL: float = 0.8     # 경고 (주황)
    SCORE_THRESHOLD_EMERGENCY: float = 0.95   # 위험 (빨강)

    # Z-Score 설정
    ZSCORE_THRESHOLD: float = 3.0

    # CUSUM 설정
    CUSUM_THRESHOLD: float = 5.0
    CUSUM_DRIFT: float = 0.5

    # Isolation Forest 설정
    IF_CONTAMINATION: float = 0.05
    IF_N_ESTIMATORS: int = 100

    # LOF 설정
    LOF_N_NEIGHBORS: int = 20
    LOF_CONTAMINATION: float = 0.05

    # 데이터 집계 설정
    AGGREGATION_WINDOW_SEC: int = 60  # 1분 집계

    # 초기 학습 설정
    INITIAL_TRAINING_SAMPLES: int = 10000  # 초기 학습용 샘플 수

    # 시뮬레이션 설정
    SIMULATION_SPEED: int = 60  # 1초당 60개 데이터 처리 (60배속)

    # 센서 컬럼 설정
    SENSOR_COLUMNS: List[str] = field(default_factory=lambda: [
        'SVM_Z_CURRENT',
        'SVM_Z_EFFECTIVE_LOAD_RATIO',
        'SVM_Z_PEAK_LOAD_RATIO',
        'SVM_Z_POSITION'
    ])

    # 로그 설정
    LOG_DIR: str = "logs"
    LOG_FILE: str = "alerts.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5

    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 데이터 경로 (기본값)
    DATA_PATH: str = r"D:\24346_ESWA_PKG_11_1_AL_FORMING_PRESS_SVM.csv"

    @property
    def log_file_path(self) -> str:
        return os.path.join(self.LOG_DIR, self.LOG_FILE)


# 경보 레벨 정의
class AlertLevel:
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


# 경보 색상 (콘솔 출력용)
ALERT_COLORS = {
    AlertLevel.NORMAL: "\033[92m",      # 녹색
    AlertLevel.WARNING: "\033[93m",     # 노랑
    AlertLevel.CRITICAL: "\033[91m",    # 주황/빨강
    AlertLevel.EMERGENCY: "\033[95m",   # 마젠타
}
RESET_COLOR = "\033[0m"


# 전역 설정 인스턴스
config = AlertConfig()
