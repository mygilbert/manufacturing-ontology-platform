"""
실시간 데이터 시뮬레이터
CSV 파일에서 데이터를 읽어 실시간으로 스트리밍
"""

import asyncio
import pandas as pd
import numpy as np
from typing import AsyncGenerator, List, Optional, Dict
from datetime import datetime

from .config import AlertConfig


class RealTimeDataSimulator:
    """CSV 기반 실시간 데이터 시뮬레이터"""

    def __init__(
        self,
        csv_path: str,
        config: AlertConfig,
        speed_multiplier: int = 60
    ):
        """
        초기화

        Args:
            csv_path: CSV 파일 경로
            config: 설정
            speed_multiplier: 속도 배율 (60 = 1초당 60개 데이터)
        """
        self.csv_path = csv_path
        self.config = config
        self.speed = speed_multiplier
        self.sensor_columns = config.SENSOR_COLUMNS

        # 데이터 로드
        print(f"[Simulator] Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"[Simulator] Loaded {len(self.df):,} rows")

        # 센서 컬럼 확인
        available_cols = [c for c in self.sensor_columns if c in self.df.columns]
        if len(available_cols) < len(self.sensor_columns):
            missing = set(self.sensor_columns) - set(available_cols)
            print(f"[Simulator] Warning: Missing columns: {missing}")
        self.sensor_columns = available_cols

        # 결측치 처리
        self.df = self.df[self.sensor_columns].dropna()
        print(f"[Simulator] After cleaning: {len(self.df):,} rows")

        # 인덱스
        self.current_idx = 0
        self.total_samples = len(self.df)
        self.is_running = False

    def get_training_data(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        초기 학습용 데이터 반환

        Args:
            n_samples: 샘플 수 (None이면 설정값 사용)

        Returns:
            학습용 데이터 배열
        """
        if n_samples is None:
            n_samples = self.config.INITIAL_TRAINING_SAMPLES

        n_samples = min(n_samples, len(self.df))

        # 앞부분 데이터 사용 (정상 데이터로 가정)
        training_df = self.df.iloc[:n_samples]

        # 1분 집계 특성 생성
        features = self._create_aggregated_features(training_df, n_samples)

        print(f"[Simulator] Training data: {len(features)} aggregated samples")
        return features

    def _create_aggregated_features(
        self,
        df: pd.DataFrame,
        total_samples: int
    ) -> np.ndarray:
        """1분 집계 특성 생성"""
        window_size = self.config.AGGREGATION_WINDOW_SEC
        n_windows = total_samples // window_size

        features = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_data = df.iloc[start:end]

            if len(window_data) == 0:
                continue

            # 각 센서별 mean, std, range 추출
            window_features = []
            for col in self.sensor_columns:
                col_data = window_data[col].values
                window_features.extend([
                    np.mean(col_data),
                    np.std(col_data),
                    np.max(col_data) - np.min(col_data)
                ])

            features.append(window_features)

        return np.array(features)

    async def stream(
        self,
        start_idx: Optional[int] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        비동기 데이터 스트리밍

        Args:
            start_idx: 시작 인덱스 (None이면 학습 데이터 이후부터)

        Yields:
            1분 집계 데이터
        """
        if start_idx is None:
            start_idx = self.config.INITIAL_TRAINING_SAMPLES

        self.current_idx = start_idx
        self.is_running = True
        window_size = self.config.AGGREGATION_WINDOW_SEC

        print(f"[Simulator] Starting stream from index {start_idx}")
        print(f"[Simulator] Speed: {self.speed}x (1 sec = {self.speed} samples)")

        window_count = 0

        while self.is_running and self.current_idx < self.total_samples:
            # 1분치 데이터 추출
            end_idx = min(self.current_idx + window_size, self.total_samples)
            window_data = self.df.iloc[self.current_idx:end_idx]

            if len(window_data) == 0:
                break

            # 집계
            aggregated = self._aggregate_window(window_data)
            aggregated['window_index'] = window_count
            aggregated['sample_start'] = self.current_idx
            aggregated['sample_end'] = end_idx
            aggregated['progress'] = end_idx / self.total_samples * 100

            yield aggregated

            # 인덱스 이동
            self.current_idx = end_idx
            window_count += 1

            # 속도 조절 (60배속 = 1초에 60개 = 1분 데이터)
            await asyncio.sleep(window_size / self.speed)

        self.is_running = False
        print(f"[Simulator] Stream ended. Processed {window_count} windows")

    def _aggregate_window(self, window_data: pd.DataFrame) -> Dict:
        """윈도우 집계"""
        result = {
            'timestamp': datetime.now(),
            'sample_count': len(window_data),
            'values': {},
            'std_values': {},
            'min_values': {},
            'max_values': {},
            'range_values': {}
        }

        for col in self.sensor_columns:
            col_data = window_data[col].values
            result['values'][col] = float(np.mean(col_data))
            result['std_values'][col] = float(np.std(col_data))
            result['min_values'][col] = float(np.min(col_data))
            result['max_values'][col] = float(np.max(col_data))
            result['range_values'][col] = float(np.max(col_data) - np.min(col_data))

        # 특성 벡터 (감지용)
        feature_vector = []
        for col in self.sensor_columns:
            feature_vector.extend([
                result['values'][col],
                result['std_values'][col],
                result['range_values'][col]
            ])
        result['feature_vector'] = feature_vector

        return result

    def stop(self):
        """스트리밍 중지"""
        self.is_running = False

    def reset(self, start_idx: int = 0):
        """인덱스 리셋"""
        self.current_idx = start_idx
        self.is_running = False

    def get_status(self) -> Dict:
        """현재 상태"""
        return {
            'current_idx': self.current_idx,
            'total_samples': self.total_samples,
            'progress': self.current_idx / self.total_samples * 100,
            'is_running': self.is_running,
            'speed': self.speed
        }
