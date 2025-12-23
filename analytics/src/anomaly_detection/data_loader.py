"""
데이터 로딩 및 전처리 모듈

FDC PV 데이터와 이상 발생 시점 정보를 로딩하고 전처리합니다.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy import stats


@dataclass
class DataCharacteristics:
    """데이터 특성 정보"""
    n_samples: int
    n_features: int
    anomaly_ratio: float
    temporal_dependency: float
    noise_level: float
    has_trend: bool
    has_seasonality: bool
    missing_ratio: float
    feature_correlations: float


class DataLoader:
    """FDC 데이터 로더 클래스"""

    def __init__(self):
        self.pv_data: Optional[pd.DataFrame] = None
        self.fault_labels: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        self.characteristics: Optional[DataCharacteristics] = None

    def load_pv_data(
        self,
        filepath: str,
        timestamp_col: str = 'timestamp',
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """PV 데이터 로딩"""
        self.pv_data = pd.read_csv(filepath)

        if parse_dates and timestamp_col in self.pv_data.columns:
            self.pv_data[timestamp_col] = pd.to_datetime(self.pv_data[timestamp_col])
            self.pv_data = self.pv_data.sort_values(timestamp_col).reset_index(drop=True)

        print(f"PV 데이터 로드 완료: {len(self.pv_data)} 샘플, {len(self.pv_data.columns)} 컬럼")
        return self.pv_data

    def load_fault_labels(
        self,
        filepath: str,
        timestamp_col: str = 'start_timestamp',
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """이상 발생 시점 정보 로딩"""
        self.fault_labels = pd.read_csv(filepath)

        if parse_dates:
            for col in ['start_timestamp', 'end_timestamp']:
                if col in self.fault_labels.columns:
                    self.fault_labels[col] = pd.to_datetime(self.fault_labels[col])

        print(f"이상 발생 정보 로드 완료: {len(self.fault_labels)} 이벤트")
        return self.fault_labels

    def merge_data(
        self,
        pv_data: Optional[pd.DataFrame] = None,
        fault_labels: Optional[pd.DataFrame] = None,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """PV 데이터와 이상 라벨 병합"""
        if pv_data is not None:
            self.pv_data = pv_data
        if fault_labels is not None:
            self.fault_labels = fault_labels

        if self.pv_data is None:
            raise ValueError("PV 데이터가 로드되지 않았습니다.")

        self.merged_data = self.pv_data.copy()

        # is_anomaly 컬럼이 없으면 생성
        if 'is_anomaly' not in self.merged_data.columns:
            self.merged_data['is_anomaly'] = 0

            if self.fault_labels is not None:
                for _, fault in self.fault_labels.iterrows():
                    mask = (
                        (self.merged_data[timestamp_col] >= fault['start_timestamp']) &
                        (self.merged_data[timestamp_col] <= fault['end_timestamp'])
                    )
                    self.merged_data.loc[mask, 'is_anomaly'] = 1

        anomaly_count = self.merged_data['is_anomaly'].sum()
        print(f"데이터 병합 완료: 이상 샘플 {anomaly_count}개 ({anomaly_count/len(self.merged_data)*100:.2f}%)")

        return self.merged_data

    def preprocess(
        self,
        data: Optional[pd.DataFrame] = None,
        fill_missing: str = 'interpolate',
        remove_outliers: bool = False,
        outlier_threshold: float = 5.0
    ) -> pd.DataFrame:
        """데이터 전처리"""
        if data is not None:
            df = data.copy()
        elif self.merged_data is not None:
            df = self.merged_data.copy()
        else:
            raise ValueError("전처리할 데이터가 없습니다.")

        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'is_anomaly' in numeric_cols:
            numeric_cols.remove('is_anomaly')

        # 결측치 처리
        if fill_missing == 'interpolate':
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        elif fill_missing == 'ffill':
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        elif fill_missing == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # 남은 결측치는 0으로 채움
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # 이상치 제거 (옵션)
        if remove_outliers:
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col]))
                df.loc[z_scores > outlier_threshold, col] = np.nan
                df[col] = df[col].interpolate(method='linear')

        self.merged_data = df
        print(f"전처리 완료: 결측치 처리={fill_missing}, 이상치 제거={remove_outliers}")

        return df

    def analyze_characteristics(
        self,
        data: Optional[pd.DataFrame] = None
    ) -> DataCharacteristics:
        """데이터 특성 분석"""
        if data is not None:
            df = data
        elif self.merged_data is not None:
            df = self.merged_data
        else:
            raise ValueError("분석할 데이터가 없습니다.")

        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['is_anomaly']
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        # 기본 특성
        n_samples = len(df)
        n_features = len(feature_cols)

        # 이상 비율
        anomaly_ratio = df['is_anomaly'].mean() if 'is_anomaly' in df.columns else 0.0

        # 시계열 의존성 (자기상관)
        temporal_dependency = self._calculate_temporal_dependency(df[feature_cols])

        # 노이즈 수준
        noise_level = self._calculate_noise_level(df[feature_cols])

        # 트렌드 존재 여부
        has_trend = self._check_trend(df[feature_cols])

        # 계절성 존재 여부
        has_seasonality = self._check_seasonality(df[feature_cols])

        # 결측 비율
        missing_ratio = df[feature_cols].isna().mean().mean()

        # 특성 간 상관관계
        feature_correlations = self._calculate_correlations(df[feature_cols])

        self.characteristics = DataCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            anomaly_ratio=anomaly_ratio,
            temporal_dependency=temporal_dependency,
            noise_level=noise_level,
            has_trend=has_trend,
            has_seasonality=has_seasonality,
            missing_ratio=missing_ratio,
            feature_correlations=feature_correlations
        )

        return self.characteristics

    def _calculate_temporal_dependency(self, data: pd.DataFrame) -> float:
        """시계열 의존성 계산 (평균 자기상관)"""
        autocorrs = []
        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 10:
                autocorr = series.autocorr(lag=1)
                if not np.isnan(autocorr):
                    autocorrs.append(abs(autocorr))
        return np.mean(autocorrs) if autocorrs else 0.0

    def _calculate_noise_level(self, data: pd.DataFrame) -> float:
        """노이즈 수준 계산 (변동계수 기반)"""
        cvs = []
        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 0 and series.mean() != 0:
                cv = series.std() / abs(series.mean())
                cvs.append(cv)
        return np.mean(cvs) if cvs else 0.0

    def _check_trend(self, data: pd.DataFrame) -> bool:
        """트렌드 존재 여부 확인"""
        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 100:
                # 간단한 선형 회귀로 트렌드 확인
                x = np.arange(len(series))
                slope, _, r_value, _, _ = stats.linregress(x, series)
                if abs(r_value) > 0.5:
                    return True
        return False

    def _check_seasonality(self, data: pd.DataFrame) -> bool:
        """계절성 존재 여부 확인 (FFT 기반)"""
        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 1000:
                # FFT로 주요 주파수 성분 확인
                fft_values = np.abs(np.fft.fft(series.values))
                fft_values = fft_values[1:len(fft_values)//2]  # 양의 주파수만

                if len(fft_values) > 0:
                    # 주요 피크가 있는지 확인
                    threshold = np.mean(fft_values) + 3 * np.std(fft_values)
                    if np.any(fft_values > threshold):
                        return True
        return False

    def _calculate_correlations(self, data: pd.DataFrame) -> float:
        """특성 간 평균 상관계수"""
        if len(data.columns) < 2:
            return 0.0

        corr_matrix = data.corr().abs()
        # 대각선 제외한 상삼각 행렬의 평균
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack().values
        return np.mean(correlations) if len(correlations) > 0 else 0.0

    def get_train_test_split(
        self,
        data: Optional[pd.DataFrame] = None,
        test_ratio: float = 0.2,
        temporal_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """학습/테스트 데이터 분할"""
        if data is not None:
            df = data
        elif self.merged_data is not None:
            df = self.merged_data
        else:
            raise ValueError("분할할 데이터가 없습니다.")

        if temporal_split:
            # 시계열 순서 유지 분할
            split_idx = int(len(df) * (1 - test_ratio))
            train_data = df.iloc[:split_idx].copy()
            test_data = df.iloc[split_idx:].copy()
        else:
            # 랜덤 분할
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                df, test_size=test_ratio, random_state=42
            )

        print(f"데이터 분할: 학습 {len(train_data)}개, 테스트 {len(test_data)}개")
        return train_data, test_data

    def get_feature_columns(self, exclude_cols: list = None) -> list:
        """특성 컬럼 목록 반환"""
        if self.merged_data is None:
            return []

        if exclude_cols is None:
            exclude_cols = ['timestamp', 'is_anomaly']

        numeric_cols = self.merged_data.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude_cols]

    def summary(self) -> Dict[str, Any]:
        """데이터 요약 정보"""
        if self.merged_data is None:
            return {"error": "데이터가 로드되지 않았습니다."}

        if self.characteristics is None:
            self.analyze_characteristics()

        feature_cols = self.get_feature_columns()

        return {
            "샘플 수": self.characteristics.n_samples,
            "특성 수": self.characteristics.n_features,
            "이상 비율": f"{self.characteristics.anomaly_ratio*100:.2f}%",
            "시계열 의존성": f"{self.characteristics.temporal_dependency:.3f}",
            "노이즈 수준": f"{self.characteristics.noise_level:.3f}",
            "트렌드 존재": self.characteristics.has_trend,
            "계절성 존재": self.characteristics.has_seasonality,
            "결측 비율": f"{self.characteristics.missing_ratio*100:.2f}%",
            "특성 상관도": f"{self.characteristics.feature_correlations:.3f}",
            "특성 컬럼": feature_cols
        }
