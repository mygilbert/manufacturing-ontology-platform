"""
특성 공학 모듈

FDC PV 데이터에서 이상감지를 위한 특성을 추출합니다.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from scipy import stats
from scipy.signal import find_peaks


class FeatureEngineer:
    """특성 추출 클래스"""

    def __init__(self):
        self.feature_names: List[str] = []
        self.original_columns: List[str] = []

    def extract_all_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        window_sizes: List[int] = [5, 10, 30, 60],
        include_fft: bool = True,
        include_peaks: bool = True
    ) -> pd.DataFrame:
        """모든 특성 추출"""
        self.original_columns = feature_cols.copy()
        result = data.copy()

        print("특성 추출 시작...")

        # 1. 이동 통계량
        print("  - 이동 통계량 계산 중...")
        result = self.add_rolling_features(result, feature_cols, window_sizes)

        # 2. 변화율 특성
        print("  - 변화율 특성 계산 중...")
        result = self.add_rate_of_change(result, feature_cols)

        # 3. 차분 특성
        print("  - 차분 특성 계산 중...")
        result = self.add_difference_features(result, feature_cols)

        # 4. 통계적 특성
        print("  - 통계적 특성 계산 중...")
        result = self.add_statistical_features(result, feature_cols, window_sizes)

        # 5. FFT 특성 (옵션)
        if include_fft:
            print("  - 주파수 도메인 특성 계산 중...")
            result = self.add_fft_features(result, feature_cols, window_size=60)

        # 6. 피크 특성 (옵션)
        if include_peaks:
            print("  - 피크 감지 특성 계산 중...")
            result = self.add_peak_features(result, feature_cols, window_size=30)

        # 결측치 처리
        result = result.fillna(method='bfill').fillna(method='ffill').fillna(0)

        self.feature_names = [c for c in result.columns if c not in ['timestamp', 'is_anomaly']]
        print(f"특성 추출 완료: 총 {len(self.feature_names)}개 특성 생성")

        return result

    def add_rolling_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        window_sizes: List[int] = [5, 10, 30]
    ) -> pd.DataFrame:
        """이동 통계량 특성 추가"""
        result = data.copy()

        for col in feature_cols:
            for window in window_sizes:
                # 이동 평균
                result[f'{col}_rolling_mean_{window}'] = result[col].rolling(
                    window=window, min_periods=1
                ).mean()

                # 이동 표준편차
                result[f'{col}_rolling_std_{window}'] = result[col].rolling(
                    window=window, min_periods=1
                ).std()

                # 이동 최대값
                result[f'{col}_rolling_max_{window}'] = result[col].rolling(
                    window=window, min_periods=1
                ).max()

                # 이동 최소값
                result[f'{col}_rolling_min_{window}'] = result[col].rolling(
                    window=window, min_periods=1
                ).min()

                # 이동 범위
                result[f'{col}_rolling_range_{window}'] = (
                    result[f'{col}_rolling_max_{window}'] -
                    result[f'{col}_rolling_min_{window}']
                )

        return result

    def add_rate_of_change(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """변화율 특성 추가"""
        result = data.copy()

        for col in feature_cols:
            # 1차 미분 (변화량)
            result[f'{col}_diff_1'] = result[col].diff(1)

            # 2차 미분 (변화 가속도)
            result[f'{col}_diff_2'] = result[col].diff(2)

            # 백분율 변화
            result[f'{col}_pct_change'] = result[col].pct_change()

            # 절대 변화량
            result[f'{col}_abs_diff'] = result[col].diff(1).abs()

        return result

    def add_difference_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        lags: List[int] = [1, 5, 10]
    ) -> pd.DataFrame:
        """래그 차분 특성 추가"""
        result = data.copy()

        for col in feature_cols:
            for lag in lags:
                # 현재값과 lag 이전 값의 차이
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
                result[f'{col}_diff_lag_{lag}'] = result[col] - result[col].shift(lag)

        return result

    def add_statistical_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        window_sizes: List[int] = [30, 60]
    ) -> pd.DataFrame:
        """통계적 특성 추가"""
        result = data.copy()

        for col in feature_cols:
            for window in window_sizes:
                rolling = result[col].rolling(window=window, min_periods=1)

                # 왜도 (Skewness)
                result[f'{col}_skew_{window}'] = rolling.skew()

                # 첨도 (Kurtosis)
                result[f'{col}_kurt_{window}'] = rolling.kurt()

                # 중앙값
                result[f'{col}_median_{window}'] = rolling.median()

                # 변동계수
                mean = rolling.mean()
                std = rolling.std()
                result[f'{col}_cv_{window}'] = std / (mean.abs() + 1e-10)

                # Z-score (현재값의 표준화)
                result[f'{col}_zscore_{window}'] = (result[col] - mean) / (std + 1e-10)

        return result

    def add_fft_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        window_size: int = 60
    ) -> pd.DataFrame:
        """FFT 기반 주파수 도메인 특성 추가"""
        result = data.copy()

        for col in feature_cols:
            # 윈도우별 FFT 특성
            fft_energy = []
            fft_peak_freq = []
            fft_spectral_entropy = []

            values = result[col].values

            for i in range(len(values)):
                start_idx = max(0, i - window_size + 1)
                window_data = values[start_idx:i+1]

                if len(window_data) >= 4:
                    # FFT 계산
                    fft_vals = np.abs(np.fft.rfft(window_data))

                    # 에너지 (스펙트럼 파워)
                    energy = np.sum(fft_vals ** 2)
                    fft_energy.append(energy)

                    # 주요 주파수
                    if len(fft_vals) > 1:
                        peak_idx = np.argmax(fft_vals[1:]) + 1
                        fft_peak_freq.append(peak_idx / len(window_data))
                    else:
                        fft_peak_freq.append(0)

                    # 스펙트럴 엔트로피
                    fft_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
                    entropy = -np.sum(fft_norm * np.log(fft_norm + 1e-10))
                    fft_spectral_entropy.append(entropy)
                else:
                    fft_energy.append(0)
                    fft_peak_freq.append(0)
                    fft_spectral_entropy.append(0)

            result[f'{col}_fft_energy'] = fft_energy
            result[f'{col}_fft_peak_freq'] = fft_peak_freq
            result[f'{col}_spectral_entropy'] = fft_spectral_entropy

        return result

    def add_peak_features(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        window_size: int = 30
    ) -> pd.DataFrame:
        """피크 감지 특성 추가"""
        result = data.copy()

        for col in feature_cols:
            peak_counts = []
            peak_heights = []

            values = result[col].values

            for i in range(len(values)):
                start_idx = max(0, i - window_size + 1)
                window_data = values[start_idx:i+1]

                if len(window_data) >= 3:
                    # 피크 감지
                    peaks, properties = find_peaks(
                        window_data,
                        height=np.mean(window_data),
                        distance=2
                    )

                    peak_counts.append(len(peaks))

                    if len(peaks) > 0 and 'peak_heights' in properties:
                        peak_heights.append(np.mean(properties['peak_heights']))
                    else:
                        peak_heights.append(0)
                else:
                    peak_counts.append(0)
                    peak_heights.append(0)

            result[f'{col}_peak_count'] = peak_counts
            result[f'{col}_peak_height_avg'] = peak_heights

        return result

    def add_cross_feature_interactions(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        top_n_pairs: int = 5
    ) -> pd.DataFrame:
        """특성 간 상호작용 특성 추가"""
        result = data.copy()

        if len(feature_cols) < 2:
            return result

        # 상관관계가 높은 쌍 선택
        corr_matrix = data[feature_cols].corr().abs()

        pairs = []
        for i, col1 in enumerate(feature_cols):
            for col2 in feature_cols[i+1:]:
                pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:top_n_pairs]

        for col1, col2, _ in top_pairs:
            # 비율
            result[f'{col1}_{col2}_ratio'] = result[col1] / (result[col2].abs() + 1e-10)

            # 곱
            result[f'{col1}_{col2}_product'] = result[col1] * result[col2]

            # 차이
            result[f'{col1}_{col2}_diff'] = result[col1] - result[col2]

        return result

    def get_feature_importance_ranking(
        self,
        data: pd.DataFrame,
        target_col: str = 'is_anomaly'
    ) -> pd.DataFrame:
        """특성 중요도 랭킹 계산 (상관관계 기반)"""
        if target_col not in data.columns:
            raise ValueError(f"타겟 컬럼 '{target_col}'이 데이터에 없습니다.")

        feature_cols = [c for c in data.columns if c not in ['timestamp', target_col]]

        importance = []
        for col in feature_cols:
            if data[col].std() > 0:
                corr = abs(data[col].corr(data[target_col]))
            else:
                corr = 0
            importance.append({'feature': col, 'importance': corr})

        importance_df = pd.DataFrame(importance)
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def select_top_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'is_anomaly',
        top_n: int = 50,
        keep_original: bool = True
    ) -> pd.DataFrame:
        """상위 N개 특성 선택"""
        importance_df = self.get_feature_importance_ranking(data, target_col)

        top_features = importance_df.head(top_n)['feature'].tolist()

        if keep_original:
            top_features = list(set(top_features + self.original_columns))

        # 항상 포함할 컬럼
        keep_cols = ['timestamp', target_col] if 'timestamp' in data.columns else [target_col]
        selected_cols = keep_cols + [f for f in top_features if f in data.columns]

        return data[selected_cols]
