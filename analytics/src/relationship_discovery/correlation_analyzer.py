"""
Correlation Analyzer
====================

PV 파라미터 간 상관관계를 분석하여 암묵적 관계를 발견

지원 분석 방법:
- Pearson 상관계수: 선형 관계
- Spearman 상관계수: 단조 관계 (비선형 포함)
- Kendall 상관계수: 순위 기반 (이상치에 강함)
- Cross-correlation: 시간 지연 관계
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from .config import CorrelationConfig, RelationType


@dataclass
class CorrelationResult:
    """상관관계 분석 결과"""
    source_param: str
    target_param: str
    method: str
    correlation: float
    p_value: float
    n_samples: int
    lag: int = 0                          # 시간 지연 (샘플 단위)
    lag_seconds: float = 0.0              # 시간 지연 (초)
    is_significant: bool = False
    confidence: float = 0.0
    relation_type: str = RelationType.CORRELATION.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_param': self.source_param,
            'target_param': self.target_param,
            'method': self.method,
            'correlation': round(self.correlation, 4),
            'p_value': round(self.p_value, 6),
            'n_samples': self.n_samples,
            'lag': self.lag,
            'lag_seconds': round(self.lag_seconds, 2),
            'is_significant': self.is_significant,
            'confidence': round(self.confidence, 4),
            'relation_type': self.relation_type,
        }


class CorrelationAnalyzer:
    """상관관계 분석기"""

    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self.results: List[CorrelationResult] = []

    def analyze(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        timestamp_col: str = 'timestamp',
        sample_rate_hz: float = 1.0
    ) -> List[CorrelationResult]:
        """
        데이터프레임의 컬럼들 간 상관관계 분석

        Args:
            data: 분석할 데이터프레임
            columns: 분석할 컬럼 목록 (None이면 숫자형 전체)
            timestamp_col: 타임스탬프 컬럼
            sample_rate_hz: 샘플링 레이트 (Hz)

        Returns:
            상관관계 결과 목록
        """
        self.results = []

        # 숫자형 컬럼만 선택
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if timestamp_col in columns:
                columns.remove(timestamp_col)

        if len(columns) < 2:
            warnings.warn("분석할 컬럼이 2개 미만입니다.")
            return []

        # 모든 컬럼 쌍에 대해 분석
        pairs = list(combinations(columns, 2))
        print(f"분석할 파라미터 쌍: {len(pairs)}개")

        for source, target in pairs:
            pair_results = self._analyze_pair(
                data[source].values,
                data[target].values,
                source,
                target,
                sample_rate_hz
            )
            self.results.extend(pair_results)

        # 유의미한 결과만 필터링
        significant_results = [r for r in self.results if r.is_significant]
        print(f"발견된 유의미한 관계: {len(significant_results)}개")

        return significant_results

    def _analyze_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source_name: str,
        target_name: str,
        sample_rate_hz: float
    ) -> List[CorrelationResult]:
        """두 파라미터 간 상관관계 분석"""
        results = []

        # NaN 제거
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        n_samples = len(x_clean)
        if n_samples < self.config.min_samples:
            return results

        # 각 방법별 분석
        for method in self.config.methods:
            result = self._compute_correlation(
                x_clean, y_clean,
                source_name, target_name,
                method, n_samples
            )
            if result:
                results.append(result)

        # Cross-correlation (시간 지연 분석)
        lag_result = self._compute_cross_correlation(
            x_clean, y_clean,
            source_name, target_name,
            n_samples, sample_rate_hz
        )
        if lag_result:
            results.append(lag_result)

        return results

    def _compute_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source: str,
        target: str,
        method: str,
        n_samples: int
    ) -> Optional[CorrelationResult]:
        """상관계수 계산"""
        try:
            if method == "pearson":
                corr, p_value = stats.pearsonr(x, y)
            elif method == "spearman":
                corr, p_value = stats.spearmanr(x, y)
            elif method == "kendall":
                corr, p_value = stats.kendalltau(x, y)
            else:
                return None

            is_significant = (
                abs(corr) >= self.config.min_correlation and
                p_value <= self.config.max_p_value
            )

            # 신뢰도 계산 (상관계수 크기 + p-value 기반)
            confidence = self._compute_confidence(corr, p_value, n_samples)

            return CorrelationResult(
                source_param=source,
                target_param=target,
                method=method,
                correlation=corr,
                p_value=p_value,
                n_samples=n_samples,
                is_significant=is_significant,
                confidence=confidence
            )

        except Exception as e:
            warnings.warn(f"상관관계 계산 실패 ({source}-{target}): {e}")
            return None

    def _compute_cross_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source: str,
        target: str,
        n_samples: int,
        sample_rate_hz: float
    ) -> Optional[CorrelationResult]:
        """교차 상관관계 (시간 지연) 분석"""
        try:
            # 정규화
            x_norm = (x - np.mean(x)) / (np.std(x) + 1e-10)
            y_norm = (y - np.mean(y)) / (np.std(y) + 1e-10)

            # 교차 상관
            cross_corr = correlate(x_norm, y_norm, mode='full')
            cross_corr = cross_corr / n_samples

            # 최대 상관 지점 찾기
            mid = len(cross_corr) // 2
            max_lag = min(100, mid)  # 최대 100 샘플 지연까지만

            # 양방향 탐색
            search_range = cross_corr[mid - max_lag:mid + max_lag + 1]
            best_idx = np.argmax(np.abs(search_range))
            best_lag = best_idx - max_lag
            best_corr = search_range[best_idx]

            # 지연이 있는 경우만 (lag=0은 일반 상관분석과 동일)
            if abs(best_lag) < 2:
                return None

            # 유의성 검정 (Bootstrap 기반 간소화)
            # 실제로는 더 정교한 검정 필요
            p_value = 2 * (1 - stats.norm.cdf(abs(best_corr) * np.sqrt(n_samples)))

            is_significant = (
                abs(best_corr) >= self.config.min_correlation and
                p_value <= self.config.max_p_value
            )

            if not is_significant:
                return None

            lag_seconds = best_lag / sample_rate_hz

            # 방향성 결정: 양의 lag면 source가 target에 선행
            if best_lag > 0:
                relation_type = RelationType.INFLUENCES.value
            else:
                relation_type = RelationType.INFLUENCES.value
                # 방향 뒤집기
                source, target = target, source
                best_lag = -best_lag
                lag_seconds = -lag_seconds

            confidence = self._compute_confidence(best_corr, p_value, n_samples)

            return CorrelationResult(
                source_param=source,
                target_param=target,
                method="cross_correlation",
                correlation=best_corr,
                p_value=p_value,
                n_samples=n_samples,
                lag=abs(best_lag),
                lag_seconds=abs(lag_seconds),
                is_significant=True,
                confidence=confidence,
                relation_type=relation_type
            )

        except Exception as e:
            warnings.warn(f"교차 상관 계산 실패 ({source}-{target}): {e}")
            return None

    def _compute_confidence(
        self,
        corr: float,
        p_value: float,
        n_samples: int
    ) -> float:
        """신뢰도 점수 계산 (0~1)"""
        # 상관계수 기여 (0~0.5)
        corr_score = min(abs(corr), 1.0) * 0.5

        # p-value 기여 (0~0.3)
        p_score = max(0, (1 - p_value / self.config.max_p_value)) * 0.3

        # 샘플 수 기여 (0~0.2)
        sample_score = min(n_samples / 1000, 1.0) * 0.2

        return corr_score + p_score + sample_score

    def analyze_rolling(
        self,
        data: pd.DataFrame,
        columns: List[str],
        window_size: int,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """시간에 따른 상관관계 변화 분석"""
        if len(columns) < 2:
            return pd.DataFrame()

        results = []
        timestamps = data[timestamp_col].values if timestamp_col in data.columns else range(len(data))

        for i in range(window_size, len(data)):
            window = data.iloc[i - window_size:i]
            window_time = timestamps[i]

            for source, target in combinations(columns, 2):
                try:
                    corr, p_value = stats.pearsonr(
                        window[source].dropna(),
                        window[target].dropna()
                    )
                    results.append({
                        'timestamp': window_time,
                        'source': source,
                        'target': target,
                        'correlation': corr,
                        'p_value': p_value
                    })
                except:
                    pass

        return pd.DataFrame(results)

    def get_correlation_matrix(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """상관관계 행렬 반환"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        return data[columns].corr(method=method)

    def find_highly_correlated(
        self,
        data: pd.DataFrame,
        threshold: float = 0.8,
        columns: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """고상관 파라미터 쌍 찾기 (다중공선성 제거용)"""
        corr_matrix = self.get_correlation_matrix(data, columns)

        highly_correlated = []
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    highly_correlated.append((
                        cols[i],
                        cols[j],
                        corr_matrix.iloc[i, j]
                    ))

        return sorted(highly_correlated, key=lambda x: abs(x[2]), reverse=True)

    def get_results_dataframe(self) -> pd.DataFrame:
        """결과를 데이터프레임으로 반환"""
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame([r.to_dict() for r in self.results])

    def summary(self) -> Dict[str, Any]:
        """분석 결과 요약"""
        if not self.results:
            return {"status": "no_results"}

        significant = [r for r in self.results if r.is_significant]
        df = self.get_results_dataframe()

        return {
            "total_analyzed": len(self.results),
            "significant_found": len(significant),
            "methods_used": df['method'].unique().tolist() if len(df) > 0 else [],
            "avg_correlation": df['correlation'].abs().mean() if len(df) > 0 else 0,
            "max_correlation": df['correlation'].abs().max() if len(df) > 0 else 0,
            "with_time_lag": len([r for r in significant if r.lag > 0]),
            "high_confidence": len([r for r in significant if r.confidence > 0.7]),
        }
