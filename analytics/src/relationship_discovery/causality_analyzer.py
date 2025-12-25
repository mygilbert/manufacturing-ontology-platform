"""
Causality Analyzer
==================

상관관계를 넘어 실제 인과관계를 분석

지원 분석 방법:
- Granger Causality: 시계열 예측력 기반 인과성
- Transfer Entropy: 정보 이론 기반 인과성
- Convergent Cross Mapping (CCM): 비선형 동적 시스템용
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from itertools import permutations
import warnings

from .config import CausalityConfig, RelationType


@dataclass
class CausalityResult:
    """인과성 분석 결과"""
    source_param: str
    target_param: str
    method: str
    statistic: float                      # 검정 통계량
    p_value: float
    optimal_lag: int                      # 최적 시간 지연
    lag_seconds: float = 0.0
    is_causal: bool = False               # 인과관계 존재 여부
    direction_strength: float = 0.0       # 인과 방향 강도 (양방향 비교용)
    confidence: float = 0.0
    relation_type: str = RelationType.CAUSES.value
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_param': self.source_param,
            'target_param': self.target_param,
            'method': self.method,
            'statistic': round(self.statistic, 4),
            'p_value': round(self.p_value, 6),
            'optimal_lag': self.optimal_lag,
            'lag_seconds': round(self.lag_seconds, 2),
            'is_causal': self.is_causal,
            'direction_strength': round(self.direction_strength, 4),
            'confidence': round(self.confidence, 4),
            'relation_type': self.relation_type,
        }


class CausalityAnalyzer:
    """인과성 분석기"""

    def __init__(self, config: Optional[CausalityConfig] = None):
        self.config = config or CausalityConfig()
        self.results: List[CausalityResult] = []

    def analyze(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        timestamp_col: str = 'timestamp',
        sample_rate_hz: float = 1.0
    ) -> List[CausalityResult]:
        """
        데이터프레임의 컬럼들 간 인과관계 분석

        Args:
            data: 분석할 데이터프레임
            columns: 분석할 컬럼 목록
            timestamp_col: 타임스탬프 컬럼
            sample_rate_hz: 샘플링 레이트 (Hz)

        Returns:
            인과성 분석 결과 목록
        """
        self.results = []

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if timestamp_col in columns:
                columns.remove(timestamp_col)

        if len(columns) < 2:
            warnings.warn("분석할 컬럼이 2개 미만입니다.")
            return []

        # 순서쌍 (A→B, B→A 모두 검정)
        pairs = list(permutations(columns, 2))
        print(f"분석할 인과 방향: {len(pairs)}개")

        for source, target in pairs:
            x = data[source].values
            y = data[target].values

            # NaN 처리
            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) < self.config.min_samples:
                continue

            # Granger 인과성 검정
            if self.config.use_granger:
                result = self._granger_causality(
                    x_clean, y_clean,
                    source, target,
                    sample_rate_hz
                )
                if result:
                    self.results.append(result)

            # Transfer Entropy
            if self.config.use_transfer_entropy:
                result = self._transfer_entropy(
                    x_clean, y_clean,
                    source, target,
                    sample_rate_hz
                )
                if result:
                    self.results.append(result)

        # 유의미한 결과 필터링
        significant = [r for r in self.results if r.is_causal]
        print(f"발견된 인과관계: {len(significant)}개")

        return significant

    def _granger_causality(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source: str,
        target: str,
        sample_rate_hz: float
    ) -> Optional[CausalityResult]:
        """
        Granger 인과성 검정

        "x가 y를 Granger-cause 한다" = x의 과거값이 y 예측에 도움이 된다
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # 데이터프레임 형태로 변환 (statsmodels 요구사항)
            df = pd.DataFrame({'y': y, 'x': x})

            # 여러 lag에 대해 검정
            max_lag = min(self.config.max_lag, len(x) // 10)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = grangercausalitytests(
                    df[['y', 'x']],  # y가 첫 번째 (종속변수)
                    maxlag=max_lag,
                    verbose=False
                )

            # 최적 lag 찾기 (가장 낮은 p-value)
            best_lag = 1
            best_p_value = 1.0
            best_f_stat = 0.0

            for lag in range(1, max_lag + 1):
                # F-test 결과 사용
                f_test = results[lag][0]['ssr_ftest']
                f_stat, p_value = f_test[0], f_test[1]

                if p_value < best_p_value:
                    best_p_value = p_value
                    best_lag = lag
                    best_f_stat = f_stat

            is_causal = best_p_value < self.config.significance_level
            lag_seconds = best_lag / sample_rate_hz

            # 신뢰도 계산
            confidence = self._compute_confidence(
                best_f_stat, best_p_value, len(x), best_lag
            )

            return CausalityResult(
                source_param=source,
                target_param=target,
                method="granger",
                statistic=best_f_stat,
                p_value=best_p_value,
                optimal_lag=best_lag,
                lag_seconds=lag_seconds,
                is_causal=is_causal,
                confidence=confidence,
                relation_type=RelationType.CAUSES.value if is_causal else RelationType.CORRELATION.value,
                details={
                    'all_lags_tested': max_lag,
                    'test_type': 'ssr_ftest'
                }
            )

        except ImportError:
            warnings.warn("statsmodels가 필요합니다: pip install statsmodels")
            return None
        except Exception as e:
            warnings.warn(f"Granger 인과성 검정 실패 ({source}→{target}): {e}")
            return None

    def _transfer_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        source: str,
        target: str,
        sample_rate_hz: float
    ) -> Optional[CausalityResult]:
        """
        Transfer Entropy 계산

        정보 이론 기반: x에서 y로 전달되는 정보량 측정
        """
        try:
            max_lag = min(self.config.max_lag, len(x) // 20)
            best_te = 0.0
            best_lag = 1

            for lag in range(1, max_lag + 1):
                te = self._compute_transfer_entropy(x, y, lag)
                if te > best_te:
                    best_te = te
                    best_lag = lag

            # 유의성 검정 (셔플 테스트)
            null_dist = []
            n_permutations = 100

            for _ in range(n_permutations):
                x_shuffled = np.random.permutation(x)
                te_null = self._compute_transfer_entropy(x_shuffled, y, best_lag)
                null_dist.append(te_null)

            null_mean = np.mean(null_dist)
            null_std = np.std(null_dist) + 1e-10
            z_score = (best_te - null_mean) / null_std
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

            is_causal = (
                best_te > null_mean + 2 * null_std and
                p_value < self.config.significance_level
            )

            lag_seconds = best_lag / sample_rate_hz
            confidence = min(z_score / 5, 1.0) if z_score > 0 else 0.0

            return CausalityResult(
                source_param=source,
                target_param=target,
                method="transfer_entropy",
                statistic=best_te,
                p_value=p_value,
                optimal_lag=best_lag,
                lag_seconds=lag_seconds,
                is_causal=is_causal,
                confidence=confidence,
                relation_type=RelationType.INFLUENCES.value if is_causal else RelationType.CORRELATION.value,
                details={
                    'z_score': z_score,
                    'null_mean': null_mean,
                    'null_std': null_std
                }
            )

        except Exception as e:
            warnings.warn(f"Transfer Entropy 계산 실패 ({source}→{target}): {e}")
            return None

    def _compute_transfer_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
        n_bins: int = 10
    ) -> float:
        """Transfer Entropy 계산 (이산화 기반)"""
        # 데이터 이산화
        x_binned = pd.qcut(x, n_bins, labels=False, duplicates='drop')
        y_binned = pd.qcut(y, n_bins, labels=False, duplicates='drop')

        n = len(y) - lag

        # 조건부 엔트로피 계산
        # TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})

        y_curr = y_binned[lag:]
        y_prev = y_binned[lag-1:-1] if lag > 1 else y_binned[:-lag]
        x_lagged = x_binned[:-lag]

        # H(Y_t | Y_{t-1})
        h_y_given_yprev = self._conditional_entropy(y_curr, y_prev)

        # H(Y_t | Y_{t-1}, X_{t-lag})
        # 결합 조건 변수 생성
        combined = y_prev * n_bins + x_lagged
        h_y_given_both = self._conditional_entropy(y_curr, combined)

        te = h_y_given_yprev - h_y_given_both
        return max(0, te)  # 음수 방지

    def _conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """조건부 엔트로피 H(X|Y) 계산"""
        # Joint probability
        joint = pd.crosstab(x, y, normalize=True)

        # Marginal probability of Y
        p_y = joint.sum(axis=0)

        # H(X|Y) = -sum(P(x,y) * log(P(x|y)))
        h = 0.0
        for y_val in joint.columns:
            for x_val in joint.index:
                p_xy = joint.loc[x_val, y_val]
                if p_xy > 0 and p_y[y_val] > 0:
                    p_x_given_y = p_xy / p_y[y_val]
                    h -= p_xy * np.log2(p_x_given_y + 1e-10)

        return h

    def _compute_confidence(
        self,
        statistic: float,
        p_value: float,
        n_samples: int,
        lag: int
    ) -> float:
        """신뢰도 점수 계산"""
        # p-value 기여
        p_score = max(0, (1 - p_value / self.config.significance_level)) * 0.4

        # 통계량 크기 기여
        stat_score = min(statistic / 10, 1.0) * 0.3

        # 샘플 수 기여
        sample_score = min(n_samples / 1000, 1.0) * 0.2

        # lag 적절성 (너무 크면 감점)
        lag_score = max(0, 1 - lag / self.config.max_lag) * 0.1

        return p_score + stat_score + sample_score + lag_score

    def find_bidirectional(self) -> List[Tuple[CausalityResult, CausalityResult]]:
        """양방향 인과관계 찾기 (피드백 루프)"""
        bidirectional = []
        causal_pairs = {
            (r.source_param, r.target_param): r
            for r in self.results if r.is_causal
        }

        checked = set()
        for (src, tgt), result in causal_pairs.items():
            if (tgt, src) in causal_pairs and (src, tgt) not in checked:
                bidirectional.append((result, causal_pairs[(tgt, src)]))
                checked.add((src, tgt))
                checked.add((tgt, src))

        return bidirectional

    def get_causal_graph(self) -> Dict[str, List[str]]:
        """인과 그래프 구조 반환"""
        graph = {}
        for result in self.results:
            if result.is_causal:
                if result.source_param not in graph:
                    graph[result.source_param] = []
                graph[result.source_param].append(result.target_param)
        return graph

    def get_results_dataframe(self) -> pd.DataFrame:
        """결과를 데이터프레임으로 반환"""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def summary(self) -> Dict[str, Any]:
        """분석 결과 요약"""
        if not self.results:
            return {"status": "no_results"}

        causal = [r for r in self.results if r.is_causal]
        bidirectional = self.find_bidirectional()

        return {
            "total_tested": len(self.results),
            "causal_found": len(causal),
            "bidirectional_pairs": len(bidirectional),
            "avg_lag": np.mean([r.optimal_lag for r in causal]) if causal else 0,
            "high_confidence": len([r for r in causal if r.confidence > 0.7]),
            "methods_used": list(set(r.method for r in self.results)),
        }
