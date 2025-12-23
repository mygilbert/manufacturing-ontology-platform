"""
SPC 관리도 모듈

다양한 관리도 타입과 Western Electric Rules를 구현합니다.

관리도 타입:
- X-bar Chart: 평균 관리도
- R Chart: 범위 관리도
- S Chart: 표준편차 관리도
- Individual Chart: 개별값 관리도
- P Chart: 불량률 관리도
- NP Chart: 불량 개수 관리도
- C Chart: 결점수 관리도
- U Chart: 단위당 결점수 관리도
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.append('/app/src')

from config import config

logger = logging.getLogger(__name__)


# A2, D3, D4 상수 (subgroup size별)
CONTROL_CHART_CONSTANTS = {
    2: {"A2": 1.880, "D3": 0, "D4": 3.267, "d2": 1.128, "A3": 2.659, "B3": 0, "B4": 3.267, "c4": 0.7979},
    3: {"A2": 1.023, "D3": 0, "D4": 2.575, "d2": 1.693, "A3": 1.954, "B3": 0, "B4": 2.568, "c4": 0.8862},
    4: {"A2": 0.729, "D3": 0, "D4": 2.282, "d2": 2.059, "A3": 1.628, "B3": 0, "B4": 2.266, "c4": 0.9213},
    5: {"A2": 0.577, "D3": 0, "D4": 2.115, "d2": 2.326, "A3": 1.427, "B3": 0, "B4": 2.089, "c4": 0.9400},
    6: {"A2": 0.483, "D3": 0, "D4": 2.004, "d2": 2.534, "A3": 1.287, "B3": 0.030, "B4": 1.970, "c4": 0.9515},
    7: {"A2": 0.419, "D3": 0.076, "D4": 1.924, "d2": 2.704, "A3": 1.182, "B3": 0.118, "B4": 1.882, "c4": 0.9594},
    8: {"A2": 0.373, "D3": 0.136, "D4": 1.864, "d2": 2.847, "A3": 1.099, "B3": 0.185, "B4": 1.815, "c4": 0.9650},
    9: {"A2": 0.337, "D3": 0.184, "D4": 1.816, "d2": 2.970, "A3": 1.032, "B3": 0.239, "B4": 1.761, "c4": 0.9693},
    10: {"A2": 0.308, "D3": 0.223, "D4": 1.777, "d2": 3.078, "A3": 0.975, "B3": 0.284, "B4": 1.716, "c4": 0.9727},
}


class ChartType(Enum):
    """관리도 타입"""
    XBAR = "xbar"
    R = "r"
    S = "s"
    INDIVIDUAL = "individual"
    MR = "mr"  # Moving Range
    P = "p"
    NP = "np"
    C = "c"
    U = "u"


class RuleViolation(Enum):
    """Western Electric Rule 위반 타입"""
    RULE1_BEYOND_3SIGMA = "RULE1_BEYOND_3SIGMA"
    RULE2_RUN_ABOVE = "RULE2_RUN_ABOVE"
    RULE2_RUN_BELOW = "RULE2_RUN_BELOW"
    RULE3_TREND_UP = "RULE3_TREND_UP"
    RULE3_TREND_DOWN = "RULE3_TREND_DOWN"
    RULE4_ALTERNATING = "RULE4_ALTERNATING"
    RULE5_2OF3_BEYOND_2SIGMA = "RULE5_2OF3_BEYOND_2SIGMA"
    RULE6_4OF5_BEYOND_1SIGMA = "RULE6_4OF5_BEYOND_1SIGMA"
    RULE7_STRATIFICATION = "RULE7_STRATIFICATION"
    RULE8_MIXTURE = "RULE8_MIXTURE"


@dataclass
class ControlLimits:
    """관리 한계선"""
    ucl: float  # Upper Control Limit
    cl: float   # Center Line
    lcl: float  # Lower Control Limit
    sigma: float = 0.0  # 표준편차


@dataclass
class ChartPoint:
    """관리도 데이터 포인트"""
    index: int
    value: float
    timestamp: Optional[datetime] = None
    subgroup_id: Optional[str] = None
    violations: List[str] = field(default_factory=list)
    status: str = "NORMAL"  # NORMAL, OOC, OOS, TREND, SHIFT


@dataclass
class ControlChartResult:
    """관리도 분석 결과"""
    chart_type: ChartType
    limits: ControlLimits
    points: List[ChartPoint]
    statistics: Dict[str, float]
    violations_summary: Dict[str, int]


class ControlChart:
    """관리도 기본 클래스"""

    def __init__(
        self,
        chart_type: ChartType,
        sigma_level: float = None,
    ):
        self.chart_type = chart_type
        self.sigma_level = sigma_level or config.spc.sigma_level

    def calculate_limits(self, data: np.ndarray, **kwargs) -> ControlLimits:
        """관리 한계선 계산 (서브클래스에서 구현)"""
        raise NotImplementedError

    def check_rules(
        self,
        values: List[float],
        limits: ControlLimits,
    ) -> List[List[str]]:
        """Western Electric Rules 검사"""
        n = len(values)
        violations = [[] for _ in range(n)]

        if n == 0:
            return violations

        ucl = limits.ucl
        lcl = limits.lcl
        cl = limits.cl
        sigma = limits.sigma if limits.sigma > 0 else (ucl - cl) / 3

        one_sigma_upper = cl + sigma
        one_sigma_lower = cl - sigma
        two_sigma_upper = cl + 2 * sigma
        two_sigma_lower = cl - 2 * sigma

        # Rule 1: 1점이 3시그마 초과
        for i, v in enumerate(values):
            if v > ucl or v < lcl:
                violations[i].append(RuleViolation.RULE1_BEYOND_3SIGMA.value)

        # Rule 2: 연속 9점이 중심선 한쪽
        if n >= 9:
            for i in range(n - 8):
                window = values[i:i+9]
                if all(v > cl for v in window):
                    for j in range(i, i + 9):
                        if RuleViolation.RULE2_RUN_ABOVE.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE2_RUN_ABOVE.value)
                elif all(v < cl for v in window):
                    for j in range(i, i + 9):
                        if RuleViolation.RULE2_RUN_BELOW.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE2_RUN_BELOW.value)

        # Rule 3: 연속 6점 증가 또는 감소
        if n >= 6:
            for i in range(n - 5):
                window = values[i:i+6]
                increasing = all(window[j] < window[j+1] for j in range(5))
                decreasing = all(window[j] > window[j+1] for j in range(5))

                if increasing:
                    for j in range(i, i + 6):
                        if RuleViolation.RULE3_TREND_UP.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE3_TREND_UP.value)
                elif decreasing:
                    for j in range(i, i + 6):
                        if RuleViolation.RULE3_TREND_DOWN.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE3_TREND_DOWN.value)

        # Rule 4: 연속 14점 교대
        if n >= 14:
            for i in range(n - 13):
                window = values[i:i+14]
                try:
                    alternating = all(
                        (window[j] < window[j+1]) != (window[j+1] < window[j+2])
                        for j in range(12)
                    )
                    if alternating:
                        for j in range(i, i + 14):
                            if RuleViolation.RULE4_ALTERNATING.value not in violations[j]:
                                violations[j].append(RuleViolation.RULE4_ALTERNATING.value)
                except (IndexError, ZeroDivisionError):
                    pass

        # Rule 5: 3점 중 2점이 2시그마 초과
        if n >= 3:
            for i in range(n - 2):
                window = values[i:i+3]
                above = sum(1 for v in window if v > two_sigma_upper)
                below = sum(1 for v in window if v < two_sigma_lower)

                if above >= 2 or below >= 2:
                    for j in range(i, i + 3):
                        if RuleViolation.RULE5_2OF3_BEYOND_2SIGMA.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE5_2OF3_BEYOND_2SIGMA.value)

        # Rule 6: 5점 중 4점이 1시그마 초과
        if n >= 5:
            for i in range(n - 4):
                window = values[i:i+5]
                above = sum(1 for v in window if v > one_sigma_upper)
                below = sum(1 for v in window if v < one_sigma_lower)

                if above >= 4 or below >= 4:
                    for j in range(i, i + 5):
                        if RuleViolation.RULE6_4OF5_BEYOND_1SIGMA.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE6_4OF5_BEYOND_1SIGMA.value)

        # Rule 7: 15점이 1시그마 이내 (층화)
        if n >= 15:
            for i in range(n - 14):
                window = values[i:i+15]
                within = all(one_sigma_lower <= v <= one_sigma_upper for v in window)

                if within:
                    for j in range(i, i + 15):
                        if RuleViolation.RULE7_STRATIFICATION.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE7_STRATIFICATION.value)

        # Rule 8: 8점이 1시그마 밖 (혼합)
        if n >= 8:
            for i in range(n - 7):
                window = values[i:i+8]
                outside = all(v > one_sigma_upper or v < one_sigma_lower for v in window)

                if outside:
                    for j in range(i, i + 8):
                        if RuleViolation.RULE8_MIXTURE.value not in violations[j]:
                            violations[j].append(RuleViolation.RULE8_MIXTURE.value)

        return violations


class XBarChart(ControlChart):
    """X-bar 관리도 (평균 관리도)"""

    def __init__(self, subgroup_size: int = 5, sigma_level: float = None):
        super().__init__(ChartType.XBAR, sigma_level)
        self.subgroup_size = subgroup_size

    def calculate_limits(
        self,
        subgroup_means: np.ndarray,
        subgroup_ranges: np.ndarray = None,
        subgroup_stds: np.ndarray = None,
    ) -> ControlLimits:
        """X-bar 관리 한계 계산"""
        x_bar_bar = np.mean(subgroup_means)

        if self.subgroup_size not in CONTROL_CHART_CONSTANTS:
            raise ValueError(f"Subgroup size {self.subgroup_size} not supported (2-10)")

        constants = CONTROL_CHART_CONSTANTS[self.subgroup_size]

        if subgroup_ranges is not None:
            # R-bar 방법
            r_bar = np.mean(subgroup_ranges)
            A2 = constants["A2"]
            ucl = x_bar_bar + A2 * r_bar
            lcl = x_bar_bar - A2 * r_bar
            sigma = r_bar / constants["d2"] / np.sqrt(self.subgroup_size)
        elif subgroup_stds is not None:
            # S-bar 방법
            s_bar = np.mean(subgroup_stds)
            A3 = constants["A3"]
            ucl = x_bar_bar + A3 * s_bar
            lcl = x_bar_bar - A3 * s_bar
            sigma = s_bar / constants["c4"] / np.sqrt(self.subgroup_size)
        else:
            raise ValueError("Either subgroup_ranges or subgroup_stds required")

        return ControlLimits(ucl=ucl, cl=x_bar_bar, lcl=lcl, sigma=sigma)


class RChart(ControlChart):
    """R 관리도 (범위 관리도)"""

    def __init__(self, subgroup_size: int = 5, sigma_level: float = None):
        super().__init__(ChartType.R, sigma_level)
        self.subgroup_size = subgroup_size

    def calculate_limits(self, subgroup_ranges: np.ndarray) -> ControlLimits:
        """R 관리 한계 계산"""
        r_bar = np.mean(subgroup_ranges)

        if self.subgroup_size not in CONTROL_CHART_CONSTANTS:
            raise ValueError(f"Subgroup size {self.subgroup_size} not supported")

        constants = CONTROL_CHART_CONSTANTS[self.subgroup_size]
        D3, D4 = constants["D3"], constants["D4"]

        ucl = D4 * r_bar
        lcl = D3 * r_bar

        sigma = r_bar / constants["d2"]

        return ControlLimits(ucl=ucl, cl=r_bar, lcl=lcl, sigma=sigma)


class IndividualChart(ControlChart):
    """개별값 관리도 (I-MR)"""

    def __init__(self, sigma_level: float = None):
        super().__init__(ChartType.INDIVIDUAL, sigma_level)

    def calculate_limits(self, values: np.ndarray) -> ControlLimits:
        """개별값 관리 한계 계산"""
        x_bar = np.mean(values)

        # Moving Range 계산
        mr = np.abs(np.diff(values))
        mr_bar = np.mean(mr)

        # d2 for n=2
        d2 = 1.128
        sigma = mr_bar / d2

        ucl = x_bar + 3 * sigma
        lcl = x_bar - 3 * sigma

        return ControlLimits(ucl=ucl, cl=x_bar, lcl=lcl, sigma=sigma)


class SPCAnalyzer:
    """SPC 통합 분석기"""

    def __init__(self):
        self.charts: Dict[str, ControlChart] = {}

    def analyze_xbar_r(
        self,
        data: pd.DataFrame,
        value_column: str = "value",
        subgroup_column: str = "subgroup_id",
        timestamp_column: str = "timestamp",
    ) -> Tuple[ControlChartResult, ControlChartResult]:
        """X-bar R 분석"""
        # 서브그룹별 집계
        subgroups = data.groupby(subgroup_column).agg({
            value_column: ["mean", "std", lambda x: x.max() - x.min(), "count"],
            timestamp_column: "first",
        })
        subgroups.columns = ["mean", "std", "range", "count", "timestamp"]
        subgroups = subgroups.reset_index()

        subgroup_size = int(subgroups["count"].mode().iloc[0])

        # 관리도 생성
        xbar_chart = XBarChart(subgroup_size=subgroup_size)
        r_chart = RChart(subgroup_size=subgroup_size)

        # 한계선 계산
        xbar_limits = xbar_chart.calculate_limits(
            subgroup_means=subgroups["mean"].values,
            subgroup_ranges=subgroups["range"].values,
        )
        r_limits = r_chart.calculate_limits(subgroups["range"].values)

        # 규칙 검사
        xbar_violations = xbar_chart.check_rules(subgroups["mean"].tolist(), xbar_limits)
        r_violations = r_chart.check_rules(subgroups["range"].tolist(), r_limits)

        # 결과 생성
        xbar_points = []
        for i, row in subgroups.iterrows():
            status = "OOC" if xbar_violations[i] else "NORMAL"
            xbar_points.append(ChartPoint(
                index=i,
                value=row["mean"],
                timestamp=row["timestamp"],
                subgroup_id=row[subgroup_column],
                violations=xbar_violations[i],
                status=status,
            ))

        r_points = []
        for i, row in subgroups.iterrows():
            status = "OOC" if r_violations[i] else "NORMAL"
            r_points.append(ChartPoint(
                index=i,
                value=row["range"],
                timestamp=row["timestamp"],
                subgroup_id=row[subgroup_column],
                violations=r_violations[i],
                status=status,
            ))

        # 통계
        xbar_stats = {
            "mean": float(subgroups["mean"].mean()),
            "std": float(subgroups["mean"].std()),
            "min": float(subgroups["mean"].min()),
            "max": float(subgroups["mean"].max()),
            "ooc_count": sum(1 for v in xbar_violations if v),
        }

        r_stats = {
            "mean": float(subgroups["range"].mean()),
            "std": float(subgroups["range"].std()),
            "min": float(subgroups["range"].min()),
            "max": float(subgroups["range"].max()),
            "ooc_count": sum(1 for v in r_violations if v),
        }

        # 위반 요약
        def summarize_violations(violations_list):
            summary = {}
            for violations in violations_list:
                for v in violations:
                    summary[v] = summary.get(v, 0) + 1
            return summary

        xbar_result = ControlChartResult(
            chart_type=ChartType.XBAR,
            limits=xbar_limits,
            points=xbar_points,
            statistics=xbar_stats,
            violations_summary=summarize_violations(xbar_violations),
        )

        r_result = ControlChartResult(
            chart_type=ChartType.R,
            limits=r_limits,
            points=r_points,
            statistics=r_stats,
            violations_summary=summarize_violations(r_violations),
        )

        return xbar_result, r_result

    def analyze_individual(
        self,
        values: np.ndarray,
        timestamps: List[datetime] = None,
    ) -> ControlChartResult:
        """개별값 분석"""
        chart = IndividualChart()
        limits = chart.calculate_limits(values)
        violations = chart.check_rules(values.tolist(), limits)

        points = []
        for i, v in enumerate(values):
            status = "OOC" if violations[i] else "NORMAL"
            points.append(ChartPoint(
                index=i,
                value=float(v),
                timestamp=timestamps[i] if timestamps else None,
                violations=violations[i],
                status=status,
            ))

        stats = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ooc_count": sum(1 for v in violations if v),
        }

        def summarize_violations(violations_list):
            summary = {}
            for viols in violations_list:
                for v in viols:
                    summary[v] = summary.get(v, 0) + 1
            return summary

        return ControlChartResult(
            chart_type=ChartType.INDIVIDUAL,
            limits=limits,
            points=points,
            statistics=stats,
            violations_summary=summarize_violations(violations),
        )
