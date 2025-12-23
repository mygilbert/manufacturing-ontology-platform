"""
공정 능력 분석 모듈

Cp, Cpk, Pp, Ppk 및 관련 지수를 계산합니다.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.append('/app/src')

from config import config

logger = logging.getLogger(__name__)


class CapabilityLevel(Enum):
    """공정 능력 등급"""
    EXCELLENT = "EXCELLENT"      # Cpk >= 2.0
    GOOD = "GOOD"                # 1.67 <= Cpk < 2.0
    ACCEPTABLE = "ACCEPTABLE"    # 1.33 <= Cpk < 1.67
    MARGINAL = "MARGINAL"        # 1.0 <= Cpk < 1.33
    POOR = "POOR"                # Cpk < 1.0


@dataclass
class CapabilityIndices:
    """공정 능력 지수"""
    # 잠재 능력 (Potential Capability)
    cp: float                    # Process Capability
    cpu: float                   # Upper Capability
    cpl: float                   # Lower Capability
    cpk: float                   # Process Capability Index

    # 성과 지수 (Performance Index) - 장기
    pp: Optional[float] = None   # Process Performance
    ppu: Optional[float] = None  # Upper Performance
    ppl: Optional[float] = None  # Lower Performance
    ppk: Optional[float] = None  # Process Performance Index

    # 추가 지수
    cpm: Optional[float] = None  # Taguchi Capability Index
    cr: Optional[float] = None   # Capability Ratio (1/Cp)

    # 예상 불량률
    ppm_upper: float = 0.0       # PPM above USL
    ppm_lower: float = 0.0       # PPM below LSL
    ppm_total: float = 0.0       # Total PPM

    # 메타데이터
    level: CapabilityLevel = CapabilityLevel.POOR
    sample_size: int = 0
    mean: float = 0.0
    std_within: float = 0.0      # 군내 표준편차
    std_overall: float = 0.0     # 전체 표준편차


@dataclass
class SpecificationLimits:
    """규격 한계"""
    usl: float                   # Upper Spec Limit
    lsl: float                   # Lower Spec Limit
    target: Optional[float] = None  # Target Value

    @property
    def tolerance(self) -> float:
        """규격 폭"""
        return self.usl - self.lsl

    @property
    def center(self) -> float:
        """규격 중심"""
        return self.target if self.target else (self.usl + self.lsl) / 2


class CapabilityAnalyzer:
    """공정 능력 분석기"""

    def __init__(self):
        self.warning_threshold = config.spc.cpk_warning_threshold
        self.critical_threshold = config.spc.cpk_critical_threshold

    def calculate_indices(
        self,
        data: np.ndarray,
        specs: SpecificationLimits,
        subgroup_size: int = None,
    ) -> CapabilityIndices:
        """
        공정 능력 지수 계산

        Args:
            data: 측정 데이터
            specs: 규격 한계
            subgroup_size: 서브그룹 크기 (None이면 개별값)

        Returns:
            공정 능력 지수
        """
        n = len(data)
        if n < 2:
            raise ValueError("At least 2 data points required")

        mean = np.mean(data)

        # 표준편차 계산
        if subgroup_size and subgroup_size > 1:
            # 군내 표준편차 (Within-subgroup)
            std_within = self._calculate_within_std(data, subgroup_size)
        else:
            # Moving Range 방법 (개별값)
            mr = np.abs(np.diff(data))
            d2 = 1.128  # n=2
            std_within = np.mean(mr) / d2

        std_overall = np.std(data, ddof=1)

        # Cp, Cpu, Cpl, Cpk (단기 능력 - 군내 변동)
        cp = specs.tolerance / (6 * std_within) if std_within > 0 else 0
        cpu = (specs.usl - mean) / (3 * std_within) if std_within > 0 else 0
        cpl = (mean - specs.lsl) / (3 * std_within) if std_within > 0 else 0
        cpk = min(cpu, cpl)

        # Pp, Ppu, Ppl, Ppk (장기 성과 - 전체 변동)
        pp = specs.tolerance / (6 * std_overall) if std_overall > 0 else 0
        ppu = (specs.usl - mean) / (3 * std_overall) if std_overall > 0 else 0
        ppl = (mean - specs.lsl) / (3 * std_overall) if std_overall > 0 else 0
        ppk = min(ppu, ppl)

        # Cpm (Taguchi - 목표값 고려)
        cpm = None
        if specs.target is not None:
            variance_from_target = np.mean((data - specs.target) ** 2)
            if variance_from_target > 0:
                cpm = specs.tolerance / (6 * np.sqrt(variance_from_target))

        # Cr (Capability Ratio)
        cr = 1 / cp if cp > 0 else float('inf')

        # PPM 계산
        ppm_upper = self._calculate_ppm(mean, std_overall, specs.usl, upper=True)
        ppm_lower = self._calculate_ppm(mean, std_overall, specs.lsl, upper=False)
        ppm_total = ppm_upper + ppm_lower

        # 등급 판정
        level = self._determine_level(cpk)

        return CapabilityIndices(
            cp=cp,
            cpu=cpu,
            cpl=cpl,
            cpk=cpk,
            pp=pp,
            ppu=ppu,
            ppl=ppl,
            ppk=ppk,
            cpm=cpm,
            cr=cr,
            ppm_upper=ppm_upper,
            ppm_lower=ppm_lower,
            ppm_total=ppm_total,
            level=level,
            sample_size=n,
            mean=mean,
            std_within=std_within,
            std_overall=std_overall,
        )

    def _calculate_within_std(self, data: np.ndarray, subgroup_size: int) -> float:
        """군내 표준편차 계산"""
        n = len(data)
        n_subgroups = n // subgroup_size

        if n_subgroups < 2:
            return np.std(data, ddof=1)

        # 서브그룹별 범위
        ranges = []
        for i in range(n_subgroups):
            subgroup = data[i*subgroup_size:(i+1)*subgroup_size]
            ranges.append(np.max(subgroup) - np.min(subgroup))

        r_bar = np.mean(ranges)

        # d2 상수 조회
        if subgroup_size in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                        6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
            d2 = d2_values[subgroup_size]
        else:
            d2 = 2.326  # 기본값 (n=5)

        return r_bar / d2

    def _calculate_ppm(
        self,
        mean: float,
        std: float,
        limit: float,
        upper: bool = True,
    ) -> float:
        """PPM (Parts Per Million) 계산"""
        if std <= 0:
            return 0.0

        z = (limit - mean) / std

        if upper:
            probability = 1 - stats.norm.cdf(z)
        else:
            probability = stats.norm.cdf(z)

        return probability * 1_000_000

    def _determine_level(self, cpk: float) -> CapabilityLevel:
        """Cpk 기반 등급 판정"""
        if cpk >= 2.0:
            return CapabilityLevel.EXCELLENT
        elif cpk >= 1.67:
            return CapabilityLevel.GOOD
        elif cpk >= 1.33:
            return CapabilityLevel.ACCEPTABLE
        elif cpk >= 1.0:
            return CapabilityLevel.MARGINAL
        else:
            return CapabilityLevel.POOR

    def analyze_trend(
        self,
        data: List[Tuple[datetime, np.ndarray]],
        specs: SpecificationLimits,
    ) -> List[Dict[str, Any]]:
        """
        시간별 공정 능력 추세 분석

        Args:
            data: (timestamp, values) 리스트
            specs: 규격 한계

        Returns:
            시간별 능력 지수
        """
        results = []

        for timestamp, values in data:
            if len(values) >= 2:
                indices = self.calculate_indices(values, specs)
                results.append({
                    "timestamp": timestamp.isoformat(),
                    "cpk": indices.cpk,
                    "ppk": indices.ppk,
                    "mean": indices.mean,
                    "std": indices.std_overall,
                    "level": indices.level.value,
                    "ppm_total": indices.ppm_total,
                    "sample_size": indices.sample_size,
                })

        return results

    def compare_equipment(
        self,
        equipment_data: Dict[str, np.ndarray],
        specs: SpecificationLimits,
    ) -> pd.DataFrame:
        """
        설비간 공정 능력 비교

        Args:
            equipment_data: {equipment_id: values} 딕셔너리
            specs: 규격 한계

        Returns:
            설비별 비교 DataFrame
        """
        results = []

        for equipment_id, values in equipment_data.items():
            if len(values) >= 2:
                indices = self.calculate_indices(values, specs)
                results.append({
                    "equipment_id": equipment_id,
                    "sample_size": indices.sample_size,
                    "mean": indices.mean,
                    "std": indices.std_overall,
                    "cp": indices.cp,
                    "cpk": indices.cpk,
                    "ppk": indices.ppk,
                    "ppm_total": indices.ppm_total,
                    "level": indices.level.value,
                })

        df = pd.DataFrame(results)

        # Cpk 기준 정렬
        if not df.empty:
            df = df.sort_values("cpk", ascending=False)

        return df

    def get_recommendations(self, indices: CapabilityIndices) -> List[str]:
        """
        공정 능력 기반 개선 권고

        Args:
            indices: 공정 능력 지수

        Returns:
            권고 사항 목록
        """
        recommendations = []

        # Cpk vs Ppk 비교 (안정성)
        if indices.ppk and indices.cpk > 0:
            ratio = indices.ppk / indices.cpk
            if ratio < 0.8:
                recommendations.append(
                    f"공정이 불안정합니다. Ppk/Cpk = {ratio:.2f}. "
                    "특별 원인을 조사하여 변동을 줄이세요."
                )

        # 중심 이탈 (Cp vs Cpk)
        if indices.cp > 0 and indices.cpk > 0:
            centering = indices.cpk / indices.cp
            if centering < 0.8:
                if indices.cpu < indices.cpl:
                    recommendations.append(
                        "공정 평균이 규격 상한에 가깝습니다. "
                        "평균을 낮추는 방향으로 조정하세요."
                    )
                else:
                    recommendations.append(
                        "공정 평균이 규격 하한에 가깝습니다. "
                        "평균을 높이는 방향으로 조정하세요."
                    )

        # 능력 등급별 권고
        if indices.level == CapabilityLevel.POOR:
            recommendations.append(
                f"Cpk = {indices.cpk:.2f}로 공정 능력이 부족합니다. "
                "즉각적인 개선 조치가 필요합니다. "
                "변동 원인 분석 및 공정 파라미터 최적화를 수행하세요."
            )
        elif indices.level == CapabilityLevel.MARGINAL:
            recommendations.append(
                f"Cpk = {indices.cpk:.2f}로 공정 능력이 한계 수준입니다. "
                "지속적인 모니터링과 개선 활동이 필요합니다."
            )
        elif indices.level == CapabilityLevel.ACCEPTABLE:
            recommendations.append(
                f"Cpk = {indices.cpk:.2f}로 공정 능력이 허용 수준입니다. "
                "현 수준 유지 및 추가 개선을 검토하세요."
            )
        elif indices.level in [CapabilityLevel.GOOD, CapabilityLevel.EXCELLENT]:
            recommendations.append(
                f"Cpk = {indices.cpk:.2f}로 공정 능력이 우수합니다. "
                "현 수준을 유지하세요."
            )

        # PPM 기반 권고
        if indices.ppm_total > 10000:
            recommendations.append(
                f"예상 불량률이 {indices.ppm_total:.0f} PPM으로 높습니다. "
                "불량 원인 분석이 필요합니다."
            )
        elif indices.ppm_total > 1000:
            recommendations.append(
                f"예상 불량률이 {indices.ppm_total:.0f} PPM입니다. "
                "개선 여지가 있습니다."
            )

        return recommendations


def calculate_quick_cpk(
    values: np.ndarray,
    usl: float,
    lsl: float,
) -> float:
    """
    빠른 Cpk 계산 (단순화)

    Args:
        values: 측정값
        usl: 규격 상한
        lsl: 규격 하한

    Returns:
        Cpk 값
    """
    if len(values) < 2:
        return 0.0

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    if std <= 0:
        return 0.0

    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)

    return min(cpu, cpl)
