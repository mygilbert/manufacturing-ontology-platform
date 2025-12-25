"""
Discovery Pipeline
==================

암묵적 관계 발견을 위한 통합 파이프라인

단계:
1. 데이터 로드 및 전처리
2. 상관관계 분석
3. 인과성 분석
4. 이벤트 패턴 발견
5. 결과 통합 및 랭킹
6. 온톨로지 저장
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
import json

from .config import DiscoveryConfig, RelationType
from .correlation_analyzer import CorrelationAnalyzer, CorrelationResult
from .causality_analyzer import CausalityAnalyzer, CausalityResult
from .pattern_detector import PatternDetector, PatternResult
from .relationship_store import RelationshipStore, DiscoveredRelationship


@dataclass
class DiscoveryReport:
    """발견 리포트"""
    timestamp: datetime
    data_info: Dict[str, Any]
    correlation_summary: Dict[str, Any]
    causality_summary: Dict[str, Any]
    pattern_summary: Dict[str, Any]
    total_relationships: int
    high_confidence_count: int
    top_relationships: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'data_info': self.data_info,
            'correlation_summary': self.correlation_summary,
            'causality_summary': self.causality_summary,
            'pattern_summary': self.pattern_summary,
            'total_relationships': self.total_relationships,
            'high_confidence_count': self.high_confidence_count,
            'top_relationships': self.top_relationships,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "암묵적 관계 발견 리포트",
            "=" * 60,
            f"분석 시각: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[ 데이터 정보 ]",
            f"  - 샘플 수: {self.data_info.get('n_samples', 'N/A'):,}",
            f"  - 파라미터 수: {self.data_info.get('n_parameters', 'N/A')}",
            f"  - 이벤트 수: {self.data_info.get('n_events', 'N/A'):,}",
            "",
            "[ 상관관계 분석 ]",
            f"  - 분석된 쌍: {self.correlation_summary.get('total_analyzed', 0)}",
            f"  - 유의미한 관계: {self.correlation_summary.get('significant_found', 0)}",
            f"  - 시간 지연 관계: {self.correlation_summary.get('with_time_lag', 0)}",
            "",
            "[ 인과성 분석 ]",
            f"  - 테스트된 방향: {self.causality_summary.get('total_tested', 0)}",
            f"  - 인과관계 발견: {self.causality_summary.get('causal_found', 0)}",
            f"  - 양방향 관계: {self.causality_summary.get('bidirectional_pairs', 0)}",
            "",
            "[ 이벤트 패턴 ]",
            f"  - 발견된 패턴: {self.pattern_summary.get('total_patterns', 0)}",
            f"  - 순차 패턴: {self.pattern_summary.get('sequential_patterns', 0)}",
            f"  - 동시 발생: {self.pattern_summary.get('co_occurrence_patterns', 0)}",
            "",
            "[ 종합 ]",
            f"  - 총 관계 수: {self.total_relationships}",
            f"  - 높은 신뢰도 (>0.7): {self.high_confidence_count}",
            "",
            "[ 상위 10개 관계 ]",
        ]

        for i, rel in enumerate(self.top_relationships[:10], 1):
            lines.append(
                f"  {i}. {rel['source']} → {rel['target']} "
                f"({rel['relation_type']}, conf={rel['confidence']:.3f})"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class DiscoveryPipeline:
    """통합 관계 발견 파이프라인"""

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()

        # 분석기 초기화
        self.correlation_analyzer = CorrelationAnalyzer(self.config.correlation)
        self.causality_analyzer = CausalityAnalyzer(self.config.causality)
        self.pattern_detector = PatternDetector(self.config.pattern)
        self.relationship_store = RelationshipStore(self.config.ontology)

        # 결과 저장
        self.correlation_results: List[CorrelationResult] = []
        self.causality_results: List[CausalityResult] = []
        self.pattern_results: List[PatternResult] = []
        self.all_relationships: List[DiscoveredRelationship] = []

        # 메타데이터
        self.data_info: Dict[str, Any] = {}
        self.last_run: Optional[datetime] = None

    def discover_all(
        self,
        pv_data: pd.DataFrame,
        event_data: Optional[pd.DataFrame] = None,
        pv_columns: Optional[List[str]] = None,
        timestamp_col: str = 'timestamp',
        event_col: str = 'event_type',
        sample_rate_hz: float = 1.0,
        run_correlation: bool = True,
        run_causality: bool = True,
        run_patterns: bool = True
    ) -> List[DiscoveredRelationship]:
        """
        모든 분석 실행

        Args:
            pv_data: PV 데이터프레임
            event_data: 이벤트 데이터프레임 (옵션)
            pv_columns: 분석할 PV 컬럼 목록
            timestamp_col: 타임스탬프 컬럼명
            event_col: 이벤트 유형 컬럼명
            sample_rate_hz: 샘플링 레이트
            run_correlation: 상관관계 분석 실행 여부
            run_causality: 인과성 분석 실행 여부
            run_patterns: 패턴 분석 실행 여부

        Returns:
            발견된 관계 목록
        """
        self.last_run = datetime.now()
        self.all_relationships = []

        # 데이터 정보 수집
        self._collect_data_info(pv_data, event_data, pv_columns, timestamp_col)

        if self.config.verbose:
            print("=" * 60)
            print("암묵적 관계 발견 파이프라인 시작")
            print("=" * 60)

        # 1. 상관관계 분석
        if run_correlation:
            if self.config.verbose:
                print("\n[1/3] 상관관계 분석...")
            self._run_correlation_analysis(
                pv_data, pv_columns, timestamp_col, sample_rate_hz
            )

        # 2. 인과성 분석
        if run_causality:
            if self.config.verbose:
                print("\n[2/3] 인과성 분석...")
            self._run_causality_analysis(
                pv_data, pv_columns, timestamp_col, sample_rate_hz
            )

        # 3. 패턴 분석
        if run_patterns and event_data is not None:
            if self.config.verbose:
                print("\n[3/3] 이벤트 패턴 분석...")
            self._run_pattern_analysis(event_data, event_col, timestamp_col)

        # 결과 통합
        self._integrate_results()

        if self.config.verbose:
            print(f"\n총 {len(self.all_relationships)}개 관계 발견")
            print("=" * 60)

        return self.all_relationships

    def _collect_data_info(
        self,
        pv_data: pd.DataFrame,
        event_data: Optional[pd.DataFrame],
        pv_columns: Optional[List[str]],
        timestamp_col: str
    ) -> None:
        """데이터 정보 수집"""
        if pv_columns is None:
            pv_columns = pv_data.select_dtypes(include=[np.number]).columns.tolist()
            if timestamp_col in pv_columns:
                pv_columns.remove(timestamp_col)

        self.data_info = {
            'n_samples': len(pv_data),
            'n_parameters': len(pv_columns),
            'parameters': pv_columns,
            'n_events': len(event_data) if event_data is not None else 0,
            'time_range': {
                'start': str(pv_data[timestamp_col].min()) if timestamp_col in pv_data else None,
                'end': str(pv_data[timestamp_col].max()) if timestamp_col in pv_data else None,
            }
        }

    def _run_correlation_analysis(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]],
        timestamp_col: str,
        sample_rate_hz: float
    ) -> None:
        """상관관계 분석 실행"""
        try:
            self.correlation_results = self.correlation_analyzer.analyze(
                data=data,
                columns=columns,
                timestamp_col=timestamp_col,
                sample_rate_hz=sample_rate_hz
            )

            # 저장소에 추가
            for result in self.correlation_results:
                rel = self.relationship_store.add_correlation(result)
                if rel:
                    self.all_relationships.append(rel)

        except Exception as e:
            warnings.warn(f"상관관계 분석 실패: {e}")
            self.correlation_results = []

    def _run_causality_analysis(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]],
        timestamp_col: str,
        sample_rate_hz: float
    ) -> None:
        """인과성 분석 실행"""
        try:
            self.causality_results = self.causality_analyzer.analyze(
                data=data,
                columns=columns,
                timestamp_col=timestamp_col,
                sample_rate_hz=sample_rate_hz
            )

            # 저장소에 추가
            for result in self.causality_results:
                rel = self.relationship_store.add_causality(result)
                if rel:
                    self.all_relationships.append(rel)

        except Exception as e:
            warnings.warn(f"인과성 분석 실패: {e}")
            self.causality_results = []

    def _run_pattern_analysis(
        self,
        event_data: pd.DataFrame,
        event_col: str,
        timestamp_col: str
    ) -> None:
        """패턴 분석 실행"""
        try:
            self.pattern_results = self.pattern_detector.detect(
                events=event_data,
                event_col=event_col,
                timestamp_col=timestamp_col
            )

            # 저장소에 추가
            for result in self.pattern_results:
                rel = self.relationship_store.add_pattern(result)
                if rel:
                    self.all_relationships.append(rel)

        except Exception as e:
            warnings.warn(f"패턴 분석 실패: {e}")
            self.pattern_results = []

    def _integrate_results(self) -> None:
        """결과 통합 및 중복 제거"""
        # 동일한 source-target 쌍에 대해 가장 높은 confidence 유지
        seen = {}

        for rel in self.all_relationships:
            key = (rel.source, rel.target, rel.relation_type)
            if key not in seen or seen[key].confidence < rel.confidence:
                seen[key] = rel

        self.all_relationships = list(seen.values())

        # confidence 기준 정렬
        self.all_relationships.sort(key=lambda x: x.confidence, reverse=True)

    def discover_for_parameter(
        self,
        data: pd.DataFrame,
        target_param: str,
        other_params: Optional[List[str]] = None,
        timestamp_col: str = 'timestamp',
        sample_rate_hz: float = 1.0
    ) -> List[DiscoveredRelationship]:
        """특정 파라미터에 영향을 주는 관계 발견"""
        if other_params is None:
            other_params = data.select_dtypes(include=[np.number]).columns.tolist()
            other_params = [c for c in other_params if c != target_param and c != timestamp_col]

        results = []

        # 상관관계
        for param in other_params:
            subset = data[[param, target_param]].dropna()
            if len(subset) >= self.config.correlation.min_samples:
                corr_results = self.correlation_analyzer.analyze(
                    subset,
                    columns=[param, target_param],
                    sample_rate_hz=sample_rate_hz
                )
                for r in corr_results:
                    if r.target_param == target_param:
                        rel = self.relationship_store.add_correlation(r)
                        if rel:
                            results.append(rel)

        # 인과성
        for param in other_params:
            subset = data[[param, target_param]].dropna()
            if len(subset) >= self.config.causality.min_samples:
                cause_results = self.causality_analyzer.analyze(
                    subset,
                    columns=[param, target_param],
                    sample_rate_hz=sample_rate_hz
                )
                for r in cause_results:
                    if r.target_param == target_param and r.is_causal:
                        rel = self.relationship_store.add_causality(r)
                        if rel:
                            results.append(rel)

        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def find_root_causes(
        self,
        event_data: pd.DataFrame,
        target_event: str,
        event_col: str = 'event_type',
        timestamp_col: str = 'timestamp',
        lookback_seconds: int = 300
    ) -> List[DiscoveredRelationship]:
        """특정 이벤트의 근본 원인 탐색"""
        patterns = self.pattern_detector.find_root_cause_patterns(
            events=event_data,
            target_event=target_event,
            event_col=event_col,
            timestamp_col=timestamp_col,
            lookback_seconds=lookback_seconds
        )

        results = []
        for pattern in patterns:
            rel = self.relationship_store.add_pattern(pattern)
            if rel:
                results.append(rel)

        return results

    def save_to_ontology(self, verified_only: bool = True) -> Dict[str, int]:
        """온톨로지에 저장"""
        return self.relationship_store.save_pending(save_all=not verified_only)

    def generate_report(self) -> DiscoveryReport:
        """분석 리포트 생성"""
        high_conf = [r for r in self.all_relationships if r.confidence > 0.7]

        top_rels = [
            {
                'source': r.source,
                'target': r.target,
                'relation_type': r.relation_type,
                'method': r.method,
                'confidence': r.confidence,
            }
            for r in self.all_relationships[:20]
        ]

        return DiscoveryReport(
            timestamp=self.last_run or datetime.now(),
            data_info=self.data_info,
            correlation_summary=self.correlation_analyzer.summary(),
            causality_summary=self.causality_analyzer.summary(),
            pattern_summary=self.pattern_detector.summary(),
            total_relationships=len(self.all_relationships),
            high_confidence_count=len(high_conf),
            top_relationships=top_rels
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """모든 결과를 데이터프레임으로"""
        if not self.all_relationships:
            return pd.DataFrame()

        data = []
        for rel in self.all_relationships:
            data.append({
                'source': rel.source,
                'target': rel.target,
                'relation_type': rel.relation_type,
                'method': rel.method,
                'confidence': rel.confidence,
                'verification_status': rel.verification_status,
                'discovered_at': rel.discovered_at,
                **rel.properties
            })

        return pd.DataFrame(data)

    def export_results(self, filepath: str, format: str = 'json') -> bool:
        """결과 내보내기"""
        try:
            if format == 'json':
                report = self.generate_report()
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)

            elif format == 'csv':
                df = self.get_results_dataframe()
                df.to_csv(filepath, index=False, encoding='utf-8-sig')

            elif format == 'html':
                self._export_html_report(filepath)

            else:
                warnings.warn(f"지원하지 않는 형식: {format}")
                return False

            return True

        except Exception as e:
            warnings.warn(f"내보내기 실패: {e}")
            return False

    def _export_html_report(self, filepath: str) -> None:
        """HTML 리포트 생성"""
        report = self.generate_report()

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>관계 발견 리포트</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .confidence-high {{ color: green; font-weight: bold; }}
        .confidence-medium {{ color: orange; }}
        .confidence-low {{ color: red; }}
    </style>
</head>
<body>
    <h1>암묵적 관계 발견 리포트</h1>
    <p>분석 시각: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="section">
        <h2>요약</h2>
        <ul>
            <li>분석 샘플: {report.data_info.get('n_samples', 'N/A'):,}개</li>
            <li>파라미터: {report.data_info.get('n_parameters', 'N/A')}개</li>
            <li>발견된 관계: {report.total_relationships}개</li>
            <li>높은 신뢰도: {report.high_confidence_count}개</li>
        </ul>
    </div>

    <div class="section">
        <h2>발견된 관계</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Source</th>
                <th>→</th>
                <th>Target</th>
                <th>유형</th>
                <th>방법</th>
                <th>신뢰도</th>
            </tr>
"""

        for i, rel in enumerate(report.top_relationships, 1):
            conf = rel['confidence']
            conf_class = 'high' if conf > 0.7 else ('medium' if conf > 0.5 else 'low')
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{rel['source']}</td>
                <td>→</td>
                <td>{rel['target']}</td>
                <td>{rel['relation_type']}</td>
                <td>{rel['method']}</td>
                <td class="confidence-{conf_class}">{conf:.3f}</td>
            </tr>
"""

        html += """
        </table>
    </div>
</body>
</html>
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

    def summary(self) -> Dict[str, Any]:
        """파이프라인 상태 요약"""
        return {
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'data_info': self.data_info,
            'total_relationships': len(self.all_relationships),
            'by_type': self._count_by_type(),
            'by_method': self._count_by_method(),
            'store_status': self.relationship_store.summary(),
        }

    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for rel in self.all_relationships:
            counts[rel.relation_type] = counts.get(rel.relation_type, 0) + 1
        return counts

    def _count_by_method(self) -> Dict[str, int]:
        counts = {}
        for rel in self.all_relationships:
            counts[rel.method] = counts.get(rel.method, 0) + 1
        return counts
