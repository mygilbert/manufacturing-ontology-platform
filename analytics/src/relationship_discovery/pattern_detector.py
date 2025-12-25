"""
Pattern Detector
================

이벤트 시퀀스에서 반복 패턴을 발견

지원 기능:
- Sequential Pattern Mining: 이벤트 발생 순서 패턴
- Association Rules: 동시 발생 패턴
- Temporal Pattern: 시간 간격 기반 패턴
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from itertools import combinations
import warnings

from .config import PatternConfig, RelationType


@dataclass
class PatternResult:
    """패턴 발견 결과"""
    pattern: Tuple[str, ...]              # 이벤트 시퀀스
    support: float                         # 지지도 (발생 빈도)
    confidence: float                      # 신뢰도
    lift: float                            # 리프트 (기대 대비 실제)
    count: int                             # 발생 횟수
    avg_time_gap: float                    # 평균 시간 간격 (초)
    pattern_type: str = "sequential"       # sequential, co_occurrence
    relation_type: str = RelationType.PRECEDES.value
    antecedent: Optional[Tuple[str, ...]] = None  # 선행 이벤트
    consequent: Optional[Tuple[str, ...]] = None  # 후행 이벤트

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern': ' → '.join(self.pattern),
            'pattern_length': len(self.pattern),
            'support': round(self.support, 4),
            'confidence': round(self.confidence, 4),
            'lift': round(self.lift, 4),
            'count': self.count,
            'avg_time_gap_seconds': round(self.avg_time_gap, 2),
            'pattern_type': self.pattern_type,
            'relation_type': self.relation_type,
        }

    def __str__(self) -> str:
        return f"{' → '.join(self.pattern)} (sup={self.support:.3f}, conf={self.confidence:.3f})"


@dataclass
class EventSequence:
    """이벤트 시퀀스"""
    events: List[str]
    timestamps: List[float]
    sequence_id: str = ""

    def get_subsequences(self, max_length: int = 5) -> List[Tuple[str, ...]]:
        """모든 부분 시퀀스 생성"""
        subsequences = []
        n = len(self.events)

        for length in range(2, min(max_length + 1, n + 1)):
            for i in range(n - length + 1):
                subseq = tuple(self.events[i:i + length])
                subsequences.append(subseq)

        return subsequences


class PatternDetector:
    """이벤트 패턴 발견기"""

    def __init__(self, config: Optional[PatternConfig] = None):
        self.config = config or PatternConfig()
        self.results: List[PatternResult] = []
        self.event_counts: Counter = Counter()
        self.total_sequences: int = 0

    def detect(
        self,
        events: pd.DataFrame,
        event_col: str = 'event_type',
        timestamp_col: str = 'timestamp',
        sequence_col: Optional[str] = None,  # 시퀀스 구분 컬럼 (예: lot_id)
    ) -> List[PatternResult]:
        """
        이벤트 데이터에서 패턴 발견

        Args:
            events: 이벤트 데이터프레임
            event_col: 이벤트 유형 컬럼
            timestamp_col: 타임스탬프 컬럼
            sequence_col: 시퀀스 구분 컬럼 (None이면 시간 윈도우로 분할)

        Returns:
            발견된 패턴 목록
        """
        self.results = []

        # 이벤트 시퀀스 생성
        sequences = self._create_sequences(
            events, event_col, timestamp_col, sequence_col
        )

        if not sequences:
            warnings.warn("생성된 시퀀스가 없습니다.")
            return []

        self.total_sequences = len(sequences)
        print(f"분석할 시퀀스: {self.total_sequences}개")

        # 이벤트 빈도 계산
        self._count_events(sequences)

        # Sequential Pattern Mining
        sequential_patterns = self._mine_sequential_patterns(sequences)
        self.results.extend(sequential_patterns)

        # Association Rules
        association_rules = self._mine_association_rules(sequences)
        self.results.extend(association_rules)

        # 필터링
        filtered = [
            r for r in self.results
            if r.support >= self.config.min_support
            and r.confidence >= self.config.min_confidence
        ]

        print(f"발견된 유의미한 패턴: {len(filtered)}개")
        return filtered

    def _create_sequences(
        self,
        events: pd.DataFrame,
        event_col: str,
        timestamp_col: str,
        sequence_col: Optional[str]
    ) -> List[EventSequence]:
        """이벤트 시퀀스 생성"""
        sequences = []

        # 타임스탬프 정렬
        events = events.sort_values(timestamp_col)

        if sequence_col and sequence_col in events.columns:
            # 시퀀스 ID로 그룹핑
            for seq_id, group in events.groupby(sequence_col):
                if len(group) >= 2:
                    seq = EventSequence(
                        events=group[event_col].tolist(),
                        timestamps=self._to_seconds(group[timestamp_col]),
                        sequence_id=str(seq_id)
                    )
                    sequences.append(seq)
        else:
            # 시간 윈도우로 분할
            window_seconds = self.config.time_window_seconds
            timestamps = self._to_seconds(events[timestamp_col])

            current_seq_events = []
            current_seq_times = []
            window_start = timestamps[0] if len(timestamps) > 0 else 0

            for i, (event, ts) in enumerate(zip(events[event_col], timestamps)):
                if ts - window_start > window_seconds:
                    if len(current_seq_events) >= 2:
                        sequences.append(EventSequence(
                            events=current_seq_events,
                            timestamps=current_seq_times,
                            sequence_id=f"window_{len(sequences)}"
                        ))
                    current_seq_events = [event]
                    current_seq_times = [ts]
                    window_start = ts
                else:
                    current_seq_events.append(event)
                    current_seq_times.append(ts)

            # 마지막 시퀀스
            if len(current_seq_events) >= 2:
                sequences.append(EventSequence(
                    events=current_seq_events,
                    timestamps=current_seq_times,
                    sequence_id=f"window_{len(sequences)}"
                ))

        return sequences

    def _to_seconds(self, timestamps: pd.Series) -> List[float]:
        """타임스탬프를 초 단위로 변환"""
        if pd.api.types.is_datetime64_any_dtype(timestamps):
            base = timestamps.min()
            return [(t - base).total_seconds() for t in timestamps]
        return timestamps.tolist()

    def _count_events(self, sequences: List[EventSequence]) -> None:
        """이벤트 빈도 계산"""
        self.event_counts = Counter()
        for seq in sequences:
            self.event_counts.update(seq.events)

    def _mine_sequential_patterns(
        self,
        sequences: List[EventSequence]
    ) -> List[PatternResult]:
        """순차 패턴 마이닝 (PrefixSpan 간소화 버전)"""
        results = []

        # 패턴별 발생 정보 수집
        pattern_info: Dict[Tuple[str, ...], List[Dict]] = defaultdict(list)

        for seq in sequences:
            subsequences = seq.get_subsequences(self.config.max_pattern_length)

            for subseq in subsequences:
                # 시간 간격 계산
                start_idx = seq.events.index(subseq[0])
                time_gaps = []

                for i in range(len(subseq) - 1):
                    try:
                        idx1 = seq.events.index(subseq[i], start_idx)
                        idx2 = seq.events.index(subseq[i + 1], idx1 + 1)
                        gap = seq.timestamps[idx2] - seq.timestamps[idx1]
                        time_gaps.append(gap)
                        start_idx = idx2
                    except (ValueError, IndexError):
                        break

                if time_gaps:
                    pattern_info[subseq].append({
                        'sequence_id': seq.sequence_id,
                        'avg_gap': np.mean(time_gaps)
                    })

        # 패턴별 통계 계산
        for pattern, occurrences in pattern_info.items():
            count = len(occurrences)
            support = count / self.total_sequences

            if support < self.config.min_support:
                continue

            # 신뢰도 계산 (선행 패턴 대비)
            antecedent = pattern[:-1]
            if antecedent in pattern_info:
                confidence = count / len(pattern_info[antecedent])
            else:
                confidence = support

            # 리프트 계산
            expected = 1.0
            for event in pattern:
                expected *= self.event_counts[event] / sum(self.event_counts.values())
            lift = support / expected if expected > 0 else 1.0

            avg_time_gap = np.mean([o['avg_gap'] for o in occurrences])

            results.append(PatternResult(
                pattern=pattern,
                support=support,
                confidence=confidence,
                lift=lift,
                count=count,
                avg_time_gap=avg_time_gap,
                pattern_type="sequential",
                relation_type=RelationType.PRECEDES.value,
                antecedent=antecedent if len(antecedent) > 0 else None,
                consequent=(pattern[-1],)
            ))

        return results

    def _mine_association_rules(
        self,
        sequences: List[EventSequence]
    ) -> List[PatternResult]:
        """연관 규칙 마이닝 (동시 발생)"""
        results = []

        # 각 시퀀스를 이벤트 집합으로 변환
        transactions = [set(seq.events) for seq in sequences]

        # 아이템셋 빈도 계산
        itemset_counts: Counter = Counter()
        for transaction in transactions:
            # 2-itemset까지만 (계산 비용)
            for size in range(1, min(3, len(transaction) + 1)):
                for itemset in combinations(sorted(transaction), size):
                    itemset_counts[itemset] += 1

        # 연관 규칙 생성
        for itemset, count in itemset_counts.items():
            if len(itemset) < 2:
                continue

            support = count / self.total_sequences
            if support < self.config.min_support:
                continue

            # A → B 규칙
            for i in range(len(itemset)):
                antecedent = tuple(x for j, x in enumerate(itemset) if j != i)
                consequent = (itemset[i],)

                ant_count = itemset_counts.get(antecedent, 0)
                if ant_count == 0:
                    continue

                confidence = count / ant_count

                # 리프트
                cons_support = itemset_counts.get(consequent, 0) / self.total_sequences
                lift = confidence / cons_support if cons_support > 0 else 1.0

                if confidence >= self.config.min_confidence:
                    results.append(PatternResult(
                        pattern=antecedent + consequent,
                        support=support,
                        confidence=confidence,
                        lift=lift,
                        count=count,
                        avg_time_gap=0,  # 동시 발생이므로
                        pattern_type="co_occurrence",
                        relation_type=RelationType.CO_OCCURS.value,
                        antecedent=antecedent,
                        consequent=consequent
                    ))

        return results

    def detect_alarm_sequences(
        self,
        alarms: pd.DataFrame,
        equipment_col: str = 'equipment_id',
        alarm_col: str = 'alarm_code',
        timestamp_col: str = 'timestamp'
    ) -> List[PatternResult]:
        """알람 시퀀스 패턴 발견 (설비별)"""
        all_patterns = []

        for equipment_id, group in alarms.groupby(equipment_col):
            patterns = self.detect(
                group,
                event_col=alarm_col,
                timestamp_col=timestamp_col,
                sequence_col=None
            )

            for p in patterns:
                p.pattern = (equipment_id,) + p.pattern  # 설비 ID 추가

            all_patterns.extend(patterns)

        return all_patterns

    def find_root_cause_patterns(
        self,
        events: pd.DataFrame,
        target_event: str,
        event_col: str = 'event_type',
        timestamp_col: str = 'timestamp',
        lookback_seconds: int = 300
    ) -> List[PatternResult]:
        """특정 이벤트의 선행 패턴 찾기 (근본 원인 분석용)"""
        results = []

        events = events.sort_values(timestamp_col)
        timestamps = self._to_seconds(events[timestamp_col])

        # 타겟 이벤트 발생 시점 찾기
        target_indices = events[events[event_col] == target_event].index.tolist()

        preceding_sequences = []

        for idx in target_indices:
            pos = events.index.get_loc(idx)
            target_time = timestamps[pos]

            # lookback 윈도우 내 선행 이벤트
            preceding_events = []
            preceding_times = []

            for i in range(pos - 1, -1, -1):
                if target_time - timestamps[i] > lookback_seconds:
                    break
                preceding_events.insert(0, events.iloc[i][event_col])
                preceding_times.insert(0, timestamps[i])

            if preceding_events:
                preceding_sequences.append(EventSequence(
                    events=preceding_events + [target_event],
                    timestamps=preceding_times + [target_time]
                ))

        if not preceding_sequences:
            return []

        # 패턴 마이닝
        self.total_sequences = len(preceding_sequences)
        self._count_events(preceding_sequences)

        patterns = self._mine_sequential_patterns(preceding_sequences)

        # 타겟 이벤트로 끝나는 패턴만 필터링
        for p in patterns:
            if p.pattern[-1] == target_event:
                p.relation_type = RelationType.ROOT_CAUSE.value
                results.append(p)

        return sorted(results, key=lambda x: x.confidence, reverse=True)

    def get_results_dataframe(self) -> pd.DataFrame:
        """결과를 데이터프레임으로 반환"""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def summary(self) -> Dict[str, Any]:
        """분석 결과 요약"""
        if not self.results:
            return {"status": "no_results"}

        sequential = [r for r in self.results if r.pattern_type == "sequential"]
        co_occur = [r for r in self.results if r.pattern_type == "co_occurrence"]

        return {
            "total_patterns": len(self.results),
            "sequential_patterns": len(sequential),
            "co_occurrence_patterns": len(co_occur),
            "avg_support": np.mean([r.support for r in self.results]),
            "avg_confidence": np.mean([r.confidence for r in self.results]),
            "max_pattern_length": max(len(r.pattern) for r in self.results),
            "high_confidence": len([r for r in self.results if r.confidence > 0.8]),
        }
