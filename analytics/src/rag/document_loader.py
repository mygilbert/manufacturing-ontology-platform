# ============================================================
# Document Loader for RAG System
# ============================================================
"""
다양한 소스에서 문서를 로드하고 청킹하는 모듈
- JSON 지식 파일 로딩
- 텍스트 파일 로딩
- 청킹 및 메타데이터 추출
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .simple_store import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """문서 로더 클래스"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 청크 최대 문자 수
            chunk_overlap: 청크 간 겹치는 문자 수
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_expert_knowledge(self, path: str) -> List[Document]:
        """
        전문가 지식 JSON 파일 로드

        Args:
            path: expert_knowledge.json 경로

        Returns:
            Document 리스트
        """
        documents = []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            source_info = f"expert_knowledge_{Path(path).stem}"

            # 1. 인과관계 (Causal Relationships)
            for i, rel in enumerate(data.get("causal_relationships", [])):
                content = self._format_causal_relationship(rel)
                doc = Document(
                    id=f"{source_info}_causal_{i}",
                    content=content,
                    metadata={
                        "doc_type": "causal_relationship",
                        "source": "expert_knowledge",
                        "source_param": rel.get("source", ""),
                        "target_param": rel.get("target", ""),
                        "relation_type": rel.get("relation_type", ""),
                        "confidence": rel.get("confidence", 0),
                    }
                )
                documents.append(doc)

            # 2. 알람 원인 (Alarm Causes)
            for i, alarm in enumerate(data.get("alarm_causes", [])):
                content = self._format_alarm_cause(alarm)
                doc = Document(
                    id=f"{source_info}_alarm_{i}",
                    content=content,
                    metadata={
                        "doc_type": "alarm_cause",
                        "source": "expert_knowledge",
                        "alarm_code": alarm.get("alarm_code", ""),
                        "cause_parameter": alarm.get("cause_parameter", ""),
                        "priority": alarm.get("priority", 0),
                        "probability": alarm.get("probability", 0),
                    }
                )
                documents.append(doc)

            # 3. 선행 지표 (Leading Indicators)
            for i, indicator in enumerate(data.get("leading_indicators", [])):
                content = self._format_leading_indicator(indicator)
                doc = Document(
                    id=f"{source_info}_indicator_{i}",
                    content=content,
                    metadata={
                        "doc_type": "leading_indicator",
                        "source": "expert_knowledge",
                        "target_event": indicator.get("target_event", ""),
                        "indicator": indicator.get("indicator", ""),
                        "lead_time": indicator.get("lead_time", 0),
                    }
                )
                documents.append(doc)

            # 4. 불가능한 관계 (Impossible Relationships)
            for i, impossible in enumerate(data.get("impossible_relationships", [])):
                content = self._format_impossible_relationship(impossible)
                doc = Document(
                    id=f"{source_info}_impossible_{i}",
                    content=content,
                    metadata={
                        "doc_type": "impossible_relationship",
                        "source": "expert_knowledge",
                        "param1": impossible.get("param1", ""),
                        "param2": impossible.get("param2", ""),
                    }
                )
                documents.append(doc)

            logger.info(f"Loaded {len(documents)} documents from expert_knowledge.json")
            return documents

        except Exception as e:
            logger.error(f"Failed to load expert_knowledge: {e}")
            return []

    def load_discovered_relationships(self, path: str) -> List[Document]:
        """
        발견된 관계 JSON 파일 로드

        Args:
            path: discovered_relationships.json 경로

        Returns:
            Document 리스트
        """
        documents = []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            source_info = f"discovered_{Path(path).stem}"

            # 상위 관계들 로드
            for i, rel in enumerate(data.get("top_relationships", [])):
                content = self._format_discovered_relationship(rel)
                doc = Document(
                    id=f"{source_info}_rel_{i}",
                    content=content,
                    metadata={
                        "doc_type": "discovered_relationship",
                        "source": "relationship_discovery",
                        "source_param": rel.get("source", ""),
                        "target_param": rel.get("target", ""),
                        "relation_type": rel.get("relation_type", ""),
                        "method": rel.get("method", ""),
                        "confidence": rel.get("confidence", 0),
                    }
                )
                documents.append(doc)

            # 요약 정보도 문서화
            if "correlation_summary" in data or "causality_summary" in data:
                summary_content = self._format_discovery_summary(data)
                doc = Document(
                    id=f"{source_info}_summary",
                    content=summary_content,
                    metadata={
                        "doc_type": "discovery_summary",
                        "source": "relationship_discovery",
                        "timestamp": data.get("timestamp", ""),
                    }
                )
                documents.append(doc)

            logger.info(f"Loaded {len(documents)} documents from discovered_relationships.json")
            return documents

        except Exception as e:
            logger.error(f"Failed to load discovered_relationships: {e}")
            return []

    def load_domain_knowledge_text(self, text: str, source_name: str = "domain_knowledge") -> List[Document]:
        """
        도메인 지식 텍스트를 청킹하여 로드

        Args:
            text: 도메인 지식 텍스트 (DOMAIN_KNOWLEDGE 상수 등)
            source_name: 소스 이름

        Returns:
            Document 리스트
        """
        documents = []

        # 섹션별로 분리 (## 헤더 기준)
        sections = self._split_by_sections(text)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # 섹션 제목 추출
            title = self._extract_section_title(section)

            doc = Document(
                id=f"{source_name}_section_{i}",
                content=section.strip(),
                metadata={
                    "doc_type": "domain_knowledge",
                    "source": source_name,
                    "section_title": title,
                    "section_index": i,
                }
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} sections from domain knowledge text")
        return documents

    def load_text_file(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        텍스트 파일을 청킹하여 로드

        Args:
            path: 텍스트 파일 경로
            metadata: 추가 메타데이터

        Returns:
            Document 리스트
        """
        documents = []

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            base_metadata = {
                "doc_type": "text_file",
                "source": Path(path).name,
                "file_path": str(path),
            }
            if metadata:
                base_metadata.update(metadata)

            # 청킹
            chunks = self.chunk_text(content)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    id=f"text_{Path(path).stem}_{i}",
                    content=chunk,
                    metadata={**base_metadata, "chunk_index": i}
                )
                documents.append(doc)

            logger.info(f"Loaded {len(documents)} chunks from {path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load text file {path}: {e}")
            return []

    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 청크로 분할

        Args:
            text: 분할할 텍스트

        Returns:
            청크 리스트
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                chunks.append(text[start:])
                break

            # 문장 경계에서 자르기 시도
            break_point = self._find_break_point(text, start, end)
            chunks.append(text[start:break_point])

            # 다음 청크 시작점 (overlap 적용)
            start = break_point - self.chunk_overlap

        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """문장 경계 찾기"""
        # 마침표, 물음표, 느낌표, 줄바꿈 순으로 찾기
        for delimiter in ["\n\n", "\n", ".", "!", "?", " "]:
            last_pos = text.rfind(delimiter, start + self.chunk_size // 2, end)
            if last_pos > start:
                return last_pos + len(delimiter)
        return end

    def _split_by_sections(self, text: str) -> List[str]:
        """마크다운 섹션별로 분리"""
        # ## 헤더로 분리
        sections = re.split(r'\n(?=##\s)', text)
        return [s for s in sections if s.strip()]

    def _extract_section_title(self, section: str) -> str:
        """섹션 제목 추출"""
        match = re.match(r'^##\s*(.+?)(?:\n|$)', section)
        if match:
            return match.group(1).strip()
        return "Untitled"

    # ============================================================
    # 포맷팅 헬퍼 메서드
    # ============================================================

    def _format_causal_relationship(self, rel: Dict) -> str:
        """인과관계를 자연어 문서로 변환"""
        direction = "증가" if rel.get("direction") == "positive" else "감소"
        lag = rel.get("lag_range", [0, 0])

        return f"""인과관계: {rel.get('source')} → {rel.get('target')}
관계 유형: {rel.get('relation_type', 'CAUSES')}
방향: {direction}
시간 지연: {lag[0]}~{lag[1]}초
신뢰도: {rel.get('confidence', 0):.0%}
물리적 설명: {rel.get('physics', 'N/A')}

{rel.get('source')}가 변화하면 {rel.get('target')}가 {direction}합니다."""

    def _format_alarm_cause(self, alarm: Dict) -> str:
        """알람 원인을 자연어 문서로 변환"""
        return f"""알람 원인 분석: {alarm.get('alarm_code')}
원인 파라미터: {alarm.get('cause_parameter')}
조건: {alarm.get('condition')}
발생 확률: {alarm.get('probability', 0):.0%}
우선순위: {alarm.get('priority')}
조치 방법: {alarm.get('action')}

{alarm.get('alarm_code')} 알람이 발생하면 {alarm.get('cause_parameter')}를 확인하세요.
권장 조치: {alarm.get('action')}"""

    def _format_leading_indicator(self, indicator: Dict) -> str:
        """선행 지표를 자연어 문서로 변환"""
        lead_time_min = indicator.get("lead_time", 0) / 60

        return f"""선행 지표: {indicator.get('indicator')} → {indicator.get('target_event')}
패턴: {indicator.get('pattern')}
선행 시간: {lead_time_min:.1f}분 ({indicator.get('lead_time')}초)

{indicator.get('indicator')}에서 {indicator.get('pattern')} 패턴이 감지되면
약 {lead_time_min:.1f}분 후 {indicator.get('target_event')}가 발생할 수 있습니다."""

    def _format_impossible_relationship(self, impossible: Dict) -> str:
        """불가능한 관계를 자연어 문서로 변환"""
        return f"""불가능한 관계: {impossible.get('param1')} ↔ {impossible.get('param2')}
이유: {impossible.get('reason')}

{impossible.get('param1')}과 {impossible.get('param2')} 사이에는
인과관계가 성립할 수 없습니다. 이유: {impossible.get('reason')}"""

    def _format_discovered_relationship(self, rel: Dict) -> str:
        """발견된 관계를 자연어 문서로 변환"""
        return f"""발견된 관계: {rel.get('source')} → {rel.get('target')}
관계 유형: {rel.get('relation_type')}
발견 방법: {rel.get('method')}
신뢰도: {rel.get('confidence', 0):.2f}

데이터 분석을 통해 {rel.get('source')}와 {rel.get('target')} 사이에
{rel.get('relation_type')} 관계가 발견되었습니다. (방법: {rel.get('method')})"""

    def _format_discovery_summary(self, data: Dict) -> str:
        """발견 요약을 자연어 문서로 변환"""
        corr = data.get("correlation_summary", {})
        caus = data.get("causality_summary", {})
        pattern = data.get("pattern_summary", {})

        return f"""관계 발견 요약
분석 시간: {data.get('timestamp', 'N/A')}
총 발견 관계: {data.get('total_relationships', 0)}개
고신뢰도 관계: {data.get('high_confidence_count', 0)}개

상관 분석:
- 분석 수: {corr.get('total_analyzed', 0)}개
- 유의미한 상관: {corr.get('significant_found', 0)}개
- 최대 상관계수: {corr.get('max_correlation', 0):.4f}

인과 분석:
- 테스트 수: {caus.get('total_tested', 0)}개
- 인과관계 발견: {caus.get('causal_found', 0)}개
- 양방향 관계: {caus.get('bidirectional_pairs', 0)}쌍
- 평균 시간 지연: {caus.get('avg_lag', 0):.1f}초

패턴 분석:
- 총 패턴: {pattern.get('total_patterns', 0)}개
- 순차 패턴: {pattern.get('sequential_patterns', 0)}개
- 동시 발생: {pattern.get('co_occurrence_patterns', 0)}개"""
