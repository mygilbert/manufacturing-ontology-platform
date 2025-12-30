# ============================================================
# RAG Retriever
# ============================================================
"""
쿼리 기반 관련 문서 검색 및 컨텍스트 포맷팅
- 시맨틱 검색
- 메타데이터 필터링
- 컨텍스트 포맷팅
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .simple_store import SimpleVectorStore, SearchResult

# 호환성 별칭
ChromaStore = SimpleVectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """검색 설정"""
    n_results: int = 5
    min_similarity: float = 0.3
    include_metadata: bool = True
    max_context_length: int = 2000


class RAGRetriever:
    """RAG 검색기"""

    def __init__(
        self,
        store: ChromaStore,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Args:
            store: ChromaDB 스토어
            config: 검색 설정
        """
        self.store = store
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        doc_types: Optional[List[str]] = None
    ) -> str:
        """
        쿼리에 관련된 컨텍스트 검색

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수 (기본: config.n_results)
            doc_types: 필터링할 문서 타입 리스트

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        if not query:
            return ""

        n = n_results or self.config.n_results

        # 메타데이터 필터 구성
        where = None
        if doc_types:
            if len(doc_types) == 1:
                where = {"doc_type": doc_types[0]}
            else:
                where = {"doc_type": {"$in": doc_types}}

        # 검색 실행
        results = self.store.search(query, n_results=n, where=where)

        # 유사도 필터링
        filtered_results = [
            r for r in results
            if r.similarity >= self.config.min_similarity
        ]

        if not filtered_results:
            logger.debug(f"No results above similarity threshold for query: {query[:50]}...")
            return ""

        # 컨텍스트 포맷팅
        context = self._format_context(filtered_results)

        # 길이 제한
        if len(context) > self.config.max_context_length:
            context = context[:self.config.max_context_length] + "\n... (truncated)"

        logger.info(f"Retrieved {len(filtered_results)} documents for query: {query[:50]}...")
        return context

    def retrieve_with_sources(
        self,
        query: str,
        n_results: Optional[int] = None,
        doc_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        소스 정보와 함께 컨텍스트 검색

        Returns:
            {
                "context": str,
                "sources": List[Dict],
                "n_results": int
            }
        """
        if not query:
            return {"context": "", "sources": [], "n_results": 0}

        n = n_results or self.config.n_results

        where = None
        if doc_types:
            if len(doc_types) == 1:
                where = {"doc_type": doc_types[0]}
            else:
                where = {"doc_type": {"$in": doc_types}}

        results = self.store.search(query, n_results=n, where=where)

        filtered_results = [
            r for r in results
            if r.similarity >= self.config.min_similarity
        ]

        context = self._format_context(filtered_results)
        sources = [
            {
                "id": r.id,
                "doc_type": r.metadata.get("doc_type", "unknown"),
                "source": r.metadata.get("source", "unknown"),
                "similarity": round(r.similarity, 3),
                "preview": r.content[:100] + "..." if len(r.content) > 100 else r.content
            }
            for r in filtered_results
        ]

        return {
            "context": context,
            "sources": sources,
            "n_results": len(filtered_results)
        }

    def retrieve_by_type(
        self,
        query: str,
        doc_type: str,
        n_results: int = 3
    ) -> List[SearchResult]:
        """특정 문서 타입으로 검색"""
        return self.store.search(
            query,
            n_results=n_results,
            where={"doc_type": doc_type}
        )

    def retrieve_for_alarm(
        self,
        alarm_code: str,
        equipment_id: Optional[str] = None
    ) -> str:
        """
        알람 코드 기반 관련 지식 검색

        Args:
            alarm_code: 알람 코드
            equipment_id: 설비 ID (옵션)

        Returns:
            알람 관련 컨텍스트
        """
        # 알람 원인 문서 우선 검색
        alarm_results = self.store.search(
            f"알람 {alarm_code} 원인 조치",
            n_results=3,
            where={"doc_type": "alarm_cause"}
        )

        # 관련 인과관계 검색
        causal_query = f"{alarm_code} 관련 파라미터 인과관계"
        causal_results = self.store.search(
            causal_query,
            n_results=3,
            where={"doc_type": "causal_relationship"}
        )

        all_results = alarm_results + causal_results

        if not all_results:
            return ""

        # 중복 제거
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique_results.append(r)

        return self._format_context(unique_results)

    def retrieve_for_parameter(
        self,
        parameter: str,
        context_type: str = "all"
    ) -> str:
        """
        파라미터 관련 지식 검색

        Args:
            parameter: 파라미터 이름
            context_type: "causes", "effects", "all"

        Returns:
            파라미터 관련 컨텍스트
        """
        query = f"{parameter} 인과관계 영향"

        doc_types = None
        if context_type == "causes":
            doc_types = ["causal_relationship"]
        elif context_type == "effects":
            doc_types = ["causal_relationship", "alarm_cause"]

        return self.retrieve(query, n_results=5, doc_types=doc_types)

    def _format_context(self, results: List[SearchResult]) -> str:
        """
        검색 결과를 LLM 프롬프트용 컨텍스트로 포맷팅

        Args:
            results: 검색 결과 리스트

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        if not results:
            return ""

        formatted_parts = []

        for i, result in enumerate(results, 1):
            doc_type = result.metadata.get("doc_type", "unknown")
            source = result.metadata.get("source", "unknown")
            similarity = result.similarity

            # 문서 타입별 헤더
            type_labels = {
                "causal_relationship": "인과관계",
                "alarm_cause": "알람 원인",
                "leading_indicator": "선행 지표",
                "discovered_relationship": "발견된 관계",
                "domain_knowledge": "도메인 지식",
                "impossible_relationship": "제외 관계",
            }
            type_label = type_labels.get(doc_type, doc_type)

            part = f"[{i}. {type_label}] (유사도: {similarity:.0%})\n{result.content}"
            formatted_parts.append(part)

        return "\n\n---\n\n".join(formatted_parts)

    def get_related_parameters(self, parameter: str, n_results: int = 5) -> List[str]:
        """
        파라미터와 관련된 다른 파라미터 목록 반환

        Args:
            parameter: 기준 파라미터

        Returns:
            관련 파라미터 이름 리스트
        """
        results = self.store.search(
            f"{parameter} 관계",
            n_results=n_results,
            where={"doc_type": {"$in": ["causal_relationship", "discovered_relationship"]}}
        )

        related = set()
        for r in results:
            source = r.metadata.get("source_param", "")
            target = r.metadata.get("target_param", "")
            if source and source != parameter:
                related.add(source)
            if target and target != parameter:
                related.add(target)

        return list(related)


class RAGManager:
    """RAG 시스템 관리자"""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "fdc_knowledge"
    ):
        # SimpleVectorStore는 JSON 파일 경로 사용
        import os
        persist_path = os.path.join(persist_dir, f"{collection_name}.json")
        self.store = SimpleVectorStore(persist_path=persist_path)
        self.retriever = RAGRetriever(self.store)
        self._initialized = False

    def initialize(
        self,
        expert_knowledge_path: Optional[str] = None,
        discovered_relationships_path: Optional[str] = None,
        domain_knowledge_text: Optional[str] = None,
        clear_existing: bool = False
    ) -> Dict[str, int]:
        """
        지식베이스 초기화

        Args:
            expert_knowledge_path: expert_knowledge.json 경로
            discovered_relationships_path: discovered_relationships.json 경로
            domain_knowledge_text: 도메인 지식 텍스트
            clear_existing: 기존 데이터 삭제 여부

        Returns:
            로드된 문서 수 딕셔너리
        """
        from .document_loader import DocumentLoader

        if clear_existing:
            self.store.clear()

        loader = DocumentLoader()
        stats = {}

        if expert_knowledge_path:
            docs = loader.load_expert_knowledge(expert_knowledge_path)
            count = self.store.add_documents(docs)
            stats["expert_knowledge"] = count

        if discovered_relationships_path:
            docs = loader.load_discovered_relationships(discovered_relationships_path)
            count = self.store.add_documents(docs)
            stats["discovered_relationships"] = count

        if domain_knowledge_text:
            docs = loader.load_domain_knowledge_text(domain_knowledge_text)
            count = self.store.add_documents(docs)
            stats["domain_knowledge"] = count

        self._initialized = True
        logger.info(f"RAG initialized with {sum(stats.values())} total documents")
        return stats

    def add_document(self, content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """단일 문서 추가"""
        from .simple_store import Document
        import uuid

        doc_id = doc_id or f"manual_{uuid.uuid4().hex[:8]}"
        doc = Document(id=doc_id, content=content, metadata=metadata)
        self.store.add_documents([doc])
        return doc_id

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """검색 (디버깅용)"""
        return self.retriever.retrieve_with_sources(query, n_results=n_results)

    def get_context_for_query(self, query: str) -> str:
        """쿼리에 대한 컨텍스트 반환 (Agent 통합용)"""
        return self.retriever.retrieve(query)

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        return self.store.get_stats()

    @property
    def is_initialized(self) -> bool:
        """초기화 여부"""
        return self._initialized or self.store.count() > 0
