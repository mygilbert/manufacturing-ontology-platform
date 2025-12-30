# ============================================================
# RAG (Retrieval-Augmented Generation) Module
# ============================================================
"""
벡터 스토어 기반 RAG 시스템
- 지식 저장 및 검색
- 문서 로딩 및 청킹
- 컨텍스트 검색 및 포맷팅

Note: ChromaDB Python 3.13 segfault 이슈로 SimpleVectorStore 사용
"""

from .simple_store import SimpleVectorStore, Document, SearchResult
from .document_loader import DocumentLoader
from .retriever import RAGRetriever, RAGManager

# ChromaStore 호환성 별칭
ChromaStore = SimpleVectorStore

__all__ = [
    "ChromaStore",
    "SimpleVectorStore",
    "Document",
    "SearchResult",
    "DocumentLoader",
    "RAGRetriever",
    "RAGManager",
]
