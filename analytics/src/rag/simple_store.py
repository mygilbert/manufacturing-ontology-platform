# ============================================================
# Simple Vector Store (ChromaDB 대안)
# ============================================================
"""
JSON 기반 벡터 스토어 - ChromaDB Python 3.13 호환성 문제 해결용
- Ollama API로 임베딩 생성
- JSON 파일에 저장
- NumPy로 코사인 유사도 계산
"""

import os
import json
import logging
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """문서 데이터 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any]
    distance: float  # 낮을수록 유사함

    @property
    def similarity(self) -> float:
        """거리를 유사도(0-1)로 변환"""
        return max(0, 1 - self.distance)


class SimpleVectorStore:
    """JSON 기반 간단한 벡터 스토어"""

    def __init__(
        self,
        persist_path: str = "./vector_store.json",
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text"
    ):
        """
        Args:
            persist_path: JSON 저장 경로
            ollama_url: Ollama API URL
            embed_model: 임베딩 모델명
        """
        self.persist_path = persist_path
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.documents: Dict[str, Document] = {}

        # 기존 데이터 로드
        self._load()

    def _load(self):
        """저장된 데이터 로드"""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_data in data.get('documents', []):
                        doc = Document(
                            id=doc_data['id'],
                            content=doc_data['content'],
                            metadata=doc_data.get('metadata', {}),
                            embedding=doc_data.get('embedding')
                        )
                        self.documents[doc.id] = doc
                logger.info(f"Loaded {len(self.documents)} documents from {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to load: {e}")

    def _save(self):
        """데이터 저장"""
        try:
            os.makedirs(os.path.dirname(self.persist_path) or '.', exist_ok=True)
            data = {
                'documents': [
                    {
                        'id': doc.id,
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'embedding': doc.embedding
                    }
                    for doc in self.documents.values()
                ]
            }
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Ollama API로 임베딩 생성"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embed",
                json={"model": self.embed_model, "input": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    def add_documents(self, documents: List[Document]) -> int:
        """문서 추가 (임베딩 자동 생성)"""
        if not documents:
            return 0

        added = 0
        for doc in documents:
            try:
                if doc.embedding is None:
                    doc.embedding = self._get_embedding(doc.content)
                self.documents[doc.id] = doc
                added += 1
            except Exception as e:
                logger.error(f"Failed to add document {doc.id}: {e}")

        self._save()
        logger.info(f"Added {added}/{len(documents)} documents")
        return added

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """시맨틱 검색"""
        if not self.documents:
            return []

        try:
            query_embedding = self._get_embedding(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        # 모든 문서와 유사도 계산
        results = []
        for doc in self.documents.values():
            # 메타데이터 필터링
            if where:
                match = all(doc.metadata.get(k) == v for k, v in where.items())
                if not match:
                    continue

            if doc.embedding:
                similarity = self._cosine_similarity(query_embedding, doc.embedding)
                distance = 1 - similarity
                results.append(SearchResult(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    distance=distance
                ))

        # 거리순 정렬 (낮을수록 유사)
        results.sort(key=lambda x: x.distance)
        return results[:n_results]

    def get_document(self, doc_id: str) -> Optional[Document]:
        """단일 문서 조회"""
        return self.documents.get(doc_id)

    def delete_documents(self, ids: List[str]) -> int:
        """문서 삭제"""
        deleted = 0
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                deleted += 1
        self._save()
        return deleted

    def count(self) -> int:
        """문서 수"""
        return len(self.documents)

    def clear(self):
        """모든 문서 삭제"""
        self.documents.clear()
        self._save()

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """문서 목록"""
        docs = list(self.documents.values())
        return docs[offset:offset + limit]

    def get_stats(self) -> Dict[str, Any]:
        """통계 정보"""
        doc_types = {}
        for doc in self.documents.values():
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

        return {
            'total_documents': len(self.documents),
            'documents_by_type': doc_types,
            'persist_path': self.persist_path,
            'embed_model': self.embed_model
        }
