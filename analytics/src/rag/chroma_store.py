# ============================================================
# ChromaDB Vector Store Wrapper
# ============================================================
"""
ChromaDB 기반 벡터 스토어
- 문서 저장 및 임베딩
- 시맨틱 검색
- 메타데이터 필터링
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """문서 데이터 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

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


class ChromaStore:
    """ChromaDB 벡터 스토어 래퍼"""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "fdc_knowledge",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Args:
            persist_dir: ChromaDB 영구 저장 경로
            collection_name: 컬렉션 이름
            embedding_model: sentence-transformers 임베딩 모델
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self._client = None
        self._collection = None
        self._embedding_function = None

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB"""
        if self._client is None:
            try:
                import chromadb
                from chromadb.utils import embedding_functions

                # 영구 저장소 클라이언트 생성
                os.makedirs(self.persist_dir, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_dir)

                # 임베딩 함수 설정
                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model
                )

                # 컬렉션 생성 또는 가져오기
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self._embedding_function,
                    metadata={"hnsw:space": "cosine"}  # 코사인 유사도 사용
                )

                logger.info(f"ChromaDB initialized: {self.persist_dir}, collection: {self.collection_name}")
                logger.info(f"Embedding model: {self.embedding_model}")
                logger.info(f"Documents in collection: {self._collection.count()}")

            except ImportError as e:
                logger.error(f"Failed to import chromadb: {e}")
                logger.error("Install with: pip install chromadb sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                raise

    @property
    def client(self):
        self._ensure_initialized()
        return self._client

    @property
    def collection(self):
        self._ensure_initialized()
        return self._collection

    def add_documents(self, documents: List[Document]) -> int:
        """
        문서 추가

        Args:
            documents: 추가할 문서 리스트

        Returns:
            추가된 문서 수
        """
        if not documents:
            return 0

        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 기존 문서 업데이트 (upsert)
        self.collection.upsert(
            ids=ids,
            documents=contents,
            metadatas=metadatas
        )

        logger.info(f"Added/updated {len(documents)} documents to collection")
        return len(documents)

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        시맨틱 검색

        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            where: 메타데이터 필터 (예: {"doc_type": "causal_relationship"})
            where_document: 문서 내용 필터

        Returns:
            검색 결과 리스트
        """
        if self.collection.count() == 0:
            logger.warning("Collection is empty, no results to return")
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )

            search_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    search_results.append(SearchResult(
                        id=doc_id,
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        distance=results["distances"][0][i] if results["distances"] else 0.0
                    ))

            logger.debug(f"Search '{query[:50]}...' returned {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_document(self, doc_id: str) -> Optional[Document]:
        """단일 문서 조회"""
        try:
            result = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
            if result and result["ids"]:
                return Document(
                    id=result["ids"][0],
                    content=result["documents"][0] if result["documents"] else "",
                    metadata=result["metadatas"][0] if result["metadatas"] else {}
                )
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
        return None

    def delete_documents(self, ids: List[str]) -> int:
        """문서 삭제"""
        if not ids:
            return 0
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return 0

    def delete_by_metadata(self, where: Dict[str, Any]) -> int:
        """메타데이터 조건으로 문서 삭제"""
        try:
            # 먼저 해당 문서들을 찾음
            results = self.collection.get(where=where, include=["metadatas"])
            if results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} documents matching {where}")
                return len(results["ids"])
        except Exception as e:
            logger.error(f"Failed to delete by metadata: {e}")
        return 0

    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """문서 목록 조회"""
        try:
            results = self.collection.get(
                where=where,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"]
            )

            documents = []
            if results and results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    documents.append(Document(
                        id=doc_id,
                        content=results["documents"][i] if results["documents"] else "",
                        metadata=results["metadatas"][i] if results["metadatas"] else {}
                    ))
            return documents

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    def count(self) -> int:
        """전체 문서 수"""
        return self.collection.count()

    def clear(self) -> None:
        """컬렉션 초기화"""
        try:
            self._ensure_initialized()
            self.client.delete_collection(self.collection_name)
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """컬렉션 통계"""
        try:
            count = self.collection.count()

            # 문서 타입별 카운트
            type_counts = {}
            docs = self.list_documents(limit=1000)
            for doc in docs:
                doc_type = doc.metadata.get("doc_type", "unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            return {
                "collection_name": self.collection_name,
                "persist_dir": self.persist_dir,
                "embedding_model": self.embedding_model,
                "total_documents": count,
                "documents_by_type": type_counts
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
