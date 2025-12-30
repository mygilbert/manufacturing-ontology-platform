#!/usr/bin/env python
"""
RAG 지식베이스 초기화 스크립트
==============================

기존 지식 소스를 ChromaDB에 로딩합니다.

실행:
    cd manufacturing-ontology-platform
    python scripts/init_rag.py

옵션:
    --clear     기존 데이터 삭제 후 재로딩
    --test      테스트 검색 실행

소스:
    - analytics/results/expert_knowledge.json
    - analytics/results/discovered_relationships.json
    - fdc_agent.py DOMAIN_KNOWLEDGE (배터리 제조 지식)
"""

import os
import sys
import argparse
import logging

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
analytics_src = os.path.join(project_root, 'analytics', 'src')
sys.path.insert(0, analytics_src)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='RAG 지식베이스 초기화')
    parser.add_argument('--clear', action='store_true', help='기존 데이터 삭제 후 재로딩')
    parser.add_argument('--test', action='store_true', help='테스트 검색 실행')
    parser.add_argument('--persist-dir', default='./chroma_db', help='ChromaDB 저장 경로')
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 지식베이스 초기화")
    print("=" * 60)

    try:
        from rag import RAGManager
        from rag.document_loader import DocumentLoader
    except ImportError as e:
        print(f"\nError: RAG 모듈을 가져올 수 없습니다: {e}")
        print("chromadb와 sentence-transformers를 설치하세요:")
        print("  pip install chromadb sentence-transformers")
        sys.exit(1)

    # RAG 매니저 초기화
    persist_dir = os.path.join(project_root, 'analytics', args.persist_dir)
    print(f"\nChromaDB 경로: {persist_dir}")

    rag_manager = RAGManager(persist_dir=persist_dir)

    if args.clear:
        print("\n기존 데이터 삭제 중...")
        rag_manager.store.clear()
        print("완료!")

    # 지식 소스 경로
    expert_path = os.path.join(project_root, 'analytics', 'results', 'expert_knowledge.json')
    discovered_path = os.path.join(project_root, 'analytics', 'results', 'discovered_relationships.json')

    # 도메인 지식 (FDC Agent에서 가져오기)
    domain_knowledge = None
    try:
        from agent.fdc_agent import FDCAnalysisAgent
        domain_knowledge = FDCAnalysisAgent.DOMAIN_KNOWLEDGE
        print("\n도메인 지식 로드: FDCAnalysisAgent.DOMAIN_KNOWLEDGE")
    except ImportError:
        print("\nWarning: FDCAnalysisAgent를 가져올 수 없습니다. 도메인 지식 스킵.")

    # 초기화 실행
    print("\n" + "-" * 40)
    print("지식베이스 초기화 중...")
    print("-" * 40)

    stats = rag_manager.initialize(
        expert_knowledge_path=expert_path if os.path.exists(expert_path) else None,
        discovered_relationships_path=discovered_path if os.path.exists(discovered_path) else None,
        domain_knowledge_text=domain_knowledge,
        clear_existing=False  # 이미 위에서 clear 했음
    )

    print(f"\n로드된 문서:")
    for source, count in stats.items():
        print(f"  - {source}: {count}개")
    print(f"\n총 문서 수: {rag_manager.store.count()}개")

    # 통계 출력
    print("\n" + "-" * 40)
    print("컬렉션 통계")
    print("-" * 40)
    store_stats = rag_manager.get_stats()
    print(f"  컬렉션: {store_stats.get('collection_name', 'N/A')}")
    print(f"  임베딩 모델: {store_stats.get('embedding_model', 'N/A')}")
    print(f"  문서 타입별:")
    for doc_type, count in store_stats.get('documents_by_type', {}).items():
        print(f"    - {doc_type}: {count}개")

    # 테스트 검색
    if args.test:
        print("\n" + "=" * 60)
        print("테스트 검색")
        print("=" * 60)

        test_queries = [
            "Cell 용량 불량 원인",
            "RF Power와 Etch Rate 관계",
            "Module 발열 이상 점검",
            "알람 ETCH_RATE_OOS 조치",
        ]

        for query in test_queries:
            print(f"\n[Query] {query}")
            print("-" * 40)

            result = rag_manager.search(query, n_results=3)

            if result.get("n_results", 0) > 0:
                for i, source in enumerate(result.get("sources", []), 1):
                    print(f"  [{i}] {source.get('doc_type', 'unknown')} (유사도: {source.get('similarity', 0):.0%})")
                    print(f"      {source.get('preview', '')[:80]}...")
            else:
                print("  검색 결과 없음")

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
