"""
Implicit Relationship Discovery Module
======================================

FDC/SPC 데이터에서 암묵적 관계를 자동으로 발견하는 모듈

주요 기능:
1. 상관관계 분석 (Correlation Analysis)
2. 인과성 분석 (Granger Causality, Transfer Entropy)
3. 이벤트 패턴 발견 (Sequential Pattern Mining)
4. 발견된 관계를 온톨로지에 저장

사용 예시:
    from relationship_discovery import DiscoveryPipeline

    pipeline = DiscoveryPipeline(config)
    relationships = pipeline.discover_all(data)
    pipeline.save_to_ontology(relationships)
"""

from .config import DiscoveryConfig
from .correlation_analyzer import CorrelationAnalyzer
from .causality_analyzer import CausalityAnalyzer
from .pattern_detector import PatternDetector
from .relationship_store import RelationshipStore
from .discovery_pipeline import DiscoveryPipeline

__all__ = [
    'DiscoveryConfig',
    'CorrelationAnalyzer',
    'CausalityAnalyzer',
    'PatternDetector',
    'RelationshipStore',
    'DiscoveryPipeline',
]

__version__ = '1.0.0'
