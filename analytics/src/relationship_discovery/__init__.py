"""
Implicit Relationship Discovery Module
======================================

FDC/SPC 데이터에서 암묵적 관계를 자동으로 발견하는 모듈

주요 기능:
1. 상관관계 분석 (Correlation Analysis)
2. 인과성 분석 (Granger Causality, Transfer Entropy)
3. 이벤트 패턴 발견 (Sequential Pattern Mining)
4. 발견된 관계를 온톨로지에 저장

CPU 버전:
    from relationship_discovery import DiscoveryPipeline

    pipeline = DiscoveryPipeline(config)
    relationships = pipeline.discover_all(data)

GPU 가속 버전 (PyTorch CUDA):
    from relationship_discovery import GPURelationshipDiscovery

    discovery = GPURelationshipDiscovery()
    relationships = discovery.discover_all(data, columns)
"""

from .config import DiscoveryConfig
from .correlation_analyzer import CorrelationAnalyzer
from .causality_analyzer import CausalityAnalyzer
from .pattern_detector import PatternDetector
from .relationship_store import RelationshipStore
from .discovery_pipeline import DiscoveryPipeline

# GPU 가속 모듈
try:
    from .gpu_accelerated import (
        GPURelationshipDiscovery,
        GPUDiscoveryConfig,
        GPUCorrelationAnalyzer,
        GPUCausalityAnalyzer,
        GPURollingStats,
        gpu_correlation_matrix,
        quick_discover,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    # CPU 버전
    'DiscoveryConfig',
    'CorrelationAnalyzer',
    'CausalityAnalyzer',
    'PatternDetector',
    'RelationshipStore',
    'DiscoveryPipeline',
    # GPU 버전
    'GPURelationshipDiscovery',
    'GPUDiscoveryConfig',
    'GPUCorrelationAnalyzer',
    'GPUCausalityAnalyzer',
    'GPURollingStats',
    'gpu_correlation_matrix',
    'quick_discover',
    'GPU_AVAILABLE',
]

__version__ = '1.1.0'  # GPU 가속 추가
