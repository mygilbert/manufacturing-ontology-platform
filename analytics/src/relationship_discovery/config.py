"""
Configuration for Relationship Discovery Module
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class DiscoveryMethod(Enum):
    """관계 발견 방법"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    GRANGER = "granger"
    TRANSFER_ENTROPY = "transfer_entropy"
    CROSS_CORRELATION = "cross_correlation"
    MUTUAL_INFORMATION = "mutual_information"


class RelationType(Enum):
    """발견된 관계 유형"""
    CORRELATION = "CORRELATES_WITH"
    INFLUENCES = "INFLUENCES"
    CAUSES = "CAUSES"
    PRECEDES = "PRECEDES"
    CO_OCCURS = "CO_OCCURS"
    ROOT_CAUSE = "ROOT_CAUSE_OF"


@dataclass
class CorrelationConfig:
    """상관관계 분석 설정"""
    methods: List[str] = field(default_factory=lambda: ["pearson", "spearman"])
    min_correlation: float = 0.5          # 최소 상관계수 임계값
    max_p_value: float = 0.05             # 최대 p-value
    min_samples: int = 100                # 최소 샘플 수
    rolling_window: Optional[int] = None  # 롤링 윈도우 (시간 변화 분석용)


@dataclass
class CausalityConfig:
    """인과성 분석 설정"""
    max_lag: int = 10                     # 최대 시간 지연 (초/샘플)
    significance_level: float = 0.05      # 유의 수준
    min_samples: int = 500                # 최소 샘플 수
    use_granger: bool = True              # Granger 인과성
    use_transfer_entropy: bool = False    # Transfer Entropy (계산 비용 높음)
    use_ccm: bool = False                 # Convergent Cross Mapping


@dataclass
class PatternConfig:
    """이벤트 패턴 발견 설정"""
    min_support: float = 0.1              # 최소 지지도
    min_confidence: float = 0.5           # 최소 신뢰도
    max_pattern_length: int = 5           # 최대 패턴 길이
    time_window_seconds: int = 300        # 이벤트 윈도우 (초)
    event_types: List[str] = field(default_factory=lambda: [
        "ALARM", "STATE_CHANGE", "RECIPE_CHANGE", "PM_EVENT"
    ])


@dataclass
class OntologyConfig:
    """온톨로지 저장 설정"""
    host: str = "localhost"
    port: int = 5432
    database: str = "manufacturing"
    user: str = "ontology"
    password: str = "ontology123"
    graph_name: str = "manufacturing"

    # 관계 저장 정책
    auto_save: bool = False               # 자동 저장 여부
    require_verification: bool = True      # 전문가 검증 필요 여부
    min_confidence: float = 0.7           # 최소 신뢰도 (저장 기준)


@dataclass
class DiscoveryConfig:
    """통합 설정"""
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    causality: CausalityConfig = field(default_factory=CausalityConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    ontology: OntologyConfig = field(default_factory=OntologyConfig)

    # 병렬 처리
    n_jobs: int = -1                      # -1 = 모든 코어 사용
    chunk_size: int = 10000               # 청크 크기

    # 로깅
    verbose: bool = True
    log_file: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'DiscoveryConfig':
        """환경 변수에서 설정 로드"""
        import os

        config = cls()
        config.ontology.host = os.getenv('POSTGRES_HOST', 'localhost')
        config.ontology.port = int(os.getenv('POSTGRES_PORT', 5432))
        config.ontology.database = os.getenv('POSTGRES_DB', 'manufacturing')
        config.ontology.user = os.getenv('POSTGRES_USER', 'ontology')
        config.ontology.password = os.getenv('POSTGRES_PASSWORD', 'ontology123')

        return config

    @classmethod
    def for_realtime(cls) -> 'DiscoveryConfig':
        """실시간 분석용 경량 설정"""
        config = cls()
        config.correlation.methods = ["pearson"]
        config.correlation.min_samples = 50
        config.causality.max_lag = 5
        config.causality.use_transfer_entropy = False
        config.pattern.max_pattern_length = 3
        config.n_jobs = 2
        return config

    @classmethod
    def for_batch(cls) -> 'DiscoveryConfig':
        """배치 분석용 상세 설정"""
        config = cls()
        config.correlation.methods = ["pearson", "spearman", "kendall"]
        config.correlation.rolling_window = 60
        config.causality.max_lag = 30
        config.causality.use_transfer_entropy = True
        config.pattern.max_pattern_length = 7
        config.n_jobs = -1
        return config
