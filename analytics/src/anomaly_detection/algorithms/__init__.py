"""
이상감지 알고리즘 패키지

통계, ML, 딥러닝 기반 이상감지 알고리즘을 제공합니다.
"""

from .statistical import ZScoreDetector, CUSUMDetector, SPCDetector
from .ml_based import IsolationForestDetector, LOFDetector, OneClassSVMDetector

__all__ = [
    # 통계 기반
    "ZScoreDetector",
    "CUSUMDetector",
    "SPCDetector",
    # ML 기반
    "IsolationForestDetector",
    "LOFDetector",
    "OneClassSVMDetector",
]

# 딥러닝 모듈 (TensorFlow 필요)
try:
    from .deep_learning import AutoEncoderDetector, LSTMAutoEncoderDetector
    __all__.extend(["AutoEncoderDetector", "LSTMAutoEncoderDetector"])
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    AutoEncoderDetector = None
    LSTMAutoEncoderDetector = None
