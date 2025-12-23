"""
알고리즘 추천 엔진

데이터 특성을 분석하여 최적의 이상감지 알고리즘을 자동으로 추천합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .data_loader import DataCharacteristics
from .evaluator import ModelEvaluator, EvaluationResult


@dataclass
class AlgorithmInfo:
    """알고리즘 정보"""
    name: str
    category: str  # 'statistical', 'ml', 'deep_learning'
    description: str
    pros: List[str]
    cons: List[str]
    suitable_for: List[str]
    min_samples: int
    complexity: str  # 'low', 'medium', 'high'
    interpretability: str  # 'high', 'medium', 'low'


# 알고리즘 메타 정보
ALGORITHM_REGISTRY = {
    'Z-Score Detector': AlgorithmInfo(
        name='Z-Score Detector',
        category='statistical',
        description='표준화 점수 기반의 단순하고 빠른 이상감지',
        pros=['빠른 속도', '해석 용이', '구현 간단'],
        cons=['다변량 패턴 감지 어려움', '정규 분포 가정'],
        suitable_for=['정규 분포 데이터', '단변량 분석', '실시간 모니터링'],
        min_samples=100,
        complexity='low',
        interpretability='high'
    ),
    'CUSUM Detector': AlgorithmInfo(
        name='CUSUM Detector',
        category='statistical',
        description='누적합 기반 평균 이동 감지에 특화',
        pros=['드리프트 감지 우수', '순차적 모니터링 적합'],
        cons=['급격한 변화 감지 지연', '파라미터 설정 필요'],
        suitable_for=['점진적 변화 감지', '프로세스 모니터링', '품질 관리'],
        min_samples=100,
        complexity='low',
        interpretability='high'
    ),
    'SPC Control Chart Detector': AlgorithmInfo(
        name='SPC Control Chart Detector',
        category='statistical',
        description='통계적 공정 관리 기반 관리도 방법',
        pros=['산업 표준', 'Western Electric 규칙 적용', '해석 용이'],
        cons=['정상 상태 데이터 필요', '복잡한 패턴 감지 어려움'],
        suitable_for=['제조 공정', '품질 관리', '규제 환경'],
        min_samples=200,
        complexity='low',
        interpretability='high'
    ),
    'Isolation Forest': AlgorithmInfo(
        name='Isolation Forest',
        category='ml',
        description='격리 기반 앙상블 이상감지',
        pros=['고차원 데이터 적합', '비선형 패턴', '학습 불필요(라벨)'],
        cons=['시계열 특성 무시', '해석 어려움'],
        suitable_for=['고차원 데이터', '비선형 패턴', '대규모 데이터'],
        min_samples=500,
        complexity='medium',
        interpretability='medium'
    ),
    'Local Outlier Factor': AlgorithmInfo(
        name='Local Outlier Factor',
        category='ml',
        description='밀도 기반 지역적 이상치 탐지',
        pros=['지역 밀도 고려', '클러스터 경계 감지'],
        cons=['계산 비용 높음', '파라미터 민감'],
        suitable_for=['군집 구조 데이터', '지역적 이상치', '중소규모 데이터'],
        min_samples=300,
        complexity='medium',
        interpretability='medium'
    ),
    'One-Class SVM': AlgorithmInfo(
        name='One-Class SVM',
        category='ml',
        description='커널 기반 단일 클래스 분류',
        pros=['비선형 경계', '고차원 적합'],
        cons=['대규모 데이터 느림', '커널 선택 필요'],
        suitable_for=['복잡한 결정 경계', '중소규모 데이터'],
        min_samples=300,
        complexity='medium',
        interpretability='low'
    ),
    'AutoEncoder': AlgorithmInfo(
        name='AutoEncoder',
        category='deep_learning',
        description='재구성 오차 기반 딥러닝 이상감지',
        pros=['복잡한 패턴 학습', '비선형 관계 모델링'],
        cons=['학습 데이터 필요', '해석 어려움', 'GPU 권장'],
        suitable_for=['복잡한 다변량 패턴', '대규모 데이터', '비선형 관계'],
        min_samples=5000,
        complexity='high',
        interpretability='low'
    ),
    'LSTM AutoEncoder': AlgorithmInfo(
        name='LSTM AutoEncoder',
        category='deep_learning',
        description='시계열 특화 LSTM 기반 이상감지',
        pros=['시계열 의존성 모델링', '시퀀스 패턴 학습'],
        cons=['학습 시간 김', '시퀀스 길이 설정', 'GPU 필요'],
        suitable_for=['강한 시계열 의존성', '시퀀스 패턴', '대규모 시계열'],
        min_samples=10000,
        complexity='high',
        interpretability='low'
    ),
}


@dataclass
class Recommendation:
    """추천 결과"""
    rank: int
    algorithm_name: str
    score: float
    reasons: List[str]
    warnings: List[str]
    algorithm_info: AlgorithmInfo
    estimated_performance: Optional[EvaluationResult] = None


class AlgorithmRecommender:
    """알고리즘 추천 엔진"""

    def __init__(self):
        self.recommendations: List[Recommendation] = []
        self.data_characteristics: Optional[DataCharacteristics] = None
        self.evaluator = ModelEvaluator()

    def analyze_and_recommend(
        self,
        characteristics: DataCharacteristics,
        priorities: Optional[Dict[str, float]] = None
    ) -> List[Recommendation]:
        """
        데이터 특성 분석 및 알고리즘 추천

        Args:
            characteristics: 데이터 특성
            priorities: 우선순위 가중치
                - 'speed': 속도 중요도
                - 'accuracy': 정확도 중요도
                - 'interpretability': 해석 가능성 중요도
                - 'early_detection': 조기 감지 중요도
        """
        self.data_characteristics = characteristics

        if priorities is None:
            priorities = {
                'speed': 0.2,
                'accuracy': 0.4,
                'interpretability': 0.2,
                'early_detection': 0.2
            }

        algorithm_scores = []

        for name, info in ALGORITHM_REGISTRY.items():
            score, reasons, warnings = self._calculate_score(
                info, characteristics, priorities
            )
            algorithm_scores.append({
                'name': name,
                'info': info,
                'score': score,
                'reasons': reasons,
                'warnings': warnings
            })

        # 점수순 정렬
        algorithm_scores.sort(key=lambda x: x['score'], reverse=True)

        self.recommendations = []
        for rank, item in enumerate(algorithm_scores, 1):
            rec = Recommendation(
                rank=rank,
                algorithm_name=item['name'],
                score=item['score'],
                reasons=item['reasons'],
                warnings=item['warnings'],
                algorithm_info=item['info']
            )
            self.recommendations.append(rec)

        return self.recommendations

    def _calculate_score(
        self,
        info: AlgorithmInfo,
        characteristics: DataCharacteristics,
        priorities: Dict[str, float]
    ) -> Tuple[float, List[str], List[str]]:
        """알고리즘 적합도 점수 계산"""
        score = 0.0
        reasons = []
        warnings = []

        # 1. 데이터 크기 적합성
        if characteristics.n_samples >= info.min_samples:
            size_score = 1.0
            reasons.append(f"데이터 크기 충분 ({characteristics.n_samples} >= {info.min_samples})")
        else:
            size_score = characteristics.n_samples / info.min_samples
            warnings.append(f"데이터 부족 ({characteristics.n_samples} < {info.min_samples} 권장)")

        score += size_score * 0.2

        # 2. 복잡도 vs 속도 우선순위
        complexity_scores = {'low': 1.0, 'medium': 0.6, 'high': 0.3}
        if priorities['speed'] > 0.3:
            speed_score = complexity_scores[info.complexity]
            if info.complexity == 'low':
                reasons.append("빠른 처리 속도")
        else:
            speed_score = 1.0 - complexity_scores[info.complexity] * 0.5

        score += speed_score * priorities['speed']

        # 3. 해석 가능성
        interpretability_scores = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        interp_score = interpretability_scores[info.interpretability]

        if priorities['interpretability'] > 0.3 and info.interpretability == 'high':
            reasons.append("높은 해석 가능성")

        score += interp_score * priorities['interpretability']

        # 4. 시계열 의존성
        if characteristics.temporal_dependency > 0.5:
            if info.name == 'LSTM AutoEncoder':
                score += 0.3
                reasons.append("강한 시계열 의존성에 적합")
            elif info.name == 'CUSUM Detector':
                score += 0.2
                reasons.append("시계열 드리프트 감지에 효과적")
            elif info.category == 'statistical':
                score += 0.1

        # 5. 노이즈 수준
        if characteristics.noise_level > 0.3:
            if info.category == 'deep_learning':
                score += 0.1
                reasons.append("노이즈에 강건한 딥러닝")
            elif info.name in ['Isolation Forest', 'Local Outlier Factor']:
                score += 0.1
        else:
            if info.category == 'statistical':
                score += 0.1
                reasons.append("낮은 노이즈에서 통계적 방법 효과적")

        # 6. 이상 비율
        if characteristics.anomaly_ratio < 0.01:
            if info.name == 'Isolation Forest':
                score += 0.15
                reasons.append("낮은 이상 비율에 Isolation Forest 효과적")
        elif characteristics.anomaly_ratio > 0.1:
            warnings.append("높은 이상 비율 - 정상 데이터 정의 재검토 필요")

        # 7. 차원 수
        if characteristics.n_features > 20:
            if info.name == 'Isolation Forest':
                score += 0.15
                reasons.append("고차원 데이터에 효과적")
            elif info.category == 'deep_learning':
                score += 0.1
            elif info.category == 'statistical':
                score -= 0.1
                warnings.append("고차원 데이터에서 성능 저하 가능")

        # 8. 특성 상관관계
        if characteristics.feature_correlations > 0.5:
            if info.category == 'deep_learning':
                score += 0.1
                reasons.append("높은 특성 상관관계 모델링 가능")

        # 9. 조기 감지 우선순위
        if priorities['early_detection'] > 0.3:
            if info.name in ['CUSUM Detector', 'LSTM AutoEncoder']:
                score += 0.1
                reasons.append("조기 감지에 효과적")

        # 점수 정규화
        score = min(1.0, max(0.0, score))

        return score, reasons, warnings

    def get_top_recommendations(self, n: int = 3) -> List[Recommendation]:
        """상위 N개 추천 반환"""
        return self.recommendations[:n]

    def run_comparison(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        algorithms: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        추천 알고리즘들의 실제 성능 비교

        Args:
            X_train, X_test: 특성 데이터
            y_train, y_test: 라벨
            algorithms: 비교할 알고리즘 목록 (None이면 상위 5개)
        """
        if algorithms is None:
            algorithms = [r.algorithm_name for r in self.recommendations[:5]]

        self.evaluator.reset()

        from .algorithms import (
            ZScoreDetector, CUSUMDetector, SPCDetector,
            IsolationForestDetector, LOFDetector, OneClassSVMDetector,
            AutoEncoderDetector, LSTMAutoEncoderDetector
        )

        algorithm_classes = {
            'Z-Score Detector': ZScoreDetector,
            'CUSUM Detector': CUSUMDetector,
            'SPC Control Chart Detector': SPCDetector,
            'Isolation Forest': IsolationForestDetector,
            'Local Outlier Factor': LOFDetector,
            'One-Class SVM': OneClassSVMDetector,
            'AutoEncoder': AutoEncoderDetector,
            'LSTM AutoEncoder': LSTMAutoEncoderDetector,
        }

        for algo_name in algorithms:
            if algo_name not in algorithm_classes:
                print(f"알 수 없는 알고리즘: {algo_name}")
                continue

            print(f"\n평가 중: {algo_name}...")

            try:
                # 모델 초기화
                detector = algorithm_classes[algo_name]()

                # 학습
                start_time = time.time()
                detector.fit(X_train, y_train)
                training_time = time.time() - start_time

                # 예측
                start_time = time.time()
                y_pred = detector.predict(X_test)
                prediction_time = time.time() - start_time

                # 점수
                y_score = detector.predict_score(X_test)

                # 평가
                result = self.evaluator.evaluate(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_score=y_score,
                    algorithm_name=algo_name,
                    training_time=training_time,
                    prediction_time=prediction_time
                )

                # 추천에 실제 성능 추가
                for rec in self.recommendations:
                    if rec.algorithm_name == algo_name:
                        rec.estimated_performance = result
                        break

                print(f"  - F1: {result.f1_score:.4f}, AUC: {result.auc_roc:.4f}")

            except Exception as e:
                print(f"  - 오류 발생: {str(e)}")

        return self.evaluator.compare_algorithms()

    def generate_recommendation_report(self) -> str:
        """추천 리포트 생성"""
        if not self.recommendations:
            return "추천 결과가 없습니다. analyze_and_recommend()를 먼저 실행하세요."

        report = []
        report.append("=" * 70)
        report.append("FDC 이상감지 알고리즘 추천 리포트")
        report.append("=" * 70)
        report.append("")

        # 데이터 특성 요약
        if self.data_characteristics:
            report.append("## 데이터 특성 분석")
            report.append(f"  - 샘플 수: {self.data_characteristics.n_samples:,}")
            report.append(f"  - 특성 수: {self.data_characteristics.n_features}")
            report.append(f"  - 이상 비율: {self.data_characteristics.anomaly_ratio*100:.2f}%")
            report.append(f"  - 시계열 의존성: {self.data_characteristics.temporal_dependency:.3f}")
            report.append(f"  - 노이즈 수준: {self.data_characteristics.noise_level:.3f}")
            report.append(f"  - 트렌드 존재: {'예' if self.data_characteristics.has_trend else '아니오'}")
            report.append(f"  - 계절성 존재: {'예' if self.data_characteristics.has_seasonality else '아니오'}")
            report.append("")

        # 추천 알고리즘
        report.append("## 추천 알고리즘 순위")
        report.append("")

        for rec in self.recommendations[:5]:
            report.append(f"### {rec.rank}위: {rec.algorithm_name} (점수: {rec.score:.2f})")
            report.append(f"    카테고리: {rec.algorithm_info.category}")
            report.append(f"    설명: {rec.algorithm_info.description}")

            if rec.reasons:
                report.append(f"    추천 이유:")
                for reason in rec.reasons:
                    report.append(f"      + {reason}")

            if rec.warnings:
                report.append(f"    주의사항:")
                for warning in rec.warnings:
                    report.append(f"      ! {warning}")

            if rec.estimated_performance:
                perf = rec.estimated_performance
                report.append(f"    실제 성능:")
                report.append(f"      - F1 Score: {perf.f1_score:.4f}")
                report.append(f"      - Precision: {perf.precision:.4f}")
                report.append(f"      - Recall: {perf.recall:.4f}")
                report.append(f"      - AUC-ROC: {perf.auc_roc:.4f}")

            report.append("")

        # 최종 추천
        top_rec = self.recommendations[0]
        report.append("=" * 70)
        report.append(f"## 최종 추천: {top_rec.algorithm_name}")
        report.append("")
        report.append("장점:")
        for pro in top_rec.algorithm_info.pros:
            report.append(f"  + {pro}")
        report.append("")
        report.append("적합한 상황:")
        for suitable in top_rec.algorithm_info.suitable_for:
            report.append(f"  - {suitable}")
        report.append("=" * 70)

        return "\n".join(report)

    def get_implementation_guide(self, algorithm_name: str) -> str:
        """알고리즘 구현 가이드"""
        if algorithm_name not in ALGORITHM_REGISTRY:
            return f"알 수 없는 알고리즘: {algorithm_name}"

        info = ALGORITHM_REGISTRY[algorithm_name]

        guide = f"""
## {algorithm_name} 구현 가이드

### 개요
{info.description}

### 카테고리
{info.category}

### 권장 사용 환경
- 최소 데이터 샘플: {info.min_samples:,}개
- 계산 복잡도: {info.complexity}
- 해석 가능성: {info.interpretability}

### 장점
{chr(10).join('- ' + p for p in info.pros)}

### 단점
{chr(10).join('- ' + c for c in info.cons)}

### 적합한 사용 사례
{chr(10).join('- ' + s for s in info.suitable_for)}

### 코드 예시
```python
from anomaly_detection.algorithms import {algorithm_name.replace(' ', '').replace('-', '')}

# 모델 초기화
detector = {algorithm_name.replace(' ', '').replace('-', '')}()

# 학습 (정상 데이터 사용)
detector.fit(X_train, y_train)

# 예측
predictions = detector.predict(X_test)
scores = detector.predict_score(X_test)
```

### 하이퍼파라미터 튜닝 가이드
{algorithm_name}의 주요 파라미터와 튜닝 방향에 대한 가이드입니다.
데이터 특성에 따라 파라미터를 조정하여 최적의 성능을 얻으세요.
"""
        return guide
