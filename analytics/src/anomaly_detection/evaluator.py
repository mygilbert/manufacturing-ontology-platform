"""
모델 평가 모듈

이상감지 알고리즘의 성능을 다양한 메트릭으로 평가합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)


@dataclass
class EvaluationResult:
    """평가 결과 데이터 클래스"""
    algorithm_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    confusion_matrix: np.ndarray
    early_detection_score: float
    training_time: float
    prediction_time: float


class ModelEvaluator:
    """모델 평가 클래스"""

    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.comparison_df: Optional[pd.DataFrame] = None

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        algorithm_name: str = "Unknown",
        timestamps: Optional[np.ndarray] = None,
        fault_start_times: Optional[List] = None,
        training_time: float = 0.0,
        prediction_time: float = 0.0
    ) -> EvaluationResult:
        """
        모델 성능 평가

        Args:
            y_true: 실제 라벨 (0: 정상, 1: 이상)
            y_pred: 예측 라벨
            y_score: 이상 점수 (AUC 계산용)
            algorithm_name: 알고리즘 이름
            timestamps: 타임스탬프 (조기 감지 평가용)
            fault_start_times: 실제 이상 시작 시점
            training_time: 학습 소요 시간
            prediction_time: 예측 소요 시간
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # 기본 메트릭 계산
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUC 계산
        if y_score is not None and len(np.unique(y_true)) > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_score)
                auc_pr = average_precision_score(y_true, y_score)
            except:
                auc_roc = 0.0
                auc_pr = 0.0
        else:
            auc_roc = 0.0
            auc_pr = 0.0

        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)

        # 조기 감지 점수
        early_detection = self._calculate_early_detection_score(
            y_true, y_pred, timestamps, fault_start_times
        )

        result = EvaluationResult(
            algorithm_name=algorithm_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            confusion_matrix=cm,
            early_detection_score=early_detection,
            training_time=training_time,
            prediction_time=prediction_time
        )

        self.results.append(result)
        return result

    def _calculate_early_detection_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        fault_start_times: Optional[List] = None
    ) -> float:
        """
        조기 감지 점수 계산

        이상이 시작되기 전 또는 시작 직후 얼마나 빨리 감지했는지 평가
        """
        if timestamps is None or fault_start_times is None:
            # 간단한 방식: 이상 구간 시작 인덱스 기준
            return self._simple_early_detection_score(y_true, y_pred)

        # 타임스탬프 기반 조기 감지 점수
        early_scores = []

        for fault_start in fault_start_times:
            # 해당 이상 구간 찾기
            fault_indices = np.where(timestamps >= fault_start)[0]
            if len(fault_indices) == 0:
                continue

            fault_start_idx = fault_indices[0]

            # 이상 시작 전후 윈도우에서 감지 여부 확인
            window_before = max(0, fault_start_idx - 60)  # 1분 전
            window_after = min(len(y_pred), fault_start_idx + 60)  # 1분 후

            pred_in_window = y_pred[window_before:window_after]

            if np.any(pred_in_window == 1):
                # 첫 감지 시점
                first_detection = np.where(pred_in_window == 1)[0][0]
                relative_position = first_detection / len(pred_in_window)

                # 빨리 감지할수록 높은 점수
                score = 1.0 - relative_position * 0.5
                early_scores.append(score)
            else:
                early_scores.append(0.0)

        return np.mean(early_scores) if early_scores else 0.0

    def _simple_early_detection_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """간단한 조기 감지 점수 계산"""
        # 이상 구간 시작점 찾기
        fault_starts = []
        in_fault = False

        for i in range(len(y_true)):
            if y_true[i] == 1 and not in_fault:
                fault_starts.append(i)
                in_fault = True
            elif y_true[i] == 0:
                in_fault = False

        if not fault_starts:
            return 0.0

        early_scores = []
        for start in fault_starts:
            # 이상 시작 전 10개 샘플 ~ 시작 후 10개 샘플 윈도우
            window_start = max(0, start - 10)
            window_end = min(len(y_pred), start + 10)

            pred_window = y_pred[window_start:window_end]

            if np.any(pred_window == 1):
                first_detection = np.where(pred_window == 1)[0][0]
                detection_offset = first_detection - (start - window_start)

                # 미리 감지하면 더 높은 점수
                if detection_offset < 0:
                    score = 1.0  # 미리 감지
                else:
                    score = max(0, 1.0 - detection_offset / 10)
                early_scores.append(score)
            else:
                early_scores.append(0.0)

        return np.mean(early_scores)

    def compare_algorithms(self) -> pd.DataFrame:
        """알고리즘 비교 테이블 생성"""
        if not self.results:
            return pd.DataFrame()

        comparison_data = []
        for result in self.results:
            comparison_data.append({
                '알고리즘': result.algorithm_name,
                '정확도': f"{result.accuracy:.4f}",
                '정밀도': f"{result.precision:.4f}",
                '재현율': f"{result.recall:.4f}",
                'F1 점수': f"{result.f1_score:.4f}",
                'AUC-ROC': f"{result.auc_roc:.4f}",
                'AUC-PR': f"{result.auc_pr:.4f}",
                '조기감지': f"{result.early_detection_score:.4f}",
                '학습시간(s)': f"{result.training_time:.2f}",
                '예측시간(s)': f"{result.prediction_time:.2f}"
            })

        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df

    def get_best_algorithm(
        self,
        metric: str = 'f1_score',
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[str, EvaluationResult]:
        """
        최고 성능 알고리즘 반환

        Args:
            metric: 기준 메트릭 ('f1_score', 'precision', 'recall', 'auc_roc', 'weighted')
            weights: 가중치 (metric='weighted'일 때 사용)
        """
        if not self.results:
            raise ValueError("평가 결과가 없습니다.")

        if metric == 'weighted':
            if weights is None:
                weights = {
                    'f1_score': 0.3,
                    'auc_roc': 0.2,
                    'early_detection_score': 0.2,
                    'precision': 0.15,
                    'recall': 0.15
                }

            best_score = -1
            best_result = None

            for result in self.results:
                score = sum(
                    getattr(result, m) * w
                    for m, w in weights.items()
                )
                if score > best_score:
                    best_score = score
                    best_result = result
        else:
            best_result = max(self.results, key=lambda r: getattr(r, metric))

        return best_result.algorithm_name, best_result

    def get_ranking(
        self,
        metric: str = 'f1_score'
    ) -> List[Tuple[str, float]]:
        """메트릭 기준 알고리즘 순위"""
        if not self.results:
            return []

        ranked = sorted(
            self.results,
            key=lambda r: getattr(r, metric),
            reverse=True
        )

        return [(r.algorithm_name, getattr(r, metric)) for r in ranked]

    def plot_comparison(
        self,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """알고리즘 비교 시각화"""
        import matplotlib.pyplot as plt

        if not self.results:
            print("평가 결과가 없습니다.")
            return

        metrics = ['precision', 'recall', 'f1_score', 'auc_roc', 'early_detection_score']
        algorithms = [r.algorithm_name for r in self.results]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 바 차트
        x = np.arange(len(algorithms))
        width = 0.15

        ax1 = axes[0]
        for i, metric in enumerate(metrics[:4]):
            values = [getattr(r, metric) for r in self.results]
            ax1.bar(x + i * width, values, width, label=metric)

        ax1.set_xlabel('알고리즘')
        ax1.set_ylabel('점수')
        ax1.set_title('알고리즘 성능 비교')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.1)

        # 레이더 차트
        ax2 = axes[1]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        ax2 = fig.add_subplot(122, projection='polar')

        for result in self.results:
            values = [getattr(result, m) for m in metrics]
            values += values[:1]
            ax2.plot(angles, values, 'o-', label=result.algorithm_name)
            ax2.fill(angles, values, alpha=0.1)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics)
        ax2.set_title('알고리즘 성능 프로파일')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.show()

        return fig

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        scores_dict: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (10, 8)
    ):
        """ROC 곡선 비교"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ROC Curve
        ax1 = axes[0]
        for name, scores in scores_dict.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()

        # Precision-Recall Curve
        ax2 = axes[1]
        for name, scores in scores_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax2.plot(recall, precision, label=f'{name} (AP={ap:.3f})')

        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        return fig

    def generate_report(self) -> str:
        """평가 리포트 생성"""
        if not self.results:
            return "평가 결과가 없습니다."

        report = []
        report.append("=" * 60)
        report.append("FDC 이상감지 알고리즘 평가 리포트")
        report.append("=" * 60)
        report.append("")

        # 비교 테이블
        comparison = self.compare_algorithms()
        report.append("## 성능 비교")
        report.append(comparison.to_string(index=False))
        report.append("")

        # 최고 성능 알고리즘
        best_name, best_result = self.get_best_algorithm('weighted')
        report.append("## 추천 알고리즘")
        report.append(f"- 최고 성능: {best_name}")
        report.append(f"  - F1 Score: {best_result.f1_score:.4f}")
        report.append(f"  - AUC-ROC: {best_result.auc_roc:.4f}")
        report.append(f"  - 조기 감지: {best_result.early_detection_score:.4f}")
        report.append("")

        # 순위
        report.append("## F1 Score 기준 순위")
        for i, (name, score) in enumerate(self.get_ranking('f1_score'), 1):
            report.append(f"  {i}. {name}: {score:.4f}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def reset(self):
        """평가 결과 초기화"""
        self.results = []
        self.comparison_df = None
