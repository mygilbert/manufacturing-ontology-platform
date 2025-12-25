"""
Real Sample Data - Relationship Discovery Test
===============================================

실제 sample_pv_data.csv로 암묵적 관계 발견 테스트
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from relationship_discovery import (
    DiscoveryConfig,
    CorrelationAnalyzer,
    CausalityAnalyzer,
    PatternDetector,
    DiscoveryPipeline
)


def load_real_data():
    """실제 샘플 데이터 로드"""
    data_path = os.path.join(
        os.path.dirname(__file__),
        '..', 'sample_data', 'sample_pv_data.csv'
    )

    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    print(f"데이터 로드 완료: {len(df):,} 샘플")
    print(f"컬럼: {list(df.columns)}")
    print(f"시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"이상 비율: {df['is_anomaly'].mean()*100:.2f}%")

    return df


def analyze_data_characteristics(df: pd.DataFrame):
    """데이터 특성 분석"""
    print("\n" + "=" * 60)
    print("데이터 특성 분석")
    print("=" * 60)

    pv_cols = ['temperature', 'pressure', 'vibration', 'current', 'flow_rate']

    print("\n기본 통계:")
    print(df[pv_cols].describe().round(3))

    # 이상 데이터 vs 정상 데이터 비교
    print("\n정상 vs 이상 데이터 평균 비교:")
    normal = df[df['is_anomaly'] == 0][pv_cols].mean()
    anomaly = df[df['is_anomaly'] == 1][pv_cols].mean()

    comparison = pd.DataFrame({
        '정상': normal,
        '이상': anomaly,
        '차이': anomaly - normal,
        '차이(%)': ((anomaly - normal) / normal * 100).round(2)
    })
    print(comparison.round(3))

    return pv_cols


def test_correlation_analysis(df: pd.DataFrame, pv_cols: list):
    """상관관계 분석"""
    print("\n" + "=" * 60)
    print("1. 상관관계 분석")
    print("=" * 60)

    # 설정 조정 (더 낮은 임계값)
    from relationship_discovery.config import CorrelationConfig
    config = CorrelationConfig(
        methods=['pearson', 'spearman'],
        min_correlation=0.3,  # 낮춤
        max_p_value=0.05,
        min_samples=100
    )

    analyzer = CorrelationAnalyzer(config)

    # 분석 실행
    results = analyzer.analyze(
        data=df,
        columns=pv_cols,
        timestamp_col='timestamp',
        sample_rate_hz=1.0
    )

    print(f"\n유의미한 상관관계: {len(results)}개")

    # 상관행렬
    print("\n상관행렬 (Pearson):")
    corr_matrix = analyzer.get_correlation_matrix(df, pv_cols)
    print(corr_matrix.round(3))

    # 결과 출력
    if results:
        print("\n발견된 관계:")
        df_results = analyzer.get_results_dataframe()
        for _, row in df_results.nlargest(15, 'correlation').iterrows():
            lag_str = f" (lag={row['lag']})" if row['lag'] > 0 else ""
            print(f"  {row['source_param']} → {row['target_param']}: "
                  f"r={row['correlation']:.3f}{lag_str} [{row['method']}]")

    # 고상관 파라미터 쌍
    high_corr = analyzer.find_highly_correlated(df, threshold=0.5, columns=pv_cols)
    if high_corr:
        print(f"\n고상관 파라미터 쌍 (|r| >= 0.5):")
        for src, tgt, corr in high_corr:
            print(f"  {src} ↔ {tgt}: r={corr:.3f}")

    return analyzer, results


def test_causality_analysis(df: pd.DataFrame, pv_cols: list):
    """인과성 분석"""
    print("\n" + "=" * 60)
    print("2. 인과성 분석 (Granger Causality)")
    print("=" * 60)

    from relationship_discovery.config import CausalityConfig
    config = CausalityConfig(
        max_lag=10,
        significance_level=0.05,
        min_samples=500,
        use_granger=True,
        use_transfer_entropy=True
    )

    analyzer = CausalityAnalyzer(config)

    # 분석 실행
    results = analyzer.analyze(
        data=df,
        columns=pv_cols,
        timestamp_col='timestamp',
        sample_rate_hz=1.0
    )

    print(f"\n인과관계 발견: {len(results)}개")

    # 결과 출력
    if results:
        print("\n발견된 인과관계:")
        df_results = analyzer.get_results_dataframe()
        causal = df_results[df_results['is_causal'] == True]

        for _, row in causal.nlargest(15, 'confidence').iterrows():
            print(f"  {row['source_param']} → {row['target_param']}: "
                  f"lag={row['optimal_lag']}, p={row['p_value']:.4f}, "
                  f"conf={row['confidence']:.3f} [{row['method']}]")

    # 양방향 관계
    bidirectional = analyzer.find_bidirectional()
    if bidirectional:
        print(f"\n양방향 인과관계 (피드백 루프): {len(bidirectional)}쌍")
        for r1, r2 in bidirectional:
            print(f"  {r1.source_param} ↔ {r1.target_param}")

    # 인과 그래프
    graph = analyzer.get_causal_graph()
    if graph:
        print("\n인과 그래프:")
        for source, targets in graph.items():
            print(f"  {source} → {', '.join(targets)}")

    return analyzer, results


def test_anomaly_pattern(df: pd.DataFrame, pv_cols: list):
    """이상 발생 패턴 분석"""
    print("\n" + "=" * 60)
    print("3. 이상 발생 전 PV 변화 패턴")
    print("=" * 60)

    # 이상 발생 시점 찾기
    anomaly_starts = []
    prev_anomaly = 0

    for i, row in df.iterrows():
        if row['is_anomaly'] == 1 and prev_anomaly == 0:
            anomaly_starts.append(i)
        prev_anomaly = row['is_anomaly']

    print(f"이상 발생 시작점: {len(anomaly_starts)}개")

    if not anomaly_starts:
        print("이상 발생 시점이 없습니다.")
        return

    # 이상 발생 전 10초 데이터 분석
    lookback = 10

    pre_anomaly_changes = {col: [] for col in pv_cols}

    for start_idx in anomaly_starts:
        if start_idx < lookback:
            continue

        # 이상 발생 직전 데이터
        pre_data = df.iloc[start_idx - lookback:start_idx]
        # 정상 기준 (더 이전 데이터)
        if start_idx >= lookback * 2:
            baseline = df.iloc[start_idx - lookback*2:start_idx - lookback]
        else:
            continue

        for col in pv_cols:
            # 변화율 계산
            pre_mean = pre_data[col].mean()
            base_mean = baseline[col].mean()
            change = (pre_mean - base_mean) / (base_mean + 1e-10) * 100
            pre_anomaly_changes[col].append(change)

    print(f"\n이상 발생 전 {lookback}초 동안의 평균 변화율:")
    for col in pv_cols:
        if pre_anomaly_changes[col]:
            avg_change = np.mean(pre_anomaly_changes[col])
            std_change = np.std(pre_anomaly_changes[col])
            print(f"  {col}: {avg_change:+.2f}% (±{std_change:.2f}%)")


def test_full_pipeline(df: pd.DataFrame, pv_cols: list):
    """전체 파이프라인 테스트"""
    print("\n" + "=" * 60)
    print("4. 통합 파이프라인 실행")
    print("=" * 60)

    # 이벤트 데이터 생성 (이상 발생을 이벤트로)
    events = []
    prev_anomaly = 0

    for i, row in df.iterrows():
        if row['is_anomaly'] == 1 and prev_anomaly == 0:
            events.append({
                'timestamp': row['timestamp'],
                'event_type': 'ANOMALY_START'
            })
        elif row['is_anomaly'] == 0 and prev_anomaly == 1:
            events.append({
                'timestamp': row['timestamp'],
                'event_type': 'ANOMALY_END'
            })
        prev_anomaly = row['is_anomaly']

    # PV 임계값 이벤트 추가
    for col in pv_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        upper = mean_val + 2 * std_val
        lower = mean_val - 2 * std_val

        for i, row in df.iterrows():
            if row[col] > upper:
                events.append({
                    'timestamp': row['timestamp'],
                    'event_type': f'{col.upper()}_HIGH'
                })
            elif row[col] < lower:
                events.append({
                    'timestamp': row['timestamp'],
                    'event_type': f'{col.upper()}_LOW'
                })

    event_df = pd.DataFrame(events).sort_values('timestamp').reset_index(drop=True)
    print(f"생성된 이벤트: {len(event_df)}개")
    print(f"이벤트 유형: {event_df['event_type'].value_counts().to_dict()}")

    # 파이프라인 설정
    config = DiscoveryConfig()
    config.correlation.min_correlation = 0.3
    config.causality.max_lag = 10
    config.pattern.min_support = 0.05
    config.pattern.min_confidence = 0.3

    # 파이프라인 실행
    pipeline = DiscoveryPipeline(config)

    relationships = pipeline.discover_all(
        pv_data=df,
        event_data=event_df if len(event_df) > 10 else None,
        pv_columns=pv_cols,
        timestamp_col='timestamp',
        event_col='event_type',
        sample_rate_hz=1.0
    )

    # 리포트 출력
    report = pipeline.generate_report()
    print(report)

    # 결과 저장
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # HTML 리포트
    html_path = os.path.join(output_dir, 'real_data_relationship_report.html')
    pipeline.export_results(html_path, format='html')
    print(f"\nHTML 리포트: {html_path}")

    # CSV
    csv_path = os.path.join(output_dir, 'real_data_relationships.csv')
    pipeline.export_results(csv_path, format='csv')
    print(f"CSV: {csv_path}")

    return pipeline, relationships


def find_anomaly_causes(df: pd.DataFrame, pv_cols: list):
    """이상의 근본 원인 분석"""
    print("\n" + "=" * 60)
    print("5. 이상 근본 원인 분석")
    print("=" * 60)

    # 이상 시점 vs 정상 시점의 파라미터 분포 비교
    normal_data = df[df['is_anomaly'] == 0]
    anomaly_data = df[df['is_anomaly'] == 1]

    print("\n파라미터별 이상 기여도 분석 (t-test):")

    from scipy import stats

    contributions = []
    for col in pv_cols:
        t_stat, p_value = stats.ttest_ind(
            normal_data[col].dropna(),
            anomaly_data[col].dropna()
        )

        # 효과 크기 (Cohen's d)
        pooled_std = np.sqrt(
            (normal_data[col].std()**2 + anomaly_data[col].std()**2) / 2
        )
        cohens_d = abs(normal_data[col].mean() - anomaly_data[col].mean()) / pooled_std

        contributions.append({
            'parameter': col,
            't_statistic': abs(t_stat),
            'p_value': p_value,
            'cohens_d': cohens_d,
            'normal_mean': normal_data[col].mean(),
            'anomaly_mean': anomaly_data[col].mean(),
            'difference': anomaly_data[col].mean() - normal_data[col].mean()
        })

    contrib_df = pd.DataFrame(contributions).sort_values('cohens_d', ascending=False)

    for _, row in contrib_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        direction = "↑" if row['difference'] > 0 else "↓"
        print(f"  {row['parameter']:12} | d={row['cohens_d']:.3f} | "
              f"정상:{row['normal_mean']:.2f} → 이상:{row['anomaly_mean']:.2f} {direction} {sig}")

    print("\n이상 예측 주요 파라미터 (Cohen's d 기준):")
    top_causes = contrib_df.nlargest(3, 'cohens_d')
    for i, (_, row) in enumerate(top_causes.iterrows(), 1):
        print(f"  {i}. {row['parameter']} (효과크기: {row['cohens_d']:.3f})")


def main():
    """메인 함수"""
    print("=" * 60)
    print("실제 샘플 데이터 관계 발견 테스트")
    print("=" * 60)

    # 데이터 로드
    df = load_real_data()

    # 데이터 특성 분석
    pv_cols = analyze_data_characteristics(df)

    # 상관관계 분석
    corr_analyzer, corr_results = test_correlation_analysis(df, pv_cols)

    # 인과성 분석
    cause_analyzer, cause_results = test_causality_analysis(df, pv_cols)

    # 이상 패턴 분석
    test_anomaly_pattern(df, pv_cols)

    # 근본 원인 분석
    find_anomaly_causes(df, pv_cols)

    # 전체 파이프라인
    pipeline, relationships = test_full_pipeline(df, pv_cols)

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
