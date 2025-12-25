"""
Relationship Discovery Module Test Script
==========================================

암묵적 관계 발견 모듈 테스트

실행:
    python scripts/test_relationship_discovery.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from relationship_discovery import (
    DiscoveryConfig,
    CorrelationAnalyzer,
    CausalityAnalyzer,
    PatternDetector,
    RelationshipStore,
    DiscoveryPipeline
)


def generate_synthetic_pv_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    인과관계가 있는 합성 PV 데이터 생성

    관계 설정:
    - RF_Power → Etch_Rate (lag=2, positive)
    - Pressure → RF_Power (lag=3, negative)
    - Temperature → Etch_Rate (lag=1, positive)
    - Gas_Flow와 Pressure는 상관관계만
    """
    np.random.seed(42)

    timestamps = pd.date_range(
        start='2024-01-01',
        periods=n_samples,
        freq='1s'
    )

    # 기본 노이즈
    noise = np.random.randn(n_samples) * 0.1

    # 독립 변수
    pressure = 100 + 5 * np.sin(np.linspace(0, 10*np.pi, n_samples)) + np.random.randn(n_samples) * 2
    gas_flow = 50 + 0.3 * pressure + np.random.randn(n_samples) * 3  # Pressure와 상관

    # RF_Power는 Pressure의 영향을 받음 (lag=3)
    rf_power = np.zeros(n_samples)
    rf_power[:3] = 500 + np.random.randn(3) * 10
    for i in range(3, n_samples):
        rf_power[i] = 500 - 0.5 * pressure[i-3] + np.random.randn() * 10

    # Temperature는 독립
    temperature = 25 + 3 * np.sin(np.linspace(0, 5*np.pi, n_samples)) + np.random.randn(n_samples) * 1

    # Etch_Rate는 RF_Power(lag=2)와 Temperature(lag=1)의 영향을 받음
    etch_rate = np.zeros(n_samples)
    etch_rate[:2] = 45 + np.random.randn(2) * 2
    for i in range(2, n_samples):
        etch_rate[i] = (
            10 +
            0.08 * rf_power[i-2] +  # RF_Power 영향 (lag=2)
            0.5 * temperature[i-1] +  # Temperature 영향 (lag=1)
            np.random.randn() * 2
        )

    return pd.DataFrame({
        'timestamp': timestamps,
        'RF_Power': rf_power,
        'Pressure': pressure,
        'Gas_Flow': gas_flow,
        'Temperature': temperature,
        'Etch_Rate': etch_rate,
    })


def generate_synthetic_event_data(n_events: int = 500) -> pd.DataFrame:
    """
    패턴이 있는 합성 이벤트 데이터 생성

    패턴:
    - PRESSURE_HIGH → RF_ADJUST → ETCH_RATE_OOS (순차 패턴)
    - PM_START → PM_END (항상 함께)
    - TEMP_HIGH → ALARM_TRIGGER (자주 발생)
    """
    np.random.seed(42)

    events = []
    current_time = datetime(2024, 1, 1)

    event_types = ['NORMAL', 'PRESSURE_HIGH', 'RF_ADJUST', 'ETCH_RATE_OOS',
                   'TEMP_HIGH', 'ALARM_TRIGGER', 'PM_START', 'PM_END']

    for _ in range(n_events):
        current_time += timedelta(seconds=np.random.randint(1, 60))

        # 패턴 1: PRESSURE_HIGH → RF_ADJUST → ETCH_RATE_OOS (20% 확률)
        if np.random.random() < 0.2:
            events.append({'timestamp': current_time, 'event_type': 'PRESSURE_HIGH'})
            current_time += timedelta(seconds=np.random.randint(5, 30))
            events.append({'timestamp': current_time, 'event_type': 'RF_ADJUST'})
            if np.random.random() < 0.7:  # 70% 확률로 OOS 발생
                current_time += timedelta(seconds=np.random.randint(10, 60))
                events.append({'timestamp': current_time, 'event_type': 'ETCH_RATE_OOS'})

        # 패턴 2: PM 이벤트 (10% 확률)
        elif np.random.random() < 0.1:
            events.append({'timestamp': current_time, 'event_type': 'PM_START'})
            current_time += timedelta(minutes=np.random.randint(30, 120))
            events.append({'timestamp': current_time, 'event_type': 'PM_END'})

        # 패턴 3: TEMP_HIGH → ALARM_TRIGGER (15% 확률)
        elif np.random.random() < 0.15:
            events.append({'timestamp': current_time, 'event_type': 'TEMP_HIGH'})
            if np.random.random() < 0.8:  # 80% 확률로 알람
                current_time += timedelta(seconds=np.random.randint(1, 10))
                events.append({'timestamp': current_time, 'event_type': 'ALARM_TRIGGER'})

        # 일반 이벤트
        else:
            events.append({
                'timestamp': current_time,
                'event_type': np.random.choice(['NORMAL', 'PRESSURE_HIGH', 'TEMP_HIGH'])
            })

    return pd.DataFrame(events).sort_values('timestamp').reset_index(drop=True)


def test_correlation_analyzer():
    """상관관계 분석기 테스트"""
    print("\n" + "=" * 60)
    print("1. 상관관계 분석기 테스트")
    print("=" * 60)

    # 데이터 생성
    pv_data = generate_synthetic_pv_data(3000)
    print(f"생성된 PV 데이터: {len(pv_data)} 샘플")

    # 분석기 초기화
    analyzer = CorrelationAnalyzer()

    # 분석 실행
    columns = ['RF_Power', 'Pressure', 'Gas_Flow', 'Temperature', 'Etch_Rate']
    results = analyzer.analyze(
        data=pv_data,
        columns=columns,
        timestamp_col='timestamp',
        sample_rate_hz=1.0
    )

    print(f"\n발견된 유의미한 상관관계: {len(results)}개")

    # 결과 출력
    df = analyzer.get_results_dataframe()
    if len(df) > 0:
        print("\n상위 10개 상관관계:")
        top_results = df.nlargest(10, 'correlation')
        for _, row in top_results.iterrows():
            lag_str = f" (lag={row['lag']})" if row['lag'] > 0 else ""
            print(f"  {row['source_param']} → {row['target_param']}: "
                  f"r={row['correlation']:.3f}{lag_str}, p={row['p_value']:.4f}")

    # 요약 출력
    print("\n요약:")
    summary = analyzer.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return analyzer


def test_causality_analyzer():
    """인과성 분석기 테스트"""
    print("\n" + "=" * 60)
    print("2. 인과성 분석기 테스트")
    print("=" * 60)

    # 데이터 생성
    pv_data = generate_synthetic_pv_data(3000)

    # 분석기 초기화
    analyzer = CausalityAnalyzer()

    # 분석 실행
    columns = ['RF_Power', 'Pressure', 'Temperature', 'Etch_Rate']
    results = analyzer.analyze(
        data=pv_data,
        columns=columns,
        timestamp_col='timestamp',
        sample_rate_hz=1.0
    )

    print(f"\n발견된 인과관계: {len(results)}개")

    # 결과 출력
    df = analyzer.get_results_dataframe()
    if len(df) > 0:
        causal_df = df[df['is_causal'] == True].nlargest(10, 'confidence')
        print("\n상위 인과관계:")
        for _, row in causal_df.iterrows():
            print(f"  {row['source_param']} → {row['target_param']}: "
                  f"lag={row['optimal_lag']}, p={row['p_value']:.4f}, conf={row['confidence']:.3f}")

    # 양방향 관계 확인
    bidirectional = analyzer.find_bidirectional()
    if bidirectional:
        print(f"\n양방향 인과관계 (피드백 루프): {len(bidirectional)}쌍")
        for r1, r2 in bidirectional:
            print(f"  {r1.source_param} ↔ {r1.target_param}")

    # 인과 그래프
    graph = analyzer.get_causal_graph()
    print("\n인과 그래프:")
    for source, targets in graph.items():
        print(f"  {source} → {', '.join(targets)}")

    return analyzer


def test_pattern_detector():
    """패턴 발견기 테스트"""
    print("\n" + "=" * 60)
    print("3. 이벤트 패턴 발견기 테스트")
    print("=" * 60)

    # 데이터 생성
    event_data = generate_synthetic_event_data(500)
    print(f"생성된 이벤트: {len(event_data)}개")
    print(f"이벤트 유형: {event_data['event_type'].unique().tolist()}")

    # 분석기 초기화
    detector = PatternDetector()

    # 패턴 발견
    results = detector.detect(
        events=event_data,
        event_col='event_type',
        timestamp_col='timestamp'
    )

    print(f"\n발견된 패턴: {len(results)}개")

    # 결과 출력
    if results:
        print("\n주요 순차 패턴:")
        sequential = [r for r in results if r.pattern_type == "sequential"]
        for pattern in sorted(sequential, key=lambda x: x.confidence, reverse=True)[:10]:
            print(f"  {pattern}")

        print("\n동시 발생 패턴:")
        co_occur = [r for r in results if r.pattern_type == "co_occurrence"]
        for pattern in sorted(co_occur, key=lambda x: x.lift, reverse=True)[:5]:
            print(f"  {pattern} (lift={pattern.lift:.2f})")

    # 근본 원인 탐색
    print("\n근본 원인 탐색 (ETCH_RATE_OOS):")
    root_causes = detector.find_root_cause_patterns(
        events=event_data,
        target_event='ETCH_RATE_OOS',
        event_col='event_type',
        timestamp_col='timestamp',
        lookback_seconds=120
    )

    for i, pattern in enumerate(root_causes[:5], 1):
        print(f"  {i}. {' → '.join(pattern.pattern[:-1])} → ETCH_RATE_OOS "
              f"(conf={pattern.confidence:.3f})")

    return detector


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("\n" + "=" * 60)
    print("4. 통합 파이프라인 테스트")
    print("=" * 60)

    # 데이터 생성
    pv_data = generate_synthetic_pv_data(5000)
    event_data = generate_synthetic_event_data(500)

    print(f"PV 데이터: {len(pv_data)} 샘플")
    print(f"이벤트 데이터: {len(event_data)} 이벤트")

    # 설정
    config = DiscoveryConfig.for_batch()
    config.ontology.auto_save = False  # DB 저장 비활성화

    # 파이프라인 초기화
    pipeline = DiscoveryPipeline(config)

    # 전체 분석 실행
    print("\n분석 실행 중...")
    relationships = pipeline.discover_all(
        pv_data=pv_data,
        event_data=event_data,
        pv_columns=['RF_Power', 'Pressure', 'Gas_Flow', 'Temperature', 'Etch_Rate'],
        timestamp_col='timestamp',
        event_col='event_type',
        sample_rate_hz=1.0
    )

    # 리포트 생성
    report = pipeline.generate_report()
    print(report)

    # 결과 데이터프레임
    df = pipeline.get_results_dataframe()
    print(f"\n전체 결과 데이터프레임: {len(df)} 행")

    # 관계 유형별 집계
    print("\n관계 유형별:")
    if len(df) > 0:
        for rel_type, count in df['relation_type'].value_counts().items():
            print(f"  {rel_type}: {count}")

    # 결과 내보내기
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # JSON 내보내기
    json_path = os.path.join(output_dir, 'discovered_relationships.json')
    pipeline.export_results(json_path, format='json')
    print(f"\nJSON 내보내기: {json_path}")

    # CSV 내보내기
    csv_path = os.path.join(output_dir, 'discovered_relationships.csv')
    pipeline.export_results(csv_path, format='csv')
    print(f"CSV 내보내기: {csv_path}")

    # HTML 리포트
    html_path = os.path.join(output_dir, 'relationship_report.html')
    pipeline.export_results(html_path, format='html')
    print(f"HTML 리포트: {html_path}")

    return pipeline


def test_specific_parameter_analysis():
    """특정 파라미터 분석 테스트"""
    print("\n" + "=" * 60)
    print("5. 특정 파라미터 영향 분석")
    print("=" * 60)

    pv_data = generate_synthetic_pv_data(5000)

    config = DiscoveryConfig()
    pipeline = DiscoveryPipeline(config)

    # Etch_Rate에 영향을 주는 파라미터 찾기
    print("\nEtch_Rate에 영향을 주는 파라미터 분석...")
    influences = pipeline.discover_for_parameter(
        data=pv_data,
        target_param='Etch_Rate',
        timestamp_col='timestamp',
        sample_rate_hz=1.0
    )

    print(f"\n발견된 영향 관계: {len(influences)}개")
    for rel in influences[:10]:
        print(f"  {rel.source} → Etch_Rate: "
              f"{rel.relation_type}, conf={rel.confidence:.3f}")

    return influences


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("암묵적 관계 발견 모듈 테스트")
    print("=" * 60)

    # 개별 모듈 테스트
    test_correlation_analyzer()
    test_causality_analyzer()
    test_pattern_detector()

    # 통합 테스트
    test_full_pipeline()

    # 특정 파라미터 분석
    test_specific_parameter_analysis()

    print("\n" + "=" * 60)
    print("모든 테스트 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
