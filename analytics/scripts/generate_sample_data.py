"""
FDC 샘플 데이터 생성 스크립트

가상의 PV(Process Variable) 데이터와 설비 이상 발생 시점 정보를 생성합니다.
- 1초 단위 센서 데이터 (온도, 압력, 진동, 전류, 유량)
- 다양한 이상 패턴 포함 (점진적 드리프트, 급격한 스파이크, 노이즈 증가)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_normal_data(n_samples: int, n_sensors: int = 5) -> np.ndarray:
    """정상 상태의 센서 데이터 생성"""
    data = np.zeros((n_samples, n_sensors))

    # 센서별 기본 특성 설정
    sensor_params = [
        {'mean': 25.0, 'std': 1.0, 'trend': 0.0001},   # 온도 (°C)
        {'mean': 1.0, 'std': 0.05, 'trend': 0.0},      # 압력 (bar)
        {'mean': 0.5, 'std': 0.1, 'trend': 0.0},       # 진동 (mm/s)
        {'mean': 10.0, 'std': 0.5, 'trend': 0.0},      # 전류 (A)
        {'mean': 100.0, 'std': 2.0, 'trend': 0.0},     # 유량 (L/min)
    ]

    for i, params in enumerate(sensor_params):
        # 기본 노이즈
        noise = np.random.normal(0, params['std'], n_samples)
        # 트렌드
        trend = np.arange(n_samples) * params['trend']
        # 주기적 패턴 (일간 변동)
        periodic = np.sin(np.arange(n_samples) * 2 * np.pi / 3600) * params['std'] * 0.5

        data[:, i] = params['mean'] + noise + trend + periodic

    return data


def inject_anomalies(data: np.ndarray, anomaly_ratio: float = 0.05) -> tuple:
    """데이터에 이상 패턴 주입"""
    n_samples = len(data)
    labels = np.zeros(n_samples, dtype=int)
    fault_records = []

    # 이상 발생 구간 수 계산
    n_anomaly_periods = int(n_samples * anomaly_ratio / 60)  # 평균 60초 지속

    if n_anomaly_periods == 0:
        n_anomaly_periods = 3

    for _ in range(n_anomaly_periods):
        # 랜덤 시작 위치와 지속 시간
        start_idx = np.random.randint(100, n_samples - 200)
        duration = np.random.randint(30, 120)
        end_idx = min(start_idx + duration, n_samples - 1)

        # 이상 유형 선택
        anomaly_type = np.random.choice([
            'spike', 'drift', 'noise_increase', 'sudden_drop', 'oscillation'
        ])

        # 영향받는 센서 선택
        affected_sensor = np.random.randint(0, data.shape[1])

        if anomaly_type == 'spike':
            # 급격한 스파이크
            spike_magnitude = np.random.uniform(3, 6) * np.std(data[:, affected_sensor])
            data[start_idx:end_idx, affected_sensor] += spike_magnitude

        elif anomaly_type == 'drift':
            # 점진적 드리프트
            drift = np.linspace(0, np.std(data[:, affected_sensor]) * 4, end_idx - start_idx)
            data[start_idx:end_idx, affected_sensor] += drift

        elif anomaly_type == 'noise_increase':
            # 노이즈 증가
            extra_noise = np.random.normal(0, np.std(data[:, affected_sensor]) * 2, end_idx - start_idx)
            data[start_idx:end_idx, affected_sensor] += extra_noise

        elif anomaly_type == 'sudden_drop':
            # 급격한 하락
            drop_magnitude = np.random.uniform(3, 5) * np.std(data[:, affected_sensor])
            data[start_idx:end_idx, affected_sensor] -= drop_magnitude

        elif anomaly_type == 'oscillation':
            # 비정상 진동
            freq = np.random.uniform(0.1, 0.5)
            oscillation = np.sin(np.arange(end_idx - start_idx) * freq) * np.std(data[:, affected_sensor]) * 3
            data[start_idx:end_idx, affected_sensor] += oscillation

        # 라벨 업데이트
        labels[start_idx:end_idx] = 1

        # 이상 기록 저장
        fault_records.append({
            'start_idx': start_idx,
            'end_idx': end_idx,
            'fault_type': anomaly_type,
            'affected_sensor': affected_sensor,
            'duration': duration
        })

    return data, labels, fault_records


def generate_sample_dataset(
    n_samples: int = 86400,  # 24시간 (1초 단위)
    start_time: datetime = None,
    output_dir: str = '../sample_data'
):
    """전체 샘플 데이터셋 생성 및 저장"""

    if start_time is None:
        start_time = datetime(2024, 1, 1, 0, 0, 0)

    # 타임스탬프 생성
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_samples)]

    # 정상 데이터 생성
    print("정상 데이터 생성 중...")
    sensor_data = generate_normal_data(n_samples)

    # 이상 패턴 주입
    print("이상 패턴 주입 중...")
    sensor_data, labels, fault_records = inject_anomalies(sensor_data)

    # PV 데이터 DataFrame 생성
    sensor_columns = ['temperature', 'pressure', 'vibration', 'current', 'flow_rate']
    pv_df = pd.DataFrame(sensor_data, columns=sensor_columns)
    pv_df.insert(0, 'timestamp', timestamps)
    pv_df['is_anomaly'] = labels

    # 이상 발생 시점 정보 DataFrame 생성
    fault_df = pd.DataFrame(fault_records)
    if len(fault_df) > 0:
        fault_df['start_timestamp'] = fault_df['start_idx'].apply(lambda x: timestamps[x])
        fault_df['end_timestamp'] = fault_df['end_idx'].apply(lambda x: timestamps[x])
        fault_df['sensor_name'] = fault_df['affected_sensor'].apply(lambda x: sensor_columns[x])

    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # CSV 저장
    pv_path = os.path.join(output_dir, 'sample_pv_data.csv')
    fault_path = os.path.join(output_dir, 'sample_fault_labels.csv')

    pv_df.to_csv(pv_path, index=False)
    print(f"PV 데이터 저장: {pv_path}")
    print(f"  - 총 샘플 수: {len(pv_df)}")
    print(f"  - 이상 샘플 수: {labels.sum()} ({labels.sum()/len(labels)*100:.2f}%)")

    if len(fault_df) > 0:
        fault_df.to_csv(fault_path, index=False)
        print(f"이상 발생 정보 저장: {fault_path}")
        print(f"  - 이상 이벤트 수: {len(fault_df)}")
        print(f"  - 이상 유형별 분포:")
        for fault_type, count in fault_df['fault_type'].value_counts().items():
            print(f"    * {fault_type}: {count}")

    return pv_df, fault_df


if __name__ == "__main__":
    # 샘플 데이터 생성 (24시간 분량)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'sample_data')

    pv_df, fault_df = generate_sample_dataset(
        n_samples=86400,  # 24시간
        output_dir=output_dir
    )

    print("\n샘플 데이터 생성 완료!")
    print(f"PV 데이터 미리보기:\n{pv_df.head()}")
