"""
GPU vs CPU Relationship Discovery Benchmark
============================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import time
import torch

print("=" * 70)
print("GPU vs CPU Relationship Discovery Benchmark")
print("=" * 70)

# GPU 확인
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU: Not available")
print("=" * 70)


def generate_test_data(n_samples=100000):
    """인과관계가 있는 테스트 데이터 생성"""
    np.random.seed(42)
    t = np.arange(n_samples)
    noise = np.random.randn(n_samples) * 0.1

    # 인과 체인: pressure -> rf_power -> etch_rate
    pressure = np.sin(t / 1000) + noise
    rf_power = np.roll(pressure, 5) * 0.8 + np.random.randn(n_samples) * 0.1
    etch_rate = np.roll(rf_power, 3) * 0.9 + np.random.randn(n_samples) * 0.1

    # 독립 체인: temperature -> flow_rate
    temperature = np.sin(t / 500) + noise
    flow_rate = temperature * 0.5 + np.random.randn(n_samples) * 0.1

    # 독립 변수
    vibration = np.random.randn(n_samples) * 0.5

    return pd.DataFrame({
        'pressure': pressure.astype(np.float32),
        'rf_power': rf_power.astype(np.float32),
        'etch_rate': etch_rate.astype(np.float32),
        'temperature': temperature.astype(np.float32),
        'flow_rate': flow_rate.astype(np.float32),
        'vibration': vibration.astype(np.float32),
    })


def benchmark_cpu(df, columns):
    """CPU 버전 벤치마크"""
    from relationship_discovery import DiscoveryPipeline, DiscoveryConfig

    config = DiscoveryConfig()
    config.correlation.min_correlation = 0.3
    config.correlation.max_lag = 20

    pipeline = DiscoveryPipeline(config)

    start = time.perf_counter()
    relationships = pipeline.discover_all(
        pv_data=df,
        pv_columns=columns,
        timestamp_col=None,
        sample_rate_hz=1.0
    )
    elapsed = time.perf_counter() - start

    return len(relationships), elapsed


def benchmark_gpu(df, columns):
    """GPU 버전 벤치마크"""
    device = torch.device('cuda')
    data = torch.from_numpy(df[columns].values).to(device)
    n_samples = len(df)
    max_lag = 20
    min_corr = 0.3

    start = time.perf_counter()

    # 1. GPU 상관행렬
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True) + 1e-8
    normalized = (data - mean) / std
    corr_matrix = torch.mm(normalized.T, normalized) / (n_samples - 1)

    # 2. GPU Cross-Correlation
    relationships = []
    n_cols = len(columns)

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            x = data[:, i]
            y = data[:, j]

            # 즉시 상관
            instant_corr = corr_matrix[i, j].item()
            if abs(instant_corr) >= min_corr:
                relationships.append({
                    'source': columns[i],
                    'target': columns[j],
                    'type': 'correlates',
                    'strength': instant_corr,
                    'lag': 0
                })

            # FFT Cross-correlation
            x_c = x - x.mean()
            y_c = y - y.mean()

            n = x.shape[0]
            pad = 2 * n
            x_pad = torch.nn.functional.pad(x_c, (0, pad - n))
            y_pad = torch.nn.functional.pad(y_c, (0, pad - n))

            X = torch.fft.fft(x_pad)
            Y = torch.fft.fft(y_pad)
            cross = torch.fft.ifft(X * torch.conj(Y)).real

            norm = torch.sqrt((x_c**2).sum() * (y_c**2).sum()) + 1e-8
            cross = cross / norm

            result = torch.cat([cross[-max_lag:], cross[:max_lag+1]])
            best_idx = torch.argmax(torch.abs(result))
            best_lag = best_idx.item() - max_lag
            best_corr = result[best_idx].item()

            if abs(best_corr) >= min_corr and best_lag != 0:
                if best_lag > 0:
                    src, tgt = columns[i], columns[j]
                else:
                    src, tgt = columns[j], columns[i]
                    best_lag = -best_lag

                relationships.append({
                    'source': src,
                    'target': tgt,
                    'type': 'precedes',
                    'strength': best_corr,
                    'lag': best_lag
                })

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return len(relationships), elapsed, relationships


# 메인 벤치마크
print("\n[1] Generating test data...")
for n_samples in [10000, 50000, 100000, 500000]:
    df = generate_test_data(n_samples)
    columns = list(df.columns)

    print(f"\n{'='*60}")
    print(f"Data size: {n_samples:,} samples x {len(columns)} parameters")
    print(f"{'='*60}")

    # CPU 벤치마크 (작은 데이터만)
    if n_samples <= 100000:
        try:
            n_rels_cpu, time_cpu = benchmark_cpu(df, columns)
            print(f"CPU: {n_rels_cpu} relationships in {time_cpu:.2f}s")
        except Exception as e:
            print(f"CPU: Error - {e}")
            time_cpu = float('inf')
    else:
        print("CPU: Skipped (too large)")
        time_cpu = float('inf')

    # GPU 벤치마크
    n_rels_gpu, time_gpu, rels = benchmark_gpu(df, columns)
    print(f"GPU: {n_rels_gpu} relationships in {time_gpu*1000:.2f}ms")

    if time_cpu != float('inf'):
        speedup = time_cpu / time_gpu
        print(f"Speedup: {speedup:.1f}x")

# 발견된 관계 출력
print("\n" + "=" * 60)
print("Discovered Relationships (GPU)")
print("=" * 60)
for rel in rels:
    lag_str = f" (lag={rel['lag']})" if rel['lag'] > 0 else ""
    print(f"  {rel['source']} -> {rel['target']}: {rel['type']}, "
          f"strength={rel['strength']:.3f}{lag_str}")

print("\n" + "=" * 60)
print("Expected relationships:")
print("  pressure -> rf_power (lag ~5)")
print("  rf_power -> etch_rate (lag ~3)")
print("  temperature -> flow_rate (lag ~0)")
print("=" * 60)
