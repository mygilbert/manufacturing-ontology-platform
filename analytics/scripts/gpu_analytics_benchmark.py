"""
GPU Accelerated Analytics Benchmark
====================================
분석 모듈 GPU 가속 테스트 - CPU vs GPU 성능 비교
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# 1. 상관분석 GPU 가속
# ============================================================

def correlation_cpu(data: np.ndarray) -> np.ndarray:
    """CPU 상관행렬 계산 (NumPy)"""
    return np.corrcoef(data.T)


def correlation_gpu(data: torch.Tensor) -> torch.Tensor:
    """GPU 상관행렬 계산 (PyTorch)"""
    # 표준화
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    normalized = (data - mean) / (std + 1e-8)

    # 상관행렬 = (X^T @ X) / (n-1)
    n = data.shape[0]
    corr = torch.mm(normalized.T, normalized) / (n - 1)
    return corr


def benchmark_correlation(n_samples=100000, n_features=100, n_runs=5):
    """상관분석 벤치마크"""
    print("\n" + "=" * 60)
    print(f"1. Correlation Matrix ({n_samples:,} x {n_features})")
    print("=" * 60)

    # 데이터 생성
    data_np = np.random.randn(n_samples, n_features).astype(np.float32)
    data_gpu = torch.from_numpy(data_np).to(device)

    # CPU 벤치마크
    cpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result_cpu = correlation_cpu(data_np)
        cpu_times.append(time.perf_counter() - start)
    cpu_avg = np.mean(cpu_times)

    # GPU 벤치마크
    torch.cuda.synchronize()
    gpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result_gpu = correlation_gpu(data_gpu)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)
    gpu_avg = np.mean(gpu_times)

    speedup = cpu_avg / gpu_avg

    print(f"  CPU (NumPy):  {cpu_avg*1000:.2f} ms")
    print(f"  GPU (PyTorch): {gpu_avg*1000:.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")

    return speedup


# ============================================================
# 2. Cross-Correlation GPU 가속
# ============================================================

def cross_correlation_cpu(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """CPU Cross-correlation"""
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        correlations.append(corr)
    return np.array(correlations)


def cross_correlation_gpu(x: torch.Tensor, y: torch.Tensor, max_lag: int) -> torch.Tensor:
    """GPU Cross-correlation using FFT"""
    n = x.shape[0]

    # FFT 기반 cross-correlation
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    # Zero-padding
    pad_size = n + max_lag
    x_padded = torch.nn.functional.pad(x_centered, (0, pad_size - n))
    y_padded = torch.nn.functional.pad(y_centered, (0, pad_size - n))

    # FFT cross-correlation
    X = torch.fft.fft(x_padded)
    Y = torch.fft.fft(y_padded)
    cross = torch.fft.ifft(X * torch.conj(Y)).real

    # 정규화
    norm = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))
    cross = cross / norm

    # lag 범위 추출
    result = torch.cat([cross[-max_lag:], cross[:max_lag+1]])

    return result


def benchmark_cross_correlation(n_samples=100000, max_lag=100, n_pairs=50, n_runs=3):
    """Cross-correlation 벤치마크"""
    print("\n" + "=" * 60)
    print(f"2. Cross-Correlation ({n_samples:,} samples, {n_pairs} pairs, lag={max_lag})")
    print("=" * 60)

    # 데이터 생성
    data_np = np.random.randn(n_pairs, n_samples).astype(np.float32)
    data_gpu = torch.from_numpy(data_np).to(device)

    # CPU 벤치마크
    cpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for i in range(n_pairs - 1):
            result = cross_correlation_cpu(data_np[i], data_np[i+1], max_lag)
        cpu_times.append(time.perf_counter() - start)
    cpu_avg = np.mean(cpu_times)

    # GPU 벤치마크
    torch.cuda.synchronize()
    gpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        for i in range(n_pairs - 1):
            result = cross_correlation_gpu(data_gpu[i], data_gpu[i+1], max_lag)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)
    gpu_avg = np.mean(gpu_times)

    speedup = cpu_avg / gpu_avg

    print(f"  CPU (NumPy):  {cpu_avg*1000:.2f} ms")
    print(f"  GPU (PyTorch): {gpu_avg*1000:.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")

    return speedup


# ============================================================
# 3. Autoencoder 이상탐지 GPU 가속
# ============================================================

class Autoencoder(nn.Module):
    """Autoencoder for Anomaly Detection"""
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for dim in reversed(hidden_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def benchmark_autoencoder(n_samples=50000, n_features=20, epochs=10, batch_size=1024):
    """Autoencoder 학습 벤치마크"""
    print("\n" + "=" * 60)
    print(f"3. Autoencoder Training ({n_samples:,} x {n_features}, {epochs} epochs)")
    print("=" * 60)

    # 데이터 생성
    data_np = np.random.randn(n_samples, n_features).astype(np.float32)
    data_tensor = torch.from_numpy(data_np)

    # CPU 학습
    print("  Training on CPU...")
    model_cpu = Autoencoder(n_features)
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    start = time.perf_counter()
    model_cpu.train()
    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            batch = data_tensor[i:i+batch_size]
            optimizer_cpu.zero_grad()
            output = model_cpu(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer_cpu.step()
    cpu_time = time.perf_counter() - start

    # GPU 학습
    print("  Training on GPU...")
    model_gpu = Autoencoder(n_features).to(device)
    optimizer_gpu = torch.optim.Adam(model_gpu.parameters(), lr=0.001)
    data_gpu = data_tensor.to(device)

    torch.cuda.synchronize()
    start = time.perf_counter()
    model_gpu.train()
    for epoch in range(epochs):
        for i in range(0, n_samples, batch_size):
            batch = data_gpu[i:i+batch_size]
            optimizer_gpu.zero_grad()
            output = model_gpu(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer_gpu.step()
    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    speedup = cpu_time / gpu_time

    print(f"  CPU: {cpu_time:.2f} s")
    print(f"  GPU: {gpu_time:.2f} s")
    print(f"  Speedup: {speedup:.1f}x")

    return speedup


# ============================================================
# 4. 시계열 이동통계 GPU 가속
# ============================================================

def rolling_stats_cpu(data: np.ndarray, window: int):
    """CPU 이동평균/표준편차"""
    n = len(data)
    means = np.zeros(n - window + 1)
    stds = np.zeros(n - window + 1)

    for i in range(n - window + 1):
        window_data = data[i:i+window]
        means[i] = np.mean(window_data)
        stds[i] = np.std(window_data)

    return means, stds


def rolling_stats_gpu(data: torch.Tensor, window: int):
    """GPU 이동평균/표준편차 (Convolution 기반)"""
    n = data.shape[0]

    # 1D convolution으로 이동평균
    kernel = torch.ones(1, 1, window, device=device) / window
    data_reshaped = data.view(1, 1, -1)

    means = torch.nn.functional.conv1d(data_reshaped, kernel).squeeze()

    # 이동분산 계산
    data_sq = data ** 2
    data_sq_reshaped = data_sq.view(1, 1, -1)
    means_sq = torch.nn.functional.conv1d(data_sq_reshaped, kernel).squeeze()
    variances = means_sq - means ** 2
    stds = torch.sqrt(torch.clamp(variances, min=0))

    return means, stds


def benchmark_rolling_stats(n_samples=1000000, window=100, n_runs=5):
    """이동통계 벤치마크"""
    print("\n" + "=" * 60)
    print(f"4. Rolling Statistics ({n_samples:,} samples, window={window})")
    print("=" * 60)

    # 데이터 생성
    data_np = np.random.randn(n_samples).astype(np.float32)
    data_gpu = torch.from_numpy(data_np).to(device)

    # CPU 벤치마크
    cpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        means, stds = rolling_stats_cpu(data_np, window)
        cpu_times.append(time.perf_counter() - start)
    cpu_avg = np.mean(cpu_times)

    # GPU 벤치마크
    torch.cuda.synchronize()
    gpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        means, stds = rolling_stats_gpu(data_gpu, window)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)
    gpu_avg = np.mean(gpu_times)

    speedup = cpu_avg / gpu_avg

    print(f"  CPU (NumPy):  {cpu_avg*1000:.2f} ms")
    print(f"  GPU (PyTorch): {gpu_avg*1000:.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")

    return speedup


# ============================================================
# 5. 대규모 행렬 SVD (차원 축소)
# ============================================================

def benchmark_svd(n_samples=10000, n_features=500, n_components=50, n_runs=3):
    """SVD 벤치마크"""
    print("\n" + "=" * 60)
    print(f"5. SVD Decomposition ({n_samples} x {n_features} -> {n_components})")
    print("=" * 60)

    # 데이터 생성
    data_np = np.random.randn(n_samples, n_features).astype(np.float32)
    data_gpu = torch.from_numpy(data_np).to(device)

    # CPU 벤치마크
    cpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        U, S, Vt = np.linalg.svd(data_np, full_matrices=False)
        cpu_times.append(time.perf_counter() - start)
    cpu_avg = np.mean(cpu_times)

    # GPU 벤치마크
    torch.cuda.synchronize()
    gpu_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        U, S, Vh = torch.linalg.svd(data_gpu, full_matrices=False)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)
    gpu_avg = np.mean(gpu_times)

    speedup = cpu_avg / gpu_avg

    print(f"  CPU (NumPy):  {cpu_avg*1000:.2f} ms")
    print(f"  GPU (PyTorch): {gpu_avg*1000:.2f} ms")
    print(f"  Speedup: {speedup:.1f}x")

    return speedup


# ============================================================
# 6. 실제 분석 시나리오: FDC 이상탐지 파이프라인
# ============================================================

def benchmark_fdc_pipeline(n_samples=100000, n_params=50):
    """FDC 이상탐지 파이프라인 벤치마크"""
    print("\n" + "=" * 60)
    print(f"6. FDC Anomaly Detection Pipeline ({n_samples:,} x {n_params})")
    print("=" * 60)

    # 샘플 FDC 데이터 생성
    np.random.seed(42)
    data = np.random.randn(n_samples, n_params).astype(np.float32)
    # 일부 이상치 추가
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data[anomaly_idx] += np.random.randn(len(anomaly_idx), n_params) * 3

    print("  Pipeline: Normalize -> Correlation -> Autoencoder -> Anomaly Score")

    # === CPU Pipeline ===
    print("\n  [CPU Pipeline]")
    start = time.perf_counter()

    # 1. 정규화
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)

    # 2. 상관행렬
    corr_matrix = np.corrcoef(data_norm.T)

    # 3. Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    anomaly_scores_cpu = iso_forest.fit_predict(data_norm)

    cpu_time = time.perf_counter() - start
    print(f"    Time: {cpu_time:.2f}s")
    print(f"    Anomalies detected: {np.sum(anomaly_scores_cpu == -1)}")

    # === GPU Pipeline ===
    print("\n  [GPU Pipeline]")
    data_gpu = torch.from_numpy(data).to(device)

    torch.cuda.synchronize()
    start = time.perf_counter()

    # 1. 정규화 (GPU)
    mean = data_gpu.mean(dim=0, keepdim=True)
    std = data_gpu.std(dim=0, keepdim=True)
    data_norm_gpu = (data_gpu - mean) / (std + 1e-8)

    # 2. 상관행렬 (GPU)
    corr_matrix_gpu = torch.mm(data_norm_gpu.T, data_norm_gpu) / (n_samples - 1)

    # 3. Autoencoder 이상탐지 (GPU)
    model = Autoencoder(n_params, hidden_dims=[32, 16, 8]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='none')

    # 빠른 학습
    batch_size = 2048
    for epoch in range(5):
        for i in range(0, n_samples, batch_size):
            batch = data_norm_gpu[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch).mean()
            loss.backward()
            optimizer.step()

    # 이상 점수 계산
    model.eval()
    with torch.no_grad():
        reconstructed = model(data_norm_gpu)
        anomaly_scores_gpu = torch.mean((data_norm_gpu - reconstructed) ** 2, dim=1)
        threshold = torch.quantile(anomaly_scores_gpu, 0.95)
        anomalies_gpu = anomaly_scores_gpu > threshold

    torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    print(f"    Time: {gpu_time:.2f}s")
    print(f"    Anomalies detected: {anomalies_gpu.sum().item()}")

    speedup = cpu_time / gpu_time
    print(f"\n  Speedup: {speedup:.1f}x")

    return speedup


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("GPU ACCELERATED ANALYTICS BENCHMARK")
    print("=" * 70)
    print(f"CPU: Intel i9-14900HX")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    results = {}

    # 벤치마크 실행
    results['correlation'] = benchmark_correlation()
    results['cross_correlation'] = benchmark_cross_correlation()
    results['autoencoder'] = benchmark_autoencoder()
    results['rolling_stats'] = benchmark_rolling_stats()
    results['svd'] = benchmark_svd()
    results['fdc_pipeline'] = benchmark_fdc_pipeline()

    # 결과 요약
    print("\n" + "=" * 70)
    print("SUMMARY: GPU Speedup vs CPU")
    print("=" * 70)
    for name, speedup in results.items():
        bar = "█" * int(speedup) + "░" * (20 - int(speedup))
        print(f"  {name:<20} {bar} {speedup:>6.1f}x")
    print("=" * 70)
    print(f"  Average Speedup: {np.mean(list(results.values())):.1f}x")
    print("=" * 70)


if __name__ == '__main__':
    main()
