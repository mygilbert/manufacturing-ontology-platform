"""
CPU + GPU Combined Stress Test
==============================
i9-14900HX (32 threads) + RTX 4070 Laptop (8GB) 동시 풀로드
"""

import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import time
import sys
import os

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_system_info():
    """시스템 정보 출력"""
    print("=" * 70)
    print("CPU + GPU Combined Stress Test")
    print("=" * 70)
    print(f"CPU Threads: {mp.cpu_count()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA Cores: {torch.cuda.get_device_properties(0).multi_processor_count} SMs")
    print("=" * 70)


# ============== GPU Stress Functions ==============

def gpu_matrix_multiply(duration_seconds=30):
    """GPU 행렬 곱셈 스트레스"""
    size = 8192  # 큰 행렬

    print(f"\n[GPU] Matrix Multiplication Stress ({size}x{size})")
    print("-" * 50)

    # GPU 메모리에 큰 행렬 할당
    A = torch.randn(size, size, device=device, dtype=torch.float32)
    B = torch.randn(size, size, device=device, dtype=torch.float32)

    iterations = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        C = torch.mm(A, B)
        torch.cuda.synchronize()  # GPU 작업 완료 대기
        iterations += 1

        if iterations % 10 == 0:
            elapsed = time.perf_counter() - start
            tflops = iterations * 2 * (size ** 3) / elapsed / 1e12
            print(f"  [GPU] {iterations} iterations, {tflops:.2f} TFLOPS")

    elapsed = time.perf_counter() - start
    total_tflops = iterations * 2 * (size ** 3) / elapsed / 1e12

    print(f"  [GPU] Completed: {iterations} iterations in {elapsed:.2f}s")
    print(f"  [GPU] Performance: {total_tflops:.2f} TFLOPS")

    del A, B, C
    torch.cuda.empty_cache()

    return total_tflops


def gpu_neural_network_stress(duration_seconds=30):
    """GPU 신경망 학습 스트레스"""

    print(f"\n[GPU] Neural Network Training Stress")
    print("-" * 50)

    # 큰 ResNet 스타일 모델
    class HeavyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(1024, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 1024),
                nn.ReLU(),
                nn.Linear(1024, 10),
            )

        def forward(self, x):
            return self.layers(x)

    model = HeavyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_size = 2048
    iterations = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        # 랜덤 배치 생성
        x = torch.randn(batch_size, 1024, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        # Forward + Backward
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        iterations += 1

        if iterations % 20 == 0:
            elapsed = time.perf_counter() - start
            throughput = iterations * batch_size / elapsed
            print(f"  [GPU] {iterations} batches, {throughput:.0f} samples/s, loss: {loss.item():.4f}")

    elapsed = time.perf_counter() - start
    total_samples = iterations * batch_size

    print(f"  [GPU] Completed: {total_samples:,} samples in {elapsed:.2f}s")
    print(f"  [GPU] Throughput: {total_samples/elapsed:,.0f} samples/s")

    del model, optimizer
    torch.cuda.empty_cache()

    return total_samples / elapsed


def gpu_convolution_stress(duration_seconds=30):
    """GPU CNN 스트레스"""

    print(f"\n[GPU] CNN Convolution Stress")
    print("-" * 50)

    # 무거운 CNN
    model = nn.Sequential(
        nn.Conv2d(3, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 512, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 256, 3, padding=1),
        nn.ReLU(),
    ).to(device)

    batch_size = 32
    img_size = 256
    iterations = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        output = model(x)
        torch.cuda.synchronize()
        iterations += 1

        if iterations % 20 == 0:
            elapsed = time.perf_counter() - start
            fps = iterations * batch_size / elapsed
            print(f"  [GPU] {iterations} batches, {fps:.0f} images/s")

    elapsed = time.perf_counter() - start
    total_images = iterations * batch_size

    print(f"  [GPU] Completed: {total_images:,} images in {elapsed:.2f}s")
    print(f"  [GPU] Throughput: {total_images/elapsed:,.0f} images/s")

    del model
    torch.cuda.empty_cache()

    return total_images / elapsed


# ============== CPU Stress Functions ==============

def cpu_heavy_worker(args):
    """CPU 헤비 워커"""
    worker_id, iterations = args
    result = 0.0

    for _ in range(iterations):
        # 복잡한 수학 연산
        x = np.random.rand(5000)
        result += np.sum(np.sin(x) * np.cos(x) * np.exp(-x) * np.sqrt(x + 1))

        # 행렬 연산
        A = np.random.rand(300, 300)
        B = np.random.rand(300, 300)
        C = np.dot(A, B)
        result += np.trace(C)

    return result


def cpu_stress_background(duration_seconds, stop_event):
    """백그라운드 CPU 스트레스 (threading)"""
    num_workers = mp.cpu_count()
    iterations_per_batch = 50
    total_batches = 0

    start = time.perf_counter()

    while not stop_event.is_set() and (time.perf_counter() - start) < duration_seconds:
        args = [(i, iterations_per_batch) for i in range(num_workers)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(cpu_heavy_worker, args))

        total_batches += 1

        if total_batches % 2 == 0:
            elapsed = time.perf_counter() - start
            print(f"  [CPU] {total_batches} batches completed, {elapsed:.1f}s elapsed")

    elapsed = time.perf_counter() - start
    print(f"  [CPU] Total: {total_batches} batches in {elapsed:.2f}s")

    return total_batches


# ============== Combined Test ==============

def run_combined_stress_test(duration_seconds=60):
    """CPU와 GPU를 동시에 스트레스"""

    print("\n" + "=" * 70)
    print(f"COMBINED CPU + GPU STRESS TEST ({duration_seconds}s)")
    print("=" * 70)
    print("Open Task Manager and GPU-Z to monitor!")
    print("CPU should hit ~100%, GPU should hit ~100%")
    print("=" * 70)

    stop_event = threading.Event()
    cpu_results = [None]
    gpu_results = {'matrix': None, 'nn': None, 'cnn': None}

    def cpu_thread_func():
        cpu_results[0] = cpu_stress_background(duration_seconds, stop_event)

    def gpu_thread_func():
        # GPU 테스트 순차 실행
        each_duration = duration_seconds // 3
        gpu_results['matrix'] = gpu_matrix_multiply(each_duration)
        gpu_results['nn'] = gpu_neural_network_stress(each_duration)
        gpu_results['cnn'] = gpu_convolution_stress(each_duration)

    # 동시 시작
    start = time.perf_counter()

    cpu_thread = threading.Thread(target=cpu_thread_func)
    gpu_thread = threading.Thread(target=gpu_thread_func)

    cpu_thread.start()
    gpu_thread.start()

    # GPU 먼저 끝나면 CPU도 정리
    gpu_thread.join()
    stop_event.set()
    cpu_thread.join()

    total_elapsed = time.perf_counter() - start

    # 결과 출력
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Time: {total_elapsed:.2f}s")
    print("-" * 70)
    print(f"[CPU] Batches completed: {cpu_results[0]}")
    print(f"[GPU] Matrix TFLOPS: {gpu_results['matrix']:.2f}")
    print(f"[GPU] NN samples/s: {gpu_results['nn']:,.0f}")
    print(f"[GPU] CNN images/s: {gpu_results['cnn']:,.0f}")
    print("=" * 70)

    return total_elapsed


def quick_gpu_benchmark():
    """빠른 GPU 벤치마크"""
    print("\n" + "=" * 70)
    print("Quick GPU Benchmark")
    print("=" * 70)

    # FP32 성능
    size = 8192
    A = torch.randn(size, size, device=device, dtype=torch.float32)
    B = torch.randn(size, size, device=device, dtype=torch.float32)

    # 워밍업
    for _ in range(3):
        C = torch.mm(A, B)
    torch.cuda.synchronize()

    # 측정
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops_fp32 = iterations * 2 * (size ** 3) / elapsed / 1e12
    print(f"FP32 Performance: {tflops_fp32:.2f} TFLOPS")

    # FP16 성능 (Tensor Cores)
    A_fp16 = A.half()
    B_fp16 = B.half()

    # 워밍업
    for _ in range(3):
        C = torch.mm(A_fp16, B_fp16)
    torch.cuda.synchronize()

    # 측정
    start = time.perf_counter()
    for _ in range(iterations):
        C = torch.mm(A_fp16, B_fp16)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops_fp16 = iterations * 2 * (size ** 3) / elapsed / 1e12
    print(f"FP16 Performance: {tflops_fp16:.2f} TFLOPS (Tensor Cores)")

    # 메모리 대역폭
    size_gb = 2  # 2GB 복사
    elements = int(size_gb * 1e9 / 4)  # float32
    src = torch.randn(elements, device=device, dtype=torch.float32)

    start = time.perf_counter()
    for _ in range(10):
        dst = src.clone()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bandwidth = 10 * 2 * size_gb / elapsed  # 읽기 + 쓰기
    print(f"Memory Bandwidth: {bandwidth:.1f} GB/s")

    del A, B, C, A_fp16, B_fp16, src, dst
    torch.cuda.empty_cache()

    return tflops_fp32, tflops_fp16, bandwidth


def main():
    print_system_info()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    # 빠른 GPU 벤치마크
    fp32, fp16, bw = quick_gpu_benchmark()

    # 동시 스트레스 테스트 (60초)
    run_combined_stress_test(60)

    print("\nTest completed! Check Task Manager for CPU/GPU utilization.")


if __name__ == '__main__':
    mp.freeze_support()
    main()
