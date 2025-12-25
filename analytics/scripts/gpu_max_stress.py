"""
GPU Maximum Stress Test + CPU
=============================
RTX 4070 Laptop GPU 100% + CPU 100% 동시 달성
"""

import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading
import time

device = torch.device('cuda')


def cpu_worker(args):
    """CPU 워커 - 무한 연산"""
    worker_id, duration = args
    end_time = time.time() + duration
    result = 0.0

    while time.time() < end_time:
        x = np.random.rand(3000)
        result += np.sum(np.sin(x) * np.cos(x) * np.exp(-x))
        A = np.random.rand(200, 200)
        B = np.random.rand(200, 200)
        result += np.trace(np.dot(A, B))

    return result


def run_cpu_stress(duration):
    """CPU 멀티프로세스 스트레스"""
    num_workers = mp.cpu_count()
    args = [(i, duration) for i in range(num_workers)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(cpu_worker, args))

    return sum(results)


def gpu_continuous_stress(duration_seconds=60):
    """GPU 연속 스트레스 - 쉬지 않고 계속 돌림"""

    print(f"\n[GPU] Continuous Maximum Stress ({duration_seconds}s)")
    print("-" * 50)

    # VRAM 최대한 사용
    size = 10000  # 10000x10000

    # 여러 개의 큰 행렬 미리 할당
    matrices = []
    for i in range(6):
        matrices.append(torch.randn(size, size, device=device, dtype=torch.float32))

    print(f"  Allocated {len(matrices)} matrices of {size}x{size}")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    iterations = 0
    start = time.perf_counter()

    # 결과 저장용 (메모리 재사용)
    result = torch.zeros(size, size, device=device, dtype=torch.float32)

    while time.perf_counter() - start < duration_seconds:
        # 연속 행렬 곱셈 체인 (GPU 쉬지 않게)
        for i in range(len(matrices) - 1):
            torch.mm(matrices[i], matrices[i+1], out=result)

        iterations += 1

        if iterations % 5 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ops_per_iter = (len(matrices) - 1) * 2 * (size ** 3)
            tflops = iterations * ops_per_iter / elapsed / 1e12
            print(f"  iter={iterations}, {tflops:.2f} TFLOPS, elapsed={elapsed:.1f}s")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ops_per_iter = (len(matrices) - 1) * 2 * (size ** 3)
    total_tflops = iterations * ops_per_iter / elapsed / 1e12

    print(f"  Completed: {iterations} iterations in {elapsed:.2f}s")
    print(f"  Performance: {total_tflops:.2f} TFLOPS")

    del matrices, result
    torch.cuda.empty_cache()

    return total_tflops


def gpu_mixed_stress(duration_seconds=60):
    """GPU 혼합 스트레스 - 행렬 + 컨볼루션 + 어텐션"""

    print(f"\n[GPU] Mixed Workload Stress ({duration_seconds}s)")
    print("-" * 50)

    # 1. 큰 행렬들 (FP16 for Tensor Cores)
    mat_size = 8192
    A = torch.randn(mat_size, mat_size, device=device, dtype=torch.float16)
    B = torch.randn(mat_size, mat_size, device=device, dtype=torch.float16)

    # 2. CNN 모델
    cnn = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.ReLU(),
    ).to(device).half()

    # 3. Transformer 어텐션 파라미터
    batch, seq_len, dim = 64, 512, 512

    iterations = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        # 행렬 곱
        C = torch.mm(A, B)

        # CNN forward
        img = torch.randn(32, 3, 224, 224, device=device, dtype=torch.float16)
        out = cnn(img)

        # Self-attention 연산
        Q = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float16)
        K = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float16)
        V = torch.randn(batch, seq_len, dim, device=device, dtype=torch.float16)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / (dim ** 0.5), dim=-1)
        out_attn = torch.bmm(attn, V)

        iterations += 1

        if iterations % 10 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f"  iter={iterations}, elapsed={elapsed:.1f}s")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  Completed: {iterations} iterations in {elapsed:.2f}s")

    del A, B, cnn
    torch.cuda.empty_cache()

    return iterations


def main():
    print("=" * 70)
    print("GPU MAX STRESS + CPU STRESS")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CPU Threads: {mp.cpu_count()}")
    print("=" * 70)
    print(">>> Watch Task Manager - GPU and CPU should hit 100%! <<<")
    print("=" * 70)

    duration = 45  # 각 테스트 45초

    # CPU 백그라운드 스레드
    cpu_result = [None]

    def cpu_thread():
        cpu_result[0] = run_cpu_stress(duration * 2 + 10)

    print("\n[CPU] Starting stress in background (32 threads)...")
    cpu_t = threading.Thread(target=cpu_thread)
    cpu_t.start()

    time.sleep(2)  # CPU 먼저 시작

    # GPU 테스트
    tflops1 = gpu_continuous_stress(duration)
    tflops2 = gpu_mixed_stress(duration)

    # CPU 대기
    print("\n[CPU] Waiting for completion...")
    cpu_t.join()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"[GPU] Continuous Matrix: {tflops1:.2f} TFLOPS")
    print(f"[GPU] Mixed Workload: {tflops2} iterations")
    print(f"[CPU] Completed: 32 threads x {duration*2}s")
    print("=" * 70)


if __name__ == '__main__':
    mp.freeze_support()
    main()
