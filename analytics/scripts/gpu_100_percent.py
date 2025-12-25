"""
GPU 100% Stress - Multi-Stream Version
=======================================
RTX 4070 GPU를 진짜 100% 찍게 만드는 코드
여러 CUDA 스트림으로 GPU가 쉴 틈 없이 일하게
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
    """CPU 워커"""
    worker_id, duration = args
    end_time = time.time() + duration
    result = 0.0
    while time.time() < end_time:
        A = np.random.rand(250, 250)
        B = np.random.rand(250, 250)
        result += np.trace(np.dot(A, B))
    return result


def run_cpu_background(duration):
    """CPU 백그라운드 스트레스"""
    num_workers = mp.cpu_count()
    args = [(i, duration) for i in range(num_workers)]
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(cpu_worker, args))


def gpu_100_percent_stress(duration_seconds=60):
    """GPU 100% 스트레스 - 멀티 스트림"""

    print("=" * 70)
    print(f"GPU 100% STRESS TEST ({duration_seconds}s)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # 여러 CUDA 스트림 생성 (병렬 실행)
    num_streams = 4
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    # 각 스트림에 할당할 행렬들
    size = 6000  # 각 스트림당 적당한 크기
    matrices_per_stream = []

    for s in range(num_streams):
        mats = [torch.randn(size, size, device=device, dtype=torch.float32) for _ in range(3)]
        matrices_per_stream.append(mats)

    print(f"  Streams: {num_streams}")
    print(f"  Matrix size: {size}x{size}")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("-" * 70)

    iterations = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        # 모든 스트림에서 동시에 행렬 곱셈 실행
        for s_idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                mats = matrices_per_stream[s_idx]
                # 연속 행렬 곱셈
                result = torch.mm(mats[0], mats[1])
                result = torch.mm(result, mats[2])
                result = torch.mm(result, mats[0])
                result = torch.mm(result, mats[1])

        iterations += 1

        if iterations % 20 == 0:
            # 진행 상황 (동기화 최소화)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            # 각 스트림에서 5번 곱셈, 4개 스트림
            ops = iterations * num_streams * 5 * 2 * (size ** 3)
            tflops = ops / elapsed / 1e12
            print(f"  iter={iterations}, {tflops:.2f} TFLOPS, {elapsed:.1f}s")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ops = iterations * num_streams * 5 * 2 * (size ** 3)
    final_tflops = ops / elapsed / 1e12

    print("-" * 70)
    print(f"  COMPLETED: {iterations} iterations")
    print(f"  PERFORMANCE: {final_tflops:.2f} TFLOPS")
    print("=" * 70)

    return final_tflops


def gpu_fp16_max_stress(duration_seconds=60):
    """FP16 Tensor Core 최대 스트레스"""

    print("\n" + "=" * 70)
    print(f"FP16 TENSOR CORE MAX STRESS ({duration_seconds}s)")
    print("=" * 70)

    # FP16으로 더 큰 행렬 가능
    size = 10000

    A = torch.randn(size, size, device=device, dtype=torch.float16)
    B = torch.randn(size, size, device=device, dtype=torch.float16)
    C = torch.randn(size, size, device=device, dtype=torch.float16)

    print(f"  Matrix size: {size}x{size} (FP16)")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("-" * 70)

    iterations = 0
    start = time.perf_counter()

    while time.perf_counter() - start < duration_seconds:
        # 연속 행렬 곱셈 (Tensor Cores 사용)
        result = torch.mm(A, B)
        result = torch.mm(result, C)
        result = torch.mm(result, A)
        result = torch.mm(result, B)
        result = torch.mm(result, C)

        iterations += 1

        if iterations % 10 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ops = iterations * 5 * 2 * (size ** 3)
            tflops = ops / elapsed / 1e12
            print(f"  iter={iterations}, {tflops:.2f} TFLOPS, {elapsed:.1f}s")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ops = iterations * 5 * 2 * (size ** 3)
    final_tflops = ops / elapsed / 1e12

    print("-" * 70)
    print(f"  COMPLETED: {iterations} iterations")
    print(f"  FP16 PERFORMANCE: {final_tflops:.2f} TFLOPS")
    print("=" * 70)

    del A, B, C, result
    torch.cuda.empty_cache()

    return final_tflops


def main():
    print("\n")
    print("*" * 70)
    print("*  CPU + GPU 100% STRESS TEST")
    print("*  Open Task Manager (Ctrl+Shift+Esc) and watch!")
    print("*" * 70)
    print(f"  CPU: {mp.cpu_count()} threads")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("*" * 70)

    duration = 30  # 각 테스트 30초

    # CPU 백그라운드 시작
    print("\n[CPU] Starting 32-thread stress in background...")
    cpu_thread = threading.Thread(target=run_cpu_background, args=(duration * 2 + 20,))
    cpu_thread.start()

    time.sleep(3)  # CPU 먼저 워밍업

    # GPU 테스트 1: 멀티스트림
    tflops1 = gpu_100_percent_stress(duration)

    # GPU 테스트 2: FP16 Tensor Core
    tflops2 = gpu_fp16_max_stress(duration)

    print("\n[CPU] Waiting for completion...")
    cpu_thread.join()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  GPU Multi-Stream FP32: {tflops1:.2f} TFLOPS")
    print(f"  GPU Tensor Core FP16:  {tflops2:.2f} TFLOPS")
    print(f"  CPU: 32 threads completed")
    print("=" * 70)


if __name__ == '__main__':
    mp.freeze_support()
    main()
