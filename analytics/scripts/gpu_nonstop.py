"""
GPU Non-Stop 100% Stress
========================
동기화 없이 GPU를 쉬지 않고 계속 돌림
"""

import torch
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading
import time

device = torch.device('cuda')


def cpu_worker(duration):
    """CPU 워커"""
    end_time = time.time() + duration
    while time.time() < end_time:
        A = np.random.rand(300, 300)
        B = np.random.rand(300, 300)
        np.dot(A, B)


def gpu_nonstop_stress(duration_seconds=60):
    """GPU 논스톱 스트레스"""

    print("=" * 70)
    print("GPU NON-STOP 100% STRESS")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Duration: {duration_seconds}s")
    print("=" * 70)
    print(">>> WATCH TASK MANAGER GPU TAB! <<<")
    print("=" * 70)

    # 더 작은 행렬을 아주 빠르게 반복 (GPU 점유율 높임)
    size = 4096

    # 행렬들 미리 할당
    A = torch.randn(size, size, device=device, dtype=torch.float16)
    B = torch.randn(size, size, device=device, dtype=torch.float16)
    C = torch.randn(size, size, device=device, dtype=torch.float16)
    D = torch.randn(size, size, device=device, dtype=torch.float16)

    # 결과 버퍼
    R1 = torch.empty(size, size, device=device, dtype=torch.float16)
    R2 = torch.empty(size, size, device=device, dtype=torch.float16)

    print(f"  Matrix size: {size}x{size}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print("-" * 70)

    iterations = 0
    last_print = time.perf_counter()
    start = time.perf_counter()

    # 메인 루프 - 동기화 최소화
    while time.perf_counter() - start < duration_seconds:
        # 여러 연산을 한꺼번에 큐에 넣음 (비동기)
        for _ in range(10):  # 10개씩 배치
            torch.mm(A, B, out=R1)
            torch.mm(R1, C, out=R2)
            torch.mm(R2, D, out=R1)
            torch.mm(R1, A, out=R2)
            torch.mm(R2, B, out=R1)
            torch.mm(R1, C, out=R2)
            torch.mm(R2, D, out=R1)
            torch.mm(R1, A, out=R2)
            iterations += 8

        # 5초마다만 출력 (동기화 최소화)
        now = time.perf_counter()
        if now - last_print >= 5.0:
            torch.cuda.synchronize()
            elapsed = now - start
            ops = iterations * 2 * (size ** 3)
            tflops = ops / elapsed / 1e12
            print(f"  {elapsed:.0f}s: {iterations} iters, {tflops:.1f} TFLOPS")
            last_print = now

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ops = iterations * 2 * (size ** 3)
    final_tflops = ops / elapsed / 1e12

    print("-" * 70)
    print(f"  DONE: {iterations} iterations in {elapsed:.1f}s")
    print(f"  TFLOPS: {final_tflops:.1f}")
    print("=" * 70)

    return final_tflops


def main():
    print("\n" + "*" * 70)
    print("*  CPU 100% + GPU 100% SIMULTANEOUS STRESS")
    print("*" * 70)
    print(f"  CPU: i9-14900HX ({mp.cpu_count()} threads)")
    print(f"  GPU: RTX 4070 Laptop")
    print("*" * 70)

    duration = 45

    # CPU 프로세스들 시작
    print("\n[CPU] Starting 32 worker processes...")
    cpu_procs = []
    for i in range(mp.cpu_count()):
        p = mp.Process(target=cpu_worker, args=(duration + 10,))
        p.start()
        cpu_procs.append(p)

    time.sleep(2)  # CPU 워밍업

    # GPU 스트레스
    print("\n[GPU] Starting non-stop stress...")
    tflops = gpu_nonstop_stress(duration)

    # CPU 정리
    print("\n[CPU] Waiting for workers...")
    for p in cpu_procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print(f"  GPU: {tflops:.1f} TFLOPS")
    print(f"  CPU: 32 threads")
    print("=" * 70)


if __name__ == '__main__':
    mp.freeze_support()
    main()
