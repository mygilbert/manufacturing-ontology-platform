"""
GPU CUDA 100% - Maximum Compute Utilization
============================================
"""

import torch
import numpy as np
import multiprocessing as mp
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

device = torch.device('cuda')


def cpu_stress_proc(duration):
    """CPU 스트레스 프로세스"""
    end = time.time() + duration
    while time.time() < end:
        A = np.random.rand(250, 250)
        B = np.random.rand(250, 250)
        np.dot(A, B)


def main():
    print("\n" + "=" * 70)
    print("GPU CUDA 100% + CPU 100%")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU: {mp.cpu_count()} threads")
    print("=" * 70)
    print(">>> Task Manager > GPU > Change graph to 'Cuda' <<<")
    print("=" * 70)

    duration = 60

    # CPU 프로세스 시작
    print("\n[CPU] Starting workers...")
    cpu_procs = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=cpu_stress_proc, args=(duration + 10,))
        p.start()
        cpu_procs.append(p)

    time.sleep(2)

    # GPU 메모리 최대 할당
    print("\n[GPU] Allocating memory...")

    size = 8192
    matrices = [torch.randn(size, size, device=device, dtype=torch.float16) for _ in range(8)]
    results = [torch.empty(size, size, device=device, dtype=torch.float16) for _ in range(4)]

    print(f"  {len(matrices)} matrices of {size}x{size}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n[GPU] Running...")
    print("-" * 70)

    iterations = 0
    start = time.perf_counter()
    last_print = start

    while time.perf_counter() - start < duration:
        for _ in range(20):
            for i in range(len(matrices) - 1):
                torch.mm(matrices[i], matrices[i+1], out=results[i % 4])
            iterations += 7

        now = time.perf_counter()
        if now - last_print >= 10:
            torch.cuda.synchronize()
            elapsed = now - start
            tflops = iterations * 2 * (size**3) / elapsed / 1e12
            print(f"  {elapsed:.0f}s: {tflops:.1f} TFLOPS")
            last_print = now

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tflops = iterations * 2 * (size**3) / elapsed / 1e12

    print("-" * 70)
    print(f"  RESULT: {tflops:.1f} TFLOPS")

    print("\n[CPU] Stopping...")
    for p in cpu_procs:
        p.terminate()

    print("\n" + "=" * 70)
    print(f"GPU: {tflops:.1f} TFLOPS | CPU: 32 threads | {elapsed:.0f}s")
    print("=" * 70)


if __name__ == '__main__':
    mp.freeze_support()
    main()
