"""
CPU Extreme Stress Test
=======================
i9-14900HX 24코어 32스레드를 100% 가까이 사용하는 극한 테스트
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import sys


def heavy_math_worker(args):
    """무거운 수학 연산 워커"""
    worker_id, iterations = args
    result = 0.0

    for i in range(iterations):
        # 삼각함수 + 지수함수 + 로그함수 조합
        x = np.random.random(10000)
        result += np.sum(np.sin(x) * np.cos(x) * np.exp(-x) * np.log(x + 1))
        result += np.sum(np.sqrt(x) * np.power(x, 0.3) * np.arctan(x))

        # 행렬 연산
        A = np.random.rand(200, 200)
        B = np.random.rand(200, 200)
        C = np.dot(A, B)
        result += np.trace(C)

    return result


def stress_all_cores(duration_seconds=30):
    """모든 코어를 duration_seconds 동안 100% 사용"""
    num_workers = mp.cpu_count()
    iterations_per_worker = 1000  # 각 워커당 반복 횟수

    print("=" * 60)
    print(f"CPU 극한 스트레스 테스트 (예상 소요: ~{duration_seconds}초)")
    print("=" * 60)
    print(f"논리 프로세서: {num_workers}개")
    print(f"워커당 반복: {iterations_per_worker}회")
    print("-" * 60)
    print("작업 관리자를 열어 CPU 사용률을 확인하세요!")
    print("-" * 60)

    args = [(i, iterations_per_worker) for i in range(num_workers)]

    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(heavy_math_worker, args))

    elapsed = time.perf_counter() - start

    print(f"\n완료!")
    print(f"소요 시간: {elapsed:.2f}초")
    print(f"총 연산량: {sum(results):.2e}")
    print(f"코어당 평균: {elapsed:.2f}초")

    return elapsed


def matrix_chain_worker(args):
    """연속 행렬 곱셈 워커"""
    worker_id, size, chains = args
    result = np.eye(size)

    for _ in range(chains):
        M = np.random.rand(size, size)
        result = np.dot(result, M)
        # 오버플로우 방지
        result = result / np.max(np.abs(result) + 1e-10)

    return np.sum(result)


def extreme_matrix_test():
    """극한 행렬 연산 테스트"""
    num_workers = mp.cpu_count()
    matrix_size = 500
    chains = 50

    print("\n" + "=" * 60)
    print(f"극한 행렬 체인 테스트 ({matrix_size}x{matrix_size}, {chains}체인)")
    print("=" * 60)

    args = [(i, matrix_size, chains) for i in range(num_workers)]

    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(matrix_chain_worker, args))

    elapsed = time.perf_counter() - start

    total_ops = num_workers * chains * 2 * (matrix_size ** 3)
    gflops = total_ops / elapsed / 1e9

    print(f"소요 시간: {elapsed:.2f}초")
    print(f"총 FLOPS: {gflops:.2f} GFLOPS")

    return elapsed


def neural_network_simulation_worker(args):
    """신경망 순전파 시뮬레이션"""
    worker_id, batch_size, layers = args

    # 입력
    x = np.random.randn(batch_size, 784)  # MNIST 크기

    for units in layers:
        W = np.random.randn(x.shape[1], units) * 0.01
        b = np.zeros(units)

        # Linear + ReLU
        x = np.dot(x, W) + b
        x = np.maximum(0, x)  # ReLU

    return np.sum(x)


def neural_network_test():
    """신경망 시뮬레이션 테스트"""
    num_workers = mp.cpu_count()
    batch_size = 1024
    layers = [512, 256, 128, 64, 10]  # MLP 구조
    iterations = 100

    print("\n" + "=" * 60)
    print(f"신경망 순전파 시뮬레이션 ({iterations}회 x {num_workers}워커)")
    print(f"배치 크기: {batch_size}, 레이어: {layers}")
    print("=" * 60)

    args = [(i, batch_size, layers) for i in range(num_workers * iterations)]

    start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(neural_network_simulation_worker, args))

    elapsed = time.perf_counter() - start

    throughput = len(args) * batch_size / elapsed

    print(f"소요 시간: {elapsed:.2f}초")
    print(f"처리량: {throughput:,.0f} 샘플/초")

    return elapsed


def memory_bandwidth_test():
    """메모리 대역폭 테스트"""
    print("\n" + "=" * 60)
    print("메모리 대역폭 테스트")
    print("=" * 60)

    # 큰 배열 할당 (4GB)
    size = 500_000_000  # 500M elements = 4GB for float64

    print(f"배열 크기: {size * 8 / 1e9:.1f} GB")

    # 할당
    start = time.perf_counter()
    arr = np.random.rand(size)
    alloc_time = time.perf_counter() - start
    print(f"할당 시간: {alloc_time:.2f}초")

    # 읽기 (합계)
    start = time.perf_counter()
    total = np.sum(arr)
    read_time = time.perf_counter() - start
    read_bandwidth = size * 8 / read_time / 1e9
    print(f"읽기 시간: {read_time:.2f}초 ({read_bandwidth:.1f} GB/s)")

    # 쓰기
    start = time.perf_counter()
    arr *= 2.0
    write_time = time.perf_counter() - start
    write_bandwidth = size * 8 / write_time / 1e9
    print(f"쓰기 시간: {write_time:.2f}초 ({write_bandwidth:.1f} GB/s)")

    # 복사
    start = time.perf_counter()
    arr2 = arr.copy()
    copy_time = time.perf_counter() - start
    copy_bandwidth = size * 8 * 2 / copy_time / 1e9  # 읽기+쓰기
    print(f"복사 시간: {copy_time:.2f}초 ({copy_bandwidth:.1f} GB/s)")

    del arr, arr2

    return read_bandwidth


def main():
    print("\n" + "=" * 60)
    print("i9-14900HX 극한 스트레스 테스트")
    print("=" * 60)
    print(f"CPU 스레드: {mp.cpu_count()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)

    results = {}
    total_start = time.perf_counter()

    # 테스트 실행
    results['stress'] = stress_all_cores(30)
    results['matrix_chain'] = extreme_matrix_test()
    results['neural_net'] = neural_network_test()
    results['memory'] = memory_bandwidth_test()

    total_elapsed = time.perf_counter() - total_start

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    for name, value in results.items():
        if name == 'memory':
            print(f"{name}: {value:.1f} GB/s (읽기 대역폭)")
        else:
            print(f"{name}: {value:.2f}초")
    print("-" * 60)
    print(f"총 소요 시간: {total_elapsed:.2f}초")
    print("=" * 60)


if __name__ == '__main__':
    mp.freeze_support()  # Windows 지원
    main()
