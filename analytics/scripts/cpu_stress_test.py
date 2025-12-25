"""
CPU Stress Test for i9-14900HX
==============================
24코어 32스레드를 최대한 활용하는 CPU 집약적 테스트
"""

import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import os
import sys

def get_cpu_info():
    """CPU 정보 출력"""
    cpu_count = mp.cpu_count()
    print("=" * 60)
    print("CPU Stress Test - i9-14900HX Edition")
    print("=" * 60)
    print(f"논리 프로세서 수: {cpu_count}")
    print(f"Python 버전: {sys.version}")
    print("=" * 60)
    return cpu_count


def matrix_multiplication_test(size=2000):
    """대규모 행렬 곱셈 (NumPy - 멀티스레드)"""
    print(f"\n[테스트 1] 행렬 곱셈 ({size}x{size})")
    print("-" * 40)

    # NumPy는 내부적으로 BLAS를 사용해 멀티스레드 처리
    A = np.random.rand(size, size).astype(np.float64)
    B = np.random.rand(size, size).astype(np.float64)

    start = time.perf_counter()
    for i in range(3):
        C = np.dot(A, B)
        print(f"  반복 {i+1}/3 완료")
    elapsed = time.perf_counter() - start

    print(f"  결과: {elapsed:.2f}초 (3회 반복)")
    print(f"  GFLOPS: {(3 * 2 * size**3 / elapsed / 1e9):.2f}")
    return elapsed


def prime_worker(args):
    """소수 판별 워커"""
    start, end = args
    primes = []
    for n in range(start, end):
        if n < 2:
            continue
        is_prime = True
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return len(primes)


def prime_number_test(limit=500000, num_workers=None):
    """소수 찾기 (멀티프로세싱)"""
    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"\n[테스트 2] 소수 찾기 (1 ~ {limit:,})")
    print(f"  워커 수: {num_workers}")
    print("-" * 40)

    # 작업 분할
    chunk_size = limit // num_workers
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
    ranges[-1] = (ranges[-1][0], limit)  # 마지막 청크 조정

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(prime_worker, ranges))
    elapsed = time.perf_counter() - start

    total_primes = sum(results)
    print(f"  발견된 소수: {total_primes:,}개")
    print(f"  소요 시간: {elapsed:.2f}초")
    print(f"  처리량: {limit/elapsed:,.0f} 숫자/초")
    return elapsed


def monte_carlo_worker(n_samples):
    """몬테카를로 파이 계산 워커"""
    inside = 0
    for _ in range(n_samples):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside


def monte_carlo_pi_test(total_samples=50_000_000, num_workers=None):
    """몬테카를로 시뮬레이션으로 파이 계산"""
    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"\n[테스트 3] 몬테카를로 파이 계산 ({total_samples:,} 샘플)")
    print(f"  워커 수: {num_workers}")
    print("-" * 40)

    samples_per_worker = total_samples // num_workers

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(monte_carlo_worker, [samples_per_worker] * num_workers))
    elapsed = time.perf_counter() - start

    total_inside = sum(results)
    pi_estimate = 4 * total_inside / total_samples

    print(f"  추정된 파이: {pi_estimate:.10f}")
    print(f"  실제 파이:   {np.pi:.10f}")
    print(f"  오차: {abs(pi_estimate - np.pi):.10f}")
    print(f"  소요 시간: {elapsed:.2f}초")
    print(f"  처리량: {total_samples/elapsed:,.0f} 샘플/초")
    return elapsed


def fft_test(size=2**20, iterations=50):
    """FFT 연산 테스트"""
    print(f"\n[테스트 4] FFT 연산 (크기: {size:,}, {iterations}회 반복)")
    print("-" * 40)

    data = np.random.rand(size) + 1j * np.random.rand(size)

    start = time.perf_counter()
    for i in range(iterations):
        result = np.fft.fft(data)
        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{iterations}")
    elapsed = time.perf_counter() - start

    print(f"  소요 시간: {elapsed:.2f}초")
    print(f"  평균: {elapsed/iterations*1000:.2f}ms/회")
    return elapsed


def eigenvalue_test(size=1500, iterations=5):
    """고유값 분해 테스트"""
    print(f"\n[테스트 5] 고유값 분해 ({size}x{size}, {iterations}회)")
    print("-" * 40)

    start = time.perf_counter()
    for i in range(iterations):
        A = np.random.rand(size, size)
        A = (A + A.T) / 2  # 대칭 행렬
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        print(f"  반복 {i+1}/{iterations} 완료")
    elapsed = time.perf_counter() - start

    print(f"  소요 시간: {elapsed:.2f}초")
    print(f"  평균: {elapsed/iterations:.2f}초/회")
    return elapsed


def svd_test(m=3000, n=2000, iterations=3):
    """SVD 분해 테스트"""
    print(f"\n[테스트 6] SVD 분해 ({m}x{n}, {iterations}회)")
    print("-" * 40)

    start = time.perf_counter()
    for i in range(iterations):
        A = np.random.rand(m, n)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        print(f"  반복 {i+1}/{iterations} 완료")
    elapsed = time.perf_counter() - start

    print(f"  소요 시간: {elapsed:.2f}초")
    print(f"  평균: {elapsed/iterations:.2f}초/회")
    return elapsed


def parallel_sort_worker(size):
    """정렬 워커"""
    arr = np.random.rand(size)
    sorted_arr = np.sort(arr)
    return len(sorted_arr)


def parallel_sort_test(array_size=5_000_000, num_arrays=32):
    """병렬 정렬 테스트"""
    print(f"\n[테스트 7] 병렬 정렬 ({array_size:,} 요소 x {num_arrays}개)")
    print(f"  워커 수: {mp.cpu_count()}")
    print("-" * 40)

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(parallel_sort_worker, [array_size] * num_arrays))
    elapsed = time.perf_counter() - start

    total_elements = sum(results)
    print(f"  정렬된 요소: {total_elements:,}개")
    print(f"  소요 시간: {elapsed:.2f}초")
    print(f"  처리량: {total_elements/elapsed:,.0f} 요소/초")
    return elapsed


def main():
    """메인 함수"""
    cpu_count = get_cpu_info()

    results = {}
    total_start = time.perf_counter()

    # 테스트 실행
    results['matrix'] = matrix_multiplication_test(2000)
    results['prime'] = prime_number_test(500000, cpu_count)
    results['monte_carlo'] = monte_carlo_pi_test(50_000_000, cpu_count)
    results['fft'] = fft_test(2**20, 50)
    results['eigenvalue'] = eigenvalue_test(1500, 5)
    results['svd'] = svd_test(3000, 2000, 3)
    results['sort'] = parallel_sort_test(5_000_000, 32)

    total_elapsed = time.perf_counter() - total_start

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    print(f"{'테스트':<20} {'소요시간':>15}")
    print("-" * 40)
    for name, elapsed in results.items():
        print(f"{name:<20} {elapsed:>12.2f}초")
    print("-" * 40)
    print(f"{'총 소요시간':<20} {total_elapsed:>12.2f}초")
    print("=" * 60)

    # CPU 활용도 코멘트
    print("\n💡 작업 관리자에서 CPU 사용률을 확인해보세요!")
    print("   i9-14900HX의 24코어 32스레드가 잘 활용되고 있나요?")


if __name__ == '__main__':
    main()
