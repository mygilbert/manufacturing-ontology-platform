"""
FDC Analysis Agent 테스트 스크립트
==================================

Ollama + deepseek-r1 모델을 사용한 FDC 분석 Agent 테스트

사용법:
    python scripts/test_fdc_agent.py
"""

import sys
import os
import time

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

from agent import FDCAnalysisAgent, OllamaClient
from agent.fdc_agent import AgentConfig


def test_ollama_connection():
    """Ollama 서버 연결 테스트"""
    print("\n" + "=" * 60)
    print("[1] Ollama 서버 연결 테스트")
    print("=" * 60)

    client = OllamaClient()

    if client.is_available():
        print("  [OK] Ollama 서버 연결 성공")
        models = client.list_models()
        print(f"  [OK] 사용 가능한 모델: {models}")
        return True
    else:
        print("  [FAIL] Ollama 서버에 연결할 수 없습니다.")
        print("  [INFO] 'ollama serve' 명령으로 서버를 시작하세요.")
        return False


def test_simple_generation():
    """간단한 텍스트 생성 테스트"""
    print("\n" + "=" * 60)
    print("[2] 간단한 텍스트 생성 테스트")
    print("=" * 60)

    client = OllamaClient(model="deepseek-r1:8b")

    prompt = "반도체 FDC 시스템이 무엇인지 한 문장으로 설명해주세요."
    print(f"  [Prompt] {prompt}")

    start_time = time.time()
    response = client.generate(prompt, temperature=0.5, max_tokens=200)
    elapsed = time.time() - start_time

    print(f"  [Response] {response.content[:200]}...")
    print(f"  [Time] {elapsed:.2f}초")

    return True


def test_tool_execution():
    """도구 실행 테스트"""
    print("\n" + "=" * 60)
    print("[3] 도구 실행 테스트")
    print("=" * 60)

    from agent.tools import create_default_registry

    registry = create_default_registry()

    # 온톨로지 검색
    print("\n  [Test] 온톨로지 검색")
    result = registry.execute("ontology_search", query_type="equipment", entity_id="ETCH-001")
    print(f"  [Result] {result.success}: {result.message}")
    if result.data:
        print(f"  [Data] {result.data}")

    # 근본원인 분석
    print("\n  [Test] 근본원인 분석")
    result = registry.execute("root_cause_analysis", alarm_code="ALM_HIGH_TEMP")
    print(f"  [Result] {result.success}: {result.message}")
    if result.data:
        print(f"  [Checklist] {result.data.get('checklist', [])}")

    # 시계열 분석
    print("\n  [Test] 시계열 분석")
    result = registry.execute("timeseries_analysis", sensor_id="TEMP_001", analysis_type="statistics")
    print(f"  [Result] {result.success}: {result.message}")
    if result.data and 'statistics' in result.data:
        stats = result.data['statistics']
        print(f"  [Stats] mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    return True


def test_agent_analysis():
    """Agent 분석 테스트"""
    print("\n" + "=" * 60)
    print("[4] FDC Agent 분석 테스트")
    print("=" * 60)

    config = AgentConfig(
        model="deepseek-r1:8b",
        temperature=0.3,
        max_tokens=1024,
        verbose=True
    )

    agent = FDCAnalysisAgent(config=config)

    # 테스트 질문들
    test_queries = [
        "ETCH-001 설비에서 온도 알람(ALM_HIGH_TEMP)이 발생했습니다. 원인을 분석해주세요.",
        # "ETCH-001의 TEMP_001 센서 상태를 확인해주세요.",
        # "ALM_HIGH_TEMP 알람 발생 시 점검 순서를 알려주세요.",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"[Test {i}] {query}")
        print(f"{'='*60}")

        result = agent.analyze(query)

        if result.success:
            print(f"\n[Answer]")
            print(result.answer)

            if result.tool_calls:
                print(f"\n[Tool Calls] {len(result.tool_calls)}개")
                for tc in result.tool_calls:
                    print(f"  - {tc['tool']}: {tc['success']}")

            if result.reasoning:
                print(f"\n[Reasoning]")
                print(result.reasoning[:300] + "..." if len(result.reasoning) > 300 else result.reasoning)

            print(f"\n[Time] {result.execution_time_ms:.0f}ms")
        else:
            print(f"\n[Error] {result.error}")

    return True


def main():
    print("\n" + "=" * 60)
    print("FDC Analysis Agent 테스트")
    print("=" * 60)
    print("Model: deepseek-r1:8b")
    print("=" * 60)

    # 1. Ollama 연결 테스트
    if not test_ollama_connection():
        print("\n[ABORT] Ollama 서버가 실행되지 않았습니다.")
        print("[INFO] 다음 명령으로 Ollama를 시작하세요:")
        print("       ollama serve")
        return

    # 2. 간단한 생성 테스트
    try:
        test_simple_generation()
    except Exception as e:
        print(f"  [Error] {e}")

    # 3. 도구 테스트
    try:
        test_tool_execution()
    except Exception as e:
        print(f"  [Error] {e}")

    # 4. Agent 분석 테스트
    try:
        test_agent_analysis()
    except Exception as e:
        print(f"  [Error] {e}")

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
