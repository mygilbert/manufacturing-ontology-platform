"""
FDC Analysis Agent
==================

Ollama LLM 기반 FDC 이상 분석 Agent

기능:
- 자연어 질문 이해
- 도구 호출을 통한 데이터 조회
- 근본원인 분석 및 추천

사용법:
    agent = FDCAnalysisAgent()
    response = agent.analyze("ETCH-001에서 온도 알람이 발생했습니다. 원인은?")
"""

import os
import sys
import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

# 모듈 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ollama_client import OllamaClient, ChatMessage
from .tools import ToolRegistry, ToolResult, create_default_registry

# 중국어/일본어 -> 한국어 변환 맵
CHINESE_TO_KOREAN = {
    # 중국어
    "过高": "과도하게 높음",
    "详细": "상세한",
    "检查": "점검",
    "磨损": "마모",
    "修正": "수정",
    "正确": "정확",
    "确认": "확인",
    "进一步": "추가적인",
    "问题": "문제",
    "原因": "원인",
    "解决": "해결",
    "需要": "필요",
    "可能": "가능",
    "建议": "권장",
    "分析": "분석",
    "数据": "데이터",
    "系统": "시스템",
    "设备": "설비",
    "温度": "온도",
    "压力": "압력",
    "流量": "유량",
    "电流": "전류",
    "功率": "파워",
    "故障": "고장",
    "异常": "이상",
    "正常": "정상",
    "警报": "알람",
    "传感器": "센서",
    "冷却": "냉각",
    "加热": "가열",
    "现象": "현상",
    "通常": "일반적으로",
    "更": "더",
    "并": "그리고",
    "高温": "고온",
    "低温": "저온",
    "确定": "확정",
    "高": "높은",
    "低": "낮은",
    "必要": "필요",
    "防止": "방지",
    "如果": "만약",
    "因为": "왜냐하면",
    "所以": "그래서",
    "但是": "하지만",
    "然后": "그 다음",
    "首先": "먼저",
    "最后": "마지막으로",
    "状态": "상태",
    "结果": "결과",
    "方法": "방법",
    "步骤": "단계",
    "确保": "확인",
    "影响": "영향",
    "导致": "초래",
    "造成": "유발",
    "引起": "일으킴",
    "保持": "유지",
    "操作": "작동",
    "运行": "실행",
    "停止": "중지",
    "开始": "시작",
    "完成": "완료",
    "发现": "발견",
    "显示": "표시",
    "表明": "나타남",
    "可以": "할 수 있음",
    "应该": "해야 함",
    "必须": "반드시",
    "已经": "이미",
    "继续": "계속",
    "增加": "증가",
    "减少": "감소",
    "维护": "유지보수",
    "修理": "수리",
    "更换": "교체",
    "安装": "설치",
    "连接": "연결",
    "断开": "분리",
    "具": "도구",
    "热": "열",
    "冷": "냉",
    "查": "조회",
    "看": "확인",
    "做": "수행",
    "成": "됨",
    "要": "해야",
    "有": "있음",
    "无": "없음",
    "对": "맞음",
    "好": "좋음",
    "坏": "나쁨",
    "大": "큰",
    "小": "작은",
    "多": "많은",
    "少": "적은",
    "快": "빠른",
    "慢": "느린",
    "新": "새",
    "旧": "오래된",
    "上": "위",
    "下": "아래",
    "前": "앞",
    "后": "뒤",
    "内": "내부",
    "外": "외부",
    "中": "중",
    "间": "사이",
    "会": "할 수",
    "能": "가능",
    "被": "되어",
    "让": "하게",
    "使": "사용",
    "把": "을",
    "给": "에게",
    "从": "부터",
    "到": "까지",
    "和": "와",
    "或": "또는",
    "与": "과",
    "比": "보다",
    "请": "하세요",
    "将": "을",
    "当": "경우",
    "还": "또",
    "再": "다시",
    "也": "또한",
    "就": "바로",
    "才": "비로소",
    "只": "오직",
    "都": "모두",
    "很": "매우",
    "太": "너무",
    "非常": "매우",
    "相当": "상당히",
    "一些": "일부",
    "一个": "하나",
    "两个": "두 개",
    "几个": "몇 개",
    "每个": "각각",
    "这个": "이것",
    "那个": "저것",
    "什么": "무엇",
    "怎么": "어떻게",
    "为什么": "왜",
    "哪里": "어디",
    "什么时候": "언제",
    "谁": "누구",
    "突然": "갑자기",
    "全面": "전면적",
    "的": "인",
    "地": "하게",
    "得": "되게",
    "了": "",
    "着": "하며",
    "过": "과도",
    "来": "오다",
    "去": "가다",
    "们": "들",
    "这": "이",
    "那": "저",
    "个": "개",
    "里": "안",
    "于": "에",
    "是": "이다",
    "在": "에서",
    "不": "않",
    "没": "없",
    "被": "되어",
    # 일본어
    "があります": "있습니다",
    "ではありません": "아닙니다",
    "です": "입니다",
    "ます": "합니다",
    "性を": "성을",
    "を": "를",
    "の": "의",
}


def replace_chinese_to_korean(text: str) -> str:
    """중국어/일본어 문자를 한국어로 변환"""
    result = text

    # 1. 변환 맵 적용 (긴 문자열부터 처리)
    sorted_items = sorted(CHINESE_TO_KOREAN.items(), key=lambda x: len(x[0]), reverse=True)
    for chinese, korean in sorted_items:
        result = result.replace(chinese, korean)

    # 2. 남은 CJK 문자 제거 (한글은 유지)
    # CJK 범위: U+4E00-U+9FFF (한자), U+3040-U+30FF (히라가나/가타카나)
    cjk_pattern = '[\u4E00-\u9FFF\u3040-\u30FF\u3400-\u4DBF\uF900-\uFAFF]'
    result = re.sub(cjk_pattern, '', result)

    # 3. 중복 공백 정리
    result = re.sub(r'  +', ' ', result)

    return result


@dataclass
class AgentConfig:
    """Agent 설정"""
    model: str = "exaone3.5:7.8b"
    temperature: float = 0.3
    max_tokens: int = 2048
    max_tool_calls: int = 5
    verbose: bool = True
    custom_system_prompt: Optional[str] = None
    custom_domain_knowledge: Optional[str] = None


@dataclass
class AnalysisResult:
    """분석 결과"""
    query: str
    answer: str
    tool_calls: List[Dict] = field(default_factory=list)
    reasoning: str = ""
    execution_time_ms: float = 0
    success: bool = True
    error: Optional[str] = None


class FDCAnalysisAgent:
    """FDC 분석 Agent"""

    SYSTEM_PROMPT = """당신은 반도체 FDC(Fault Detection & Classification) 전문가 Agent입니다.

## 역할
- 설비 이상 현상의 근본 원인을 분석합니다
- 알람 발생 시 점검 순서를 제안합니다
- 공정 데이터를 분석하여 이상 패턴을 탐지합니다

## 사용 가능한 도구
다음 도구를 사용하여 정보를 수집할 수 있습니다:

{tool_descriptions}

## 도구 호출 형식
도구를 사용하려면 다음 형식을 사용하세요:
```tool
{{"tool": "도구이름", "params": {{"param1": "value1"}}}}
```

## 언어 규칙 (매우 중요!)
- 반드시 100% 한국어로만 응답하세요
- 중국어 한자를 절대 사용하지 마세요 (过高, 详细, 检查, 磨损, 修正 등 금지)
- 한글로 표현하세요: 과도하게 높음, 상세한, 점검, 마모, 수정
- 기술 용어만 영어 허용 (RF Power, Temperature, Bearing 등)

## 응답 규칙
1. 질문을 분석하고 필요한 도구를 호출하세요
2. 도구 결과를 바탕으로 분석하세요
3. 근본 원인과 권장 조치를 제시하세요
4. 불확실한 경우 "추가 분석 필요"라고 명시하세요

## 검증된 인과관계
{domain_knowledge}
"""

    DOMAIN_KNOWLEDGE = """
- RF_POWER → TEMP (lag: 1초): RF 파워 증가 → 플라즈마 온도 상승
- FLOW → PRESSURE (lag: 3초): 가스 유량 변화 → 챔버 압력 변화
- TEMP → PRESSURE (lag: 2초): 온도 상승 → 압력 상승
- TEMP 상한 초과 시 점검 순서: RF_POWER → FLOW → 센서 캘리브레이션
"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        data_path: Optional[str] = None
    ):
        self.config = config or AgentConfig()
        self.client = OllamaClient(model=self.config.model)
        self.tools = tool_registry or create_default_registry(data_path)
        self.conversation_history: List[ChatMessage] = []

        # 커스텀 프롬프트 저장
        self._custom_system_prompt: Optional[str] = self.config.custom_system_prompt
        self._custom_domain_knowledge: Optional[str] = self.config.custom_domain_knowledge

    def set_system_prompt(self, prompt: str) -> None:
        """시스템 프롬프트 동적 변경"""
        self._custom_system_prompt = prompt

    def set_domain_knowledge(self, knowledge: str) -> None:
        """도메인 지식 동적 변경"""
        self._custom_domain_knowledge = knowledge

    def get_current_prompt(self) -> Dict[str, Any]:
        """현재 프롬프트 조회"""
        return {
            "system_prompt": self._custom_system_prompt or self.SYSTEM_PROMPT,
            "domain_knowledge": self._custom_domain_knowledge or self.DOMAIN_KNOWLEDGE,
            "is_custom_system_prompt": self._custom_system_prompt is not None,
            "is_custom_domain_knowledge": self._custom_domain_knowledge is not None
        }

    def reset_prompt(self) -> None:
        """프롬프트를 기본값으로 리셋"""
        self._custom_system_prompt = None
        self._custom_domain_knowledge = None

    def _build_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        tool_descriptions = ""
        for tool_def in self.tools.get_all_definitions():
            tool_descriptions += f"\n### {tool_def['name']}\n"
            tool_descriptions += f"{tool_def['description']}\n"
            tool_descriptions += f"파라미터: {json.dumps(tool_def['parameters'], ensure_ascii=False)}\n"

        # 커스텀 프롬프트 사용 여부 확인
        base_prompt = self._custom_system_prompt or self.SYSTEM_PROMPT
        domain_knowledge = self._custom_domain_knowledge or self.DOMAIN_KNOWLEDGE

        return base_prompt.format(
            tool_descriptions=tool_descriptions,
            domain_knowledge=domain_knowledge
        )

    def _parse_tool_calls(self, response: str) -> List[Dict]:
        """응답에서 도구 호출 추출"""
        tool_calls = []

        # ```tool ... ``` 블록 찾기
        tool_pattern = r'```tool\s*\n?(.*?)\n?```'
        matches = re.findall(tool_pattern, response, re.DOTALL)

        for match in matches:
            try:
                # JSON 파싱
                tool_call = json.loads(match.strip())
                if "tool" in tool_call:
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # JSON이 아닌 경우 무시
                continue

        # 간단한 형식도 지원: TOOL: tool_name(params)
        simple_pattern = r'TOOL:\s*(\w+)\((.*?)\)'
        simple_matches = re.findall(simple_pattern, response)
        for tool_name, params_str in simple_matches:
            try:
                params = json.loads(params_str) if params_str else {}
                tool_calls.append({"tool": tool_name, "params": params})
            except:
                pass

        return tool_calls

    def _execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        """도구 실행"""
        results = []

        for call in tool_calls:
            tool_name = call.get("tool")
            params = call.get("params", {})

            if self.config.verbose:
                print(f"  [Tool] {tool_name}: {params}")

            result = self.tools.execute(tool_name, **params)

            results.append({
                "tool": tool_name,
                "params": params,
                "result": result.data if result.success else None,
                "success": result.success,
                "message": result.message
            })

            if self.config.verbose:
                status = "OK" if result.success else "FAIL"
                print(f"  [{status}] {result.message}")

        return results

    def _format_tool_results(self, results: List[Dict]) -> str:
        """도구 결과 포맷팅"""
        formatted = "\n## 도구 실행 결과\n"

        for r in results:
            formatted += f"\n### {r['tool']}\n"
            if r['success']:
                formatted += f"```json\n{json.dumps(r['result'], ensure_ascii=False, indent=2)}\n```\n"
            else:
                formatted += f"오류: {r['message']}\n"

        return formatted

    def analyze(self, query: str) -> AnalysisResult:
        """질문 분석 및 응답 생성"""
        start_time = datetime.now()

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"[Query] {query}")
            print(f"{'='*60}")

        # Ollama 서버 확인
        if not self.client.is_available():
            return AnalysisResult(
                query=query,
                answer="",
                success=False,
                error="Ollama 서버에 연결할 수 없습니다. 'ollama serve' 명령으로 서버를 시작하세요."
            )

        try:
            # 대화 초기화
            messages = [
                ChatMessage(role="system", content=self._build_system_prompt()),
                ChatMessage(role="user", content=query)
            ]

            all_tool_calls = []
            iteration = 0

            while iteration < self.config.max_tool_calls:
                iteration += 1

                if self.config.verbose:
                    print(f"\n[Iteration {iteration}] LLM 호출 중...")

                # LLM 응답 받기
                response = self.client.chat(
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )

                assistant_response = response.content

                if self.config.verbose:
                    print(f"[Response] {assistant_response[:200]}...")

                # 도구 호출 파싱
                tool_calls = self._parse_tool_calls(assistant_response)

                if not tool_calls:
                    # 도구 호출이 없으면 최종 응답
                    break

                # 도구 실행
                tool_results = self._execute_tools(tool_calls)
                all_tool_calls.extend(tool_results)

                # 도구 결과를 대화에 추가
                messages.append(ChatMessage(role="assistant", content=assistant_response))
                messages.append(ChatMessage(
                    role="user",
                    content=self._format_tool_results(tool_results) + "\n위 결과를 바탕으로 분석을 완료해주세요."
                ))

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # 최종 응답에서 도구 호출 블록 제거
            final_answer = re.sub(r'```tool.*?```', '', assistant_response, flags=re.DOTALL).strip()

            # <think> 태그 내용 추출 (deepseek-r1 모델용)
            reasoning = ""
            think_match = re.search(r'<think>(.*?)</think>', assistant_response, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                final_answer = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL).strip()

            # 중국어 -> 한국어 후처리
            final_answer = replace_chinese_to_korean(final_answer)
            reasoning = replace_chinese_to_korean(reasoning)

            return AnalysisResult(
                query=query,
                answer=final_answer,
                tool_calls=all_tool_calls,
                reasoning=reasoning,
                execution_time_ms=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return AnalysisResult(
                query=query,
                answer="",
                execution_time_ms=execution_time,
                success=False,
                error=str(e)
            )

    def analyze_alarm(self, equipment_id: str, alarm_code: str) -> AnalysisResult:
        """알람 분석 (편의 메서드)"""
        query = f"{equipment_id} 설비에서 {alarm_code} 알람이 발생했습니다. 원인과 조치 방법을 알려주세요."
        return self.analyze(query)

    def analyze_anomaly(self, equipment_id: str, parameter: str, value: float) -> AnalysisResult:
        """이상값 분석 (편의 메서드)"""
        query = f"{equipment_id} 설비의 {parameter} 값이 {value}입니다. 이상 여부와 원인을 분석해주세요."
        return self.analyze(query)

    def get_checklist(self, alarm_code: str) -> AnalysisResult:
        """알람별 점검 체크리스트 (편의 메서드)"""
        query = f"{alarm_code} 알람 발생 시 점검해야 할 항목을 순서대로 알려주세요."
        return self.analyze(query)

    def reset_conversation(self):
        """대화 이력 초기화"""
        self.conversation_history = []


class AgentStreamHandler:
    """스트리밍 응답 핸들러 (향후 확장용)"""

    def __init__(self, agent: FDCAnalysisAgent):
        self.agent = agent

    def stream_analyze(self, query: str):
        """스트리밍 분석 (향후 구현)"""
        # TODO: 스트리밍 응답 구현
        pass


# 간편 사용을 위한 함수
def quick_analyze(query: str, model: str = "deepseek-r1:8b", verbose: bool = False) -> str:
    """빠른 분석"""
    config = AgentConfig(model=model, verbose=verbose)
    agent = FDCAnalysisAgent(config=config)
    result = agent.analyze(query)

    if result.success:
        return result.answer
    else:
        return f"Error: {result.error}"
