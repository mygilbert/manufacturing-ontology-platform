"""
FDC Analysis Agent Router
=========================

Ollama LLM 기반 FDC 분석 Agent API 엔드포인트
"""

import sys
import os
import re
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio
import json


# 중국어/일본어 -> 한국어 변환
def clean_cjk_text(text: str) -> str:
    """응답에서 중국어/일본어 문자를 한국어로 변환 또는 제거"""
    if not text:
        return text

    # 변환 맵 (자주 출현하는 단어)
    replacements = {
        # 기본 단어
        "过高": "과도하게 높음", "详细": "상세한", "检查": "점검", "磨损": "마모",
        "突然": "갑자기", "高温": "고온", "低温": "저온", "必要": "필요",
        "防止": "방지", "原因": "원인", "分析": "분석", "结果": "결과",
        "问题": "문제", "解决": "해결", "确认": "확인", "正常": "정상",
        "异常": "이상", "故障": "고장", "传感器": "센서", "冷却": "냉각",
        "温度": "온도", "压力": "압力", "流量": "유량", "设备": "설비",
        "系统": "시스템", "状态": "상태", "方法": "방법", "步骤": "단계",
        "增加": "증가", "减少": "감소", "维护": "유지보수", "修理": "수리",
        "更换": "교체", "连接": "연결", "运行": "실행", "停止": "중지",
        "开始": "시작", "完成": "완료", "发现": "발견", "显示": "표시",
        "进一步": "추가적인", "可能": "가능", "应该": "해야", "需要": "필요",
        "因素": "요인", "现象": "현상", "通常": "일반적으로", "主要": "주요",
        "全面": "전면적", "潜在": "잠재적", "排除": "배제", "基于": "기반으로",
        "覆盖": "포함", "追跡": "추적", "内": "이내", "其他": "기타",
        "之间": "사이", "润滑剂": "윤활제", "润滑": "윤활",
        # EXAONE 추가 단어
        "常见": "흔한", "加速": "가속", "诱发": "유발", "诱發": "유발",
        "负载": "부하", "监控": "모니터링", "校准": "교정", "具体": "구체적",
        "严重": "심각한", "導致": "초래", "確認": "확인", "保养": "유지보수",
        "校正": "교정", "深": "깊", "常": "상", "见": "견",
        "處理": "처리", "執行": "실행", "檢查": "점검", "維護": "유지보수",
        "發生": "발생", "發現": "발견", "狀態": "상태", "設備": "설비",
        "問題": "문제", "溫度": "온도", "壓力": "압력", "電流": "전류",
        "功率": "파워", "振動": "진동", "異常": "이상", "故障": "고장",
        "警報": "알람", "警告": "경고", "正常": "정상", "測量": "측정",
        # 접속사/조사
        "的": "", "与": "와", "和": "와", "或": "또는", "对": "에 대해",
        "如果": "만약", "因为": "왜냐하면", "所以": "그래서", "但是": "하지만",
        "而且": "그리고", "之后": "이후", "之前": "이전", "以上": "이상",
        "以下": "이하", "左右": "정도", "这": "이", "那": "저",
        "会": "할 수 있", "能": "가능", "要": "해야", "把": "",
        "了": "", "着": "", "过": "", "得": "", "地": "",
        # 일본어
        "があります": "있습니다", "です": "입니다", "ます": "합니다",
        # 베트남어 (EXAONE에서 가끔 출현)
        "đột": "갑자기",
    }

    result = text

    # <think> 태그 제거 (deepseek 모델의 thinking 출력)
    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)

    # 긴 문자열부터 먼저 처리
    sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
    for cn, kr in sorted_replacements:
        result = result.replace(cn, kr)

    # 남은 CJK 문자 제거 (한자, 히라가나, 가타카나) - 한글 제외
    # CJK Unified Ideographs: U+4E00-U+9FFF
    # Hiragana: U+3040-U+309F
    # Katakana: U+30A0-U+30FF
    # CJK Extension A: U+3400-U+4DBF
    cjk_chars = ''.join([
        '\u4e00-\u9fff',  # CJK Unified
        '\u3040-\u309f',  # Hiragana
        '\u30a0-\u30ff',  # Katakana
        '\u3400-\u4dbf',  # CJK Extension A
        '\uf900-\ufaff',  # CJK Compatibility
    ])
    result = re.sub(f'[{cjk_chars}]+', '', result)

    # 중복 공백 및 연속 줄바꿈 정리
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\n +', '\n', result)
    result = re.sub(r'\n\n\n+', '\n\n', result)

    return result

# Analytics 모듈 경로 추가
analytics_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'analytics', 'src'))
sys.path.insert(0, analytics_src)

router = APIRouter()


# ============ Request/Response Models ============

class AnalyzeRequest(BaseModel):
    """분석 요청"""
    query: str = Field(..., description="분석 질문", example="ETCH-001에서 온도 알람이 발생했습니다. 원인은?")
    model: str = Field(default="exaone3.5:7.8b", description="사용할 LLM 모델")
    temperature: float = Field(default=0.3, ge=0, le=1, description="생성 온도")
    max_tokens: int = Field(default=1024, ge=100, le=4096, description="최대 토큰 수")


class AlarmAnalyzeRequest(BaseModel):
    """알람 분석 요청"""
    equipment_id: str = Field(..., description="설비 ID", example="ETCH-001")
    alarm_code: str = Field(..., description="알람 코드", example="ALM_HIGH_TEMP")


class AnomalyAnalyzeRequest(BaseModel):
    """이상값 분석 요청"""
    equipment_id: str = Field(..., description="설비 ID", example="ETCH-001")
    parameter: str = Field(..., description="파라미터 이름", example="TEMP_001")
    value: float = Field(..., description="측정값", example=85.5)


class ToolCallInfo(BaseModel):
    """도구 호출 정보"""
    tool: str
    params: dict
    success: bool
    message: str


class AnalyzeResponse(BaseModel):
    """분석 응답"""
    success: bool
    query: str
    answer: str
    tool_calls: List[ToolCallInfo] = []
    reasoning: Optional[str] = None
    execution_time_ms: float
    error: Optional[str] = None


class AgentStatusResponse(BaseModel):
    """Agent 상태 응답"""
    available: bool
    model: str
    ollama_connected: bool
    available_tools: List[str]


class PromptUpdateRequest(BaseModel):
    """프롬프트 업데이트 요청"""
    system_prompt: Optional[str] = Field(None, description="커스텀 시스템 프롬프트")
    domain_knowledge: Optional[str] = Field(None, description="커스텀 도메인 지식")


class PromptResponse(BaseModel):
    """프롬프트 응답"""
    system_prompt: str
    domain_knowledge: str
    is_custom_system_prompt: bool
    is_custom_domain_knowledge: bool


# ============ Lazy Loading ============

_agent = None
_client = None


def get_agent():
    """Agent 인스턴스 (lazy loading)"""
    global _agent
    if _agent is None:
        try:
            from agent import FDCAnalysisAgent
            from agent.fdc_agent import AgentConfig

            config = AgentConfig(
                model="exaone3.5:7.8b",
                verbose=False
            )
            _agent = FDCAnalysisAgent(config=config)
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"Agent module not found: {e}")
    return _agent


def get_ollama_client():
    """Ollama 클라이언트 (lazy loading)"""
    global _client
    if _client is None:
        try:
            from agent import OllamaClient
            _client = OllamaClient()
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"Ollama client not found: {e}")
    return _client


# ============ Endpoints ============

@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Agent 상태 확인"""
    try:
        client = get_ollama_client()
        ollama_connected = client.is_available()
        models = client.list_models() if ollama_connected else []

        agent = get_agent()
        tools = agent.tools.list_tools()

        return AgentStatusResponse(
            available=ollama_connected,
            model="deepseek-r1:8b" if "deepseek-r1:8b" in models else (models[0] if models else "none"),
            ollama_connected=ollama_connected,
            available_tools=tools
        )
    except Exception as e:
        return AgentStatusResponse(
            available=False,
            model="none",
            ollama_connected=False,
            available_tools=[]
        )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_query(request: AnalyzeRequest):
    """
    자연어 질문 분석

    FDC 관련 질문을 분석하고 근본원인, 점검 순서 등을 제공합니다.

    예시 질문:
    - "ETCH-001에서 온도 알람이 발생했습니다. 원인은?"
    - "ALM_HIGH_TEMP 알람의 점검 순서를 알려주세요."
    - "TEMP_001 센서의 현재 상태를 분석해주세요."
    """
    try:
        agent = get_agent()

        # 모델 변경이 필요한 경우
        if request.model != agent.config.model:
            agent.client.model = request.model

        # 분석 실행
        result = agent.analyze(request.query)

        # 중국어/일본어 후처리
        cleaned_answer = clean_cjk_text(result.answer)
        cleaned_reasoning = clean_cjk_text(result.reasoning) if result.reasoning else None

        return AnalyzeResponse(
            success=result.success,
            query=result.query,
            answer=cleaned_answer,
            tool_calls=[
                ToolCallInfo(
                    tool=tc.get('tool', ''),
                    params=tc.get('params', {}),
                    success=tc.get('success', False),
                    message=tc.get('message', '')
                )
                for tc in result.tool_calls
            ],
            reasoning=cleaned_reasoning,
            execution_time_ms=result.execution_time_ms,
            error=result.error
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/alarm", response_model=AnalyzeResponse)
async def analyze_alarm(request: AlarmAnalyzeRequest):
    """
    알람 분석

    특정 설비의 알람 발생 원인을 분석하고 점검 순서를 제공합니다.
    """
    try:
        agent = get_agent()
        result = agent.analyze_alarm(request.equipment_id, request.alarm_code)

        cleaned_answer = clean_cjk_text(result.answer)
        cleaned_reasoning = clean_cjk_text(result.reasoning) if result.reasoning else None

        return AnalyzeResponse(
            success=result.success,
            query=result.query,
            answer=cleaned_answer,
            tool_calls=[
                ToolCallInfo(
                    tool=tc.get('tool', ''),
                    params=tc.get('params', {}),
                    success=tc.get('success', False),
                    message=tc.get('message', '')
                )
                for tc in result.tool_calls
            ],
            reasoning=cleaned_reasoning,
            execution_time_ms=result.execution_time_ms,
            error=result.error
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/anomaly", response_model=AnalyzeResponse)
async def analyze_anomaly(request: AnomalyAnalyzeRequest):
    """
    이상값 분석

    특정 파라미터의 측정값이 이상인지 분석하고 원인을 제공합니다.
    """
    try:
        agent = get_agent()
        result = agent.analyze_anomaly(request.equipment_id, request.parameter, request.value)

        cleaned_answer = clean_cjk_text(result.answer)
        cleaned_reasoning = clean_cjk_text(result.reasoning) if result.reasoning else None

        return AnalyzeResponse(
            success=result.success,
            query=result.query,
            answer=cleaned_answer,
            tool_calls=[
                ToolCallInfo(
                    tool=tc.get('tool', ''),
                    params=tc.get('params', {}),
                    success=tc.get('success', False),
                    message=tc.get('message', '')
                )
                for tc in result.tool_calls
            ],
            reasoning=cleaned_reasoning,
            execution_time_ms=result.execution_time_ms,
            error=result.error
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def list_tools():
    """
    사용 가능한 도구 목록

    Agent가 사용할 수 있는 도구와 파라미터 정보를 반환합니다.
    """
    try:
        agent = get_agent()
        definitions = agent.tools.get_all_definitions()

        return {
            "tools": definitions,
            "count": len(definitions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, params: dict = {}):
    """
    도구 직접 실행

    특정 도구를 직접 실행하고 결과를 반환합니다.
    """
    try:
        agent = get_agent()
        result = agent.tools.execute(tool_name, **params)

        return {
            "tool": tool_name,
            "success": result.success,
            "data": result.data,
            "message": result.message,
            "execution_time_ms": result.execution_time_ms
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: AnalyzeRequest):
    """
    스트리밍 채팅 (Server-Sent Events)

    실시간으로 응답을 스트리밍합니다.
    """
    try:
        client = get_ollama_client()

        if not client.is_available():
            raise HTTPException(status_code=503, detail="Ollama server not available")

        from agent.ollama_client import ChatMessage

        messages = [
            ChatMessage(role="user", content=request.query)
        ]

        async def generate():
            try:
                for chunk in client.chat_stream(
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ Prompt Management ============

@router.get("/prompt", response_model=PromptResponse)
async def get_prompt():
    """
    현재 프롬프트 조회

    현재 설정된 시스템 프롬프트와 도메인 지식을 반환합니다.
    """
    try:
        agent = get_agent()
        prompt_info = agent.get_current_prompt()

        return PromptResponse(
            system_prompt=prompt_info["system_prompt"],
            domain_knowledge=prompt_info["domain_knowledge"],
            is_custom_system_prompt=prompt_info["is_custom_system_prompt"],
            is_custom_domain_knowledge=prompt_info["is_custom_domain_knowledge"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/prompt")
async def update_prompt(request: PromptUpdateRequest):
    """
    프롬프트 업데이트

    시스템 프롬프트와 도메인 지식을 동적으로 변경합니다.

    **참고**: 시스템 프롬프트에는 `{tool_descriptions}`와 `{domain_knowledge}` 플레이스홀더가 필요합니다.

    **예시 domain_knowledge**:
    ```
    - VIBRATION → BEARING_FAULT (lag: 5초): 진동 증가 → 베어링 마모
    - CURRENT → MOTOR_OVERLOAD (lag: 2초): 전류 증가 → 모터 과부하
    ```
    """
    try:
        agent = get_agent()

        if request.system_prompt is not None:
            agent.set_system_prompt(request.system_prompt)

        if request.domain_knowledge is not None:
            agent.set_domain_knowledge(request.domain_knowledge)

        prompt_info = agent.get_current_prompt()

        return {
            "success": True,
            "message": "프롬프트가 업데이트되었습니다.",
            "is_custom_system_prompt": prompt_info["is_custom_system_prompt"],
            "is_custom_domain_knowledge": prompt_info["is_custom_domain_knowledge"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompt/reset")
async def reset_prompt():
    """
    프롬프트 초기화

    시스템 프롬프트와 도메인 지식을 기본값으로 리셋합니다.
    """
    try:
        agent = get_agent()
        agent.reset_prompt()

        return {
            "success": True,
            "message": "프롬프트가 기본값으로 초기화되었습니다."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/prompt/domain-knowledge")
async def update_domain_knowledge_only(request: dict):
    """
    도메인 지식만 업데이트

    도메인 지식(인과관계, 점검 순서 등)만 변경합니다.

    **Request Body**:
    ```json
    {
      "knowledge": "- RF_POWER → TEMP: RF 파워 증가 → 온도 상승\\n- FLOW → PRESSURE: 유량 변화 → 압력 변화"
    }
    ```
    """
    try:
        agent = get_agent()
        knowledge = request.get("knowledge", "")
        agent.set_domain_knowledge(knowledge)

        return {
            "success": True,
            "message": "도메인 지식이 업데이트되었습니다.",
            "domain_knowledge": knowledge
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
