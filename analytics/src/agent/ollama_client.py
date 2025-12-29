"""
Ollama API Client
=================

로컬 Ollama 서버와 통신하는 클라이언트
"""

import json
import requests
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """채팅 메시지"""
    role: str  # system, user, assistant
    content: str


@dataclass
class OllamaResponse:
    """Ollama 응답"""
    content: str
    model: str
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None
    done: bool = True


class OllamaClient:
    """Ollama API 클라이언트"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:8b",
        timeout: int = 120
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> OllamaResponse:
        """텍스트 생성 (단일 프롬프트)"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return OllamaResponse(
                    content=data.get('response', ''),
                    model=data.get('model', self.model),
                    total_duration=data.get('total_duration'),
                    eval_count=data.get('eval_count'),
                    done=data.get('done', True)
                )
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except requests.exceptions.Timeout:
            raise Exception("Ollama request timeout")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama server")

    def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> OllamaResponse:
        """채팅 형식 대화"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                message = data.get('message', {})
                return OllamaResponse(
                    content=message.get('content', ''),
                    model=data.get('model', self.model),
                    total_duration=data.get('total_duration'),
                    eval_count=data.get('eval_count'),
                    done=data.get('done', True)
                )
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except requests.exceptions.Timeout:
            raise Exception("Ollama request timeout")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama server")

    def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """스트리밍 채팅"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True
            )

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    message = data.get('message', {})
                    content = message.get('content', '')
                    if content:
                        yield content
                    if data.get('done', False):
                        break

        except requests.exceptions.Timeout:
            raise Exception("Ollama request timeout")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama server")


# 간편 사용을 위한 기본 클라이언트
_default_client: Optional[OllamaClient] = None


def get_client(model: str = "deepseek-r1:8b") -> OllamaClient:
    """기본 클라이언트 반환"""
    global _default_client
    if _default_client is None or _default_client.model != model:
        _default_client = OllamaClient(model=model)
    return _default_client
