"""
WebSocket 연결 관리자
실시간 대시보드 및 경보 스트림 관리
"""

import json
import asyncio
from typing import Dict, Set, List, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect


class WebSocketManager:
    """WebSocket 연결 관리자"""

    def __init__(self):
        # 채널별 연결 관리
        self.connections: Dict[str, Set[WebSocket]] = {
            "dashboard": set(),
            "alerts": set()
        }

        # 연결 메타데이터
        self.connection_info: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket, channel: str = "dashboard") -> None:
        """
        WebSocket 연결

        Args:
            websocket: WebSocket 객체
            channel: 채널 이름 (dashboard, alerts)
        """
        await websocket.accept()

        if channel not in self.connections:
            self.connections[channel] = set()

        self.connections[channel].add(websocket)
        self.connection_info[websocket] = {
            "channel": channel,
            "connected_at": datetime.now(),
            "messages_sent": 0
        }

        print(f"[WebSocket] Client connected to '{channel}' channel. "
              f"Total: {len(self.connections[channel])}")

        # 연결 확인 메시지 전송
        await self._send_json(websocket, {
            "type": "connected",
            "channel": channel,
            "timestamp": datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket, channel: Optional[str] = None) -> None:
        """
        WebSocket 연결 해제

        Args:
            websocket: WebSocket 객체
            channel: 채널 이름 (None이면 자동 탐색)
        """
        if channel is None:
            info = self.connection_info.get(websocket, {})
            channel = info.get("channel", "dashboard")

        if channel in self.connections:
            self.connections[channel].discard(websocket)

        if websocket in self.connection_info:
            del self.connection_info[websocket]

        print(f"[WebSocket] Client disconnected from '{channel}' channel. "
              f"Remaining: {len(self.connections.get(channel, set()))}")

    async def broadcast(self, data: Dict[str, Any], channel: str = "dashboard") -> int:
        """
        채널에 메시지 브로드캐스트

        Args:
            data: 전송할 데이터
            channel: 대상 채널

        Returns:
            전송된 클라이언트 수
        """
        if channel not in self.connections:
            return 0

        sent_count = 0
        disconnected = []

        for websocket in self.connections[channel].copy():
            try:
                await self._send_json(websocket, data)
                sent_count += 1

                if websocket in self.connection_info:
                    self.connection_info[websocket]["messages_sent"] += 1

            except Exception as e:
                print(f"[WebSocket] Error sending to client: {e}")
                disconnected.append(websocket)

        # 연결 실패한 클라이언트 제거
        for ws in disconnected:
            self.disconnect(ws, channel)

        return sent_count

    async def broadcast_all(self, data: Dict[str, Any]) -> Dict[str, int]:
        """
        모든 채널에 브로드캐스트

        Returns:
            채널별 전송 수
        """
        results = {}
        for channel in self.connections:
            results[channel] = await self.broadcast(data, channel)
        return results

    async def send_measurement(self, measurement: Dict) -> int:
        """측정값 전송 (dashboard 채널)"""
        data = {
            "type": "measurement",
            "timestamp": datetime.now().isoformat(),
            "data": measurement
        }
        return await self.broadcast(data, "dashboard")

    async def send_detection(self, detection: Dict) -> int:
        """감지 결과 전송 (dashboard 채널)"""
        data = {
            "type": "detection",
            "timestamp": datetime.now().isoformat(),
            "data": detection
        }
        return await self.broadcast(data, "dashboard")

    async def send_alert(self, alert: Dict) -> int:
        """
        경보 전송 (alerts 채널 + dashboard 채널)

        Returns:
            총 전송 수
        """
        data = {
            "type": "alert",
            "timestamp": datetime.now().isoformat(),
            "data": alert
        }

        count = await self.broadcast(data, "alerts")
        count += await self.broadcast(data, "dashboard")
        return count

    async def send_status(self, status: Dict) -> int:
        """시스템 상태 전송"""
        data = {
            "type": "status",
            "timestamp": datetime.now().isoformat(),
            "data": status
        }
        return await self.broadcast(data, "dashboard")

    async def _send_json(self, websocket: WebSocket, data: Dict) -> None:
        """JSON 데이터 전송"""
        await websocket.send_text(json.dumps(data, default=str, ensure_ascii=False))

    def get_connection_count(self, channel: Optional[str] = None) -> Dict[str, int]:
        """연결 수 조회"""
        if channel:
            return {channel: len(self.connections.get(channel, set()))}

        return {ch: len(conns) for ch, conns in self.connections.items()}

    def get_statistics(self) -> Dict:
        """통계 조회"""
        total_messages = sum(
            info.get("messages_sent", 0)
            for info in self.connection_info.values()
        )

        return {
            "connections": self.get_connection_count(),
            "total_connections": sum(len(c) for c in self.connections.values()),
            "total_messages_sent": total_messages
        }


# 전역 WebSocket 관리자 인스턴스
ws_manager = WebSocketManager()
