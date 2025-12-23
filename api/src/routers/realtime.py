"""
실시간 API 라우터

WebSocket 기반 실시간 데이터 스트리밍 및 알림
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


# ============================================================
# WebSocket 연결 관리
# ============================================================

class ConnectionManager:
    """WebSocket 연결 관리자"""

    def __init__(self):
        # 활성 연결
        self.active_connections: Dict[str, WebSocket] = {}

        # 구독 정보 {client_id: set of channels}
        self.subscriptions: Dict[str, Set[str]] = {}

        # 채널별 구독자 {channel: set of client_ids}
        self.channel_subscribers: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """클라이언트 연결"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        """클라이언트 연결 해제"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        # 구독 정리
        if client_id in self.subscriptions:
            for channel in self.subscriptions[client_id]:
                if channel in self.channel_subscribers:
                    self.channel_subscribers[channel].discard(client_id)
            del self.subscriptions[client_id]

        logger.info(f"Client {client_id} disconnected. Total: {len(self.active_connections)}")

    def subscribe(self, client_id: str, channel: str):
        """채널 구독"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(channel)

        if channel not in self.channel_subscribers:
            self.channel_subscribers[channel] = set()
        self.channel_subscribers[channel].add(client_id)

        logger.debug(f"Client {client_id} subscribed to {channel}")

    def unsubscribe(self, client_id: str, channel: str):
        """채널 구독 해제"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(channel)

        if channel in self.channel_subscribers:
            self.channel_subscribers[channel].discard(client_id)

    async def send_to_client(self, client_id: str, message: dict):
        """특정 클라이언트에게 메시지 전송"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast_to_channel(self, channel: str, message: dict):
        """채널 구독자에게 브로드캐스트"""
        if channel in self.channel_subscribers:
            disconnected = []
            for client_id in self.channel_subscribers[channel]:
                if client_id in self.active_connections:
                    try:
                        await self.active_connections[client_id].send_json(message)
                    except Exception as e:
                        logger.error(f"Failed to send to {client_id}: {e}")
                        disconnected.append(client_id)

            # 연결 끊긴 클라이언트 정리
            for client_id in disconnected:
                self.disconnect(client_id)

    async def broadcast_all(self, message: dict):
        """모든 클라이언트에게 브로드캐스트"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                disconnected.append(client_id)

        for client_id in disconnected:
            self.disconnect(client_id)

    def get_stats(self) -> dict:
        """연결 통계"""
        return {
            "active_connections": len(self.active_connections),
            "channels": list(self.channel_subscribers.keys()),
            "subscriptions_per_channel": {
                ch: len(subs) for ch, subs in self.channel_subscribers.items()
            },
        }


# 전역 연결 관리자
manager = ConnectionManager()


# ============================================================
# WebSocket 엔드포인트
# ============================================================

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
):
    """
    메인 WebSocket 엔드포인트

    메시지 형식:
    - 구독: {"type": "subscribe", "channel": "alerts"}
    - 구독 해제: {"type": "unsubscribe", "channel": "alerts"}
    - Ping: {"type": "ping"}

    채널:
    - alerts: 실시간 알람
    - measurements.{equipment_id}: 특정 설비 측정값
    - spc.{equipment_id}.{item_id}: SPC 데이터
    - anomalies: 이상 감지 결과
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # 클라이언트 메시지 수신
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "subscribe":
                channel = data.get("channel")
                if channel:
                    manager.subscribe(client_id, channel)
                    await manager.send_to_client(client_id, {
                        "type": "subscribed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            elif message_type == "unsubscribe":
                channel = data.get("channel")
                if channel:
                    manager.unsubscribe(client_id, channel)
                    await manager.send_to_client(client_id, {
                        "type": "unsubscribed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            elif message_type == "ping":
                await manager.send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat(),
                })

            elif message_type == "get_stats":
                stats = manager.get_stats()
                await manager.send_to_client(client_id, {
                    "type": "stats",
                    "data": stats,
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)


@router.websocket("/ws/equipment/{equipment_id}")
async def equipment_stream(
    websocket: WebSocket,
    equipment_id: str,
):
    """특정 설비 실시간 데이터 스트림"""
    client_id = f"equipment_{equipment_id}_{id(websocket)}"
    await manager.connect(websocket, client_id)

    # 자동 구독
    channel = f"measurements.{equipment_id}"
    manager.subscribe(client_id, channel)
    manager.subscribe(client_id, f"alerts.{equipment_id}")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "equipment_id": equipment_id,
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(client_id)


# ============================================================
# 브로드캐스트 API (내부용)
# ============================================================

@router.post("/broadcast/alert")
async def broadcast_alert(
    alert_type: str,
    equipment_id: Optional[str] = None,
    severity: str = "INFO",
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
):
    """알람 브로드캐스트 (내부 API)"""
    alert_message = {
        "type": "alert",
        "alert_type": alert_type,
        "equipment_id": equipment_id,
        "severity": severity,
        "message": message,
        "data": data or {},
        "timestamp": datetime.utcnow().isoformat(),
    }

    # 전체 알람 채널
    await manager.broadcast_to_channel("alerts", alert_message)

    # 설비별 알람 채널
    if equipment_id:
        await manager.broadcast_to_channel(f"alerts.{equipment_id}", alert_message)

    return {"status": "sent", "recipients": len(manager.channel_subscribers.get("alerts", set()))}


@router.post("/broadcast/measurement")
async def broadcast_measurement(
    equipment_id: str,
    param_id: str,
    value: float,
    status: str = "NORMAL",
    timestamp: Optional[str] = None,
):
    """측정값 브로드캐스트 (내부 API)"""
    measurement_message = {
        "type": "measurement",
        "equipment_id": equipment_id,
        "param_id": param_id,
        "value": value,
        "status": status,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
    }

    channel = f"measurements.{equipment_id}"
    await manager.broadcast_to_channel(channel, measurement_message)

    return {"status": "sent", "channel": channel}


@router.post("/broadcast/spc")
async def broadcast_spc_update(
    equipment_id: str,
    item_id: str,
    value: float,
    ucl: float,
    cl: float,
    lcl: float,
    status: str = "NORMAL",
    violations: Optional[List[str]] = None,
):
    """SPC 업데이트 브로드캐스트 (내부 API)"""
    spc_message = {
        "type": "spc_update",
        "equipment_id": equipment_id,
        "item_id": item_id,
        "value": value,
        "ucl": ucl,
        "cl": cl,
        "lcl": lcl,
        "status": status,
        "violations": violations or [],
        "timestamp": datetime.utcnow().isoformat(),
    }

    channel = f"spc.{equipment_id}.{item_id}"
    await manager.broadcast_to_channel(channel, spc_message)

    return {"status": "sent", "channel": channel}


@router.post("/broadcast/anomaly")
async def broadcast_anomaly(
    equipment_id: str,
    anomaly_type: str,
    severity: str,
    score: float,
    features: Optional[Dict[str, float]] = None,
    message: Optional[str] = None,
):
    """이상 감지 브로드캐스트 (내부 API)"""
    anomaly_message = {
        "type": "anomaly",
        "equipment_id": equipment_id,
        "anomaly_type": anomaly_type,
        "severity": severity,
        "score": score,
        "features": features or {},
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    await manager.broadcast_to_channel("anomalies", anomaly_message)
    await manager.broadcast_to_channel(f"alerts.{equipment_id}", anomaly_message)

    return {"status": "sent"}


# ============================================================
# 연결 상태 API
# ============================================================

@router.get("/connections")
async def get_connections():
    """현재 WebSocket 연결 상태"""
    return manager.get_stats()


@router.get("/channels")
async def get_channels():
    """사용 가능한 채널 목록"""
    return {
        "channels": [
            {
                "name": "alerts",
                "description": "실시간 알람 (전체)",
                "pattern": "alerts",
            },
            {
                "name": "alerts.{equipment_id}",
                "description": "특정 설비 알람",
                "pattern": "alerts.EQP001",
            },
            {
                "name": "measurements.{equipment_id}",
                "description": "특정 설비 측정값",
                "pattern": "measurements.EQP001",
            },
            {
                "name": "spc.{equipment_id}.{item_id}",
                "description": "SPC 관리도 데이터",
                "pattern": "spc.EQP001.TEMP001",
            },
            {
                "name": "anomalies",
                "description": "이상 감지 결과",
                "pattern": "anomalies",
            },
        ]
    }
