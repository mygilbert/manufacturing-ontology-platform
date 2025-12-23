"""
WebSocket 엔드포인트 라우터
실시간 대시보드 및 경보 스트림
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..websocket_manager import ws_manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """
    대시보드 WebSocket 엔드포인트

    실시간으로 다음 데이터를 수신:
    - measurement: 센서 측정값
    - detection: 이상 감지 결과
    - alert: 경보
    - status: 시스템 상태
    """
    await ws_manager.connect(websocket, "dashboard")
    try:
        while True:
            # 클라이언트로부터의 메시지 대기 (연결 유지)
            data = await websocket.receive_text()
            # 클라이언트 메시지 처리 (필요시)
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "dashboard")
    except Exception as e:
        print(f"[WebSocket] Dashboard error: {e}")
        ws_manager.disconnect(websocket, "dashboard")


@router.websocket("/ws/alerts")
async def alerts_websocket(websocket: WebSocket):
    """
    경보 전용 WebSocket 엔드포인트

    실시간으로 경보만 수신
    """
    await ws_manager.connect(websocket, "alerts")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, "alerts")
    except Exception as e:
        print(f"[WebSocket] Alerts error: {e}")
        ws_manager.disconnect(websocket, "alerts")
