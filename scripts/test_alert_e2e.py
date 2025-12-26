#!/usr/bin/env python3
"""
E2E 알람 테스트 스크립트

백엔드 API를 통해 테스트 알람을 발송하고 WebSocket 수신을 확인합니다.

사용법:
    python scripts/test_alert_e2e.py [--api-url http://localhost:8000]
"""
import asyncio
import argparse
import json
import sys
from datetime import datetime
from typing import Optional

import httpx
import websockets


class AlertTester:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip("/")
        self.ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://")
        self.received_messages = []

    async def check_health(self) -> bool:
        """백엔드 서버 상태 확인"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_url}/health", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ 서버 상태: {data.get('status', 'unknown')}")
                    print(f"   버전: {data.get('version', 'unknown')}")
                    return True
                else:
                    print(f"❌ 서버 응답 오류: {response.status_code}")
                    return False
        except httpx.ConnectError:
            print(f"❌ 서버 연결 실패: {self.api_url}")
            print("   → 서버가 실행 중인지 확인하세요")
            return False
        except Exception as e:
            print(f"❌ 상태 확인 오류: {e}")
            return False

    async def send_test_alarm(
        self,
        severity: str = "WARNING",
        equipment_id: str = "CVD-001",
        message: str = "테스트 알람입니다"
    ) -> bool:
        """테스트 알람 발송"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/api/realtime/broadcast/alert",
                    params={
                        "alert_type": "test",
                        "equipment_id": equipment_id,
                        "severity": severity,
                        "message": message,
                    },
                    timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ 알람 발송 성공")
                    print(f"   수신자 수: {data.get('recipients', 0)}")
                    return True
                else:
                    print(f"❌ 알람 발송 실패: {response.status_code}")
                    print(f"   응답: {response.text}")
                    return False
        except Exception as e:
            print(f"❌ 알람 발송 오류: {e}")
            return False

    async def send_test_anomaly(
        self,
        equipment_id: str = "CVD-001",
        score: float = 0.85,
        severity: str = "WARNING"
    ) -> bool:
        """테스트 이상감지 발송"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_url}/api/realtime/broadcast/anomaly",
                    params={
                        "equipment_id": equipment_id,
                        "anomaly_type": "test",
                        "severity": severity,
                        "score": score,
                        "message": f"테스트 이상감지 (score: {score})"
                    },
                    timeout=10.0
                )

                if response.status_code == 200:
                    print(f"✅ 이상감지 발송 성공")
                    return True
                else:
                    print(f"❌ 이상감지 발송 실패: {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ 이상감지 발송 오류: {e}")
            return False

    async def test_websocket_connection(self, timeout: float = 10.0) -> bool:
        """WebSocket 연결 테스트"""
        client_id = f"test_client_{datetime.now().strftime('%H%M%S')}"
        ws_endpoint = f"{self.ws_url}/api/realtime/ws/{client_id}"

        print(f"\n📡 WebSocket 연결 테스트: {ws_endpoint}")

        try:
            async with websockets.connect(ws_endpoint) as websocket:
                print("✅ WebSocket 연결 성공")

                # 알람 채널 구독
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "channel": "alerts"
                })
                await websocket.send(subscribe_msg)
                print("   → 'alerts' 채널 구독 요청")

                # 구독 확인 대기
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=5.0
                    )
                    data = json.loads(response)
                    if data.get("type") == "subscribed":
                        print(f"   ✅ 채널 구독 완료: {data.get('channel')}")
                except asyncio.TimeoutError:
                    print("   ⚠️ 구독 확인 타임아웃 (무시)")

                # 테스트 알람 발송 및 수신 확인
                print("\n📤 테스트 알람 발송 중...")

                # 별도 태스크로 알람 발송
                asyncio.create_task(self._delayed_send_alarm())

                # 메시지 수신 대기
                print("📥 알람 수신 대기 중...")
                try:
                    while True:
                        response = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=timeout
                        )
                        data = json.loads(response)
                        msg_type = data.get("type")

                        if msg_type == "alert":
                            print(f"\n🔔 알람 수신!")
                            print(f"   Severity: {data.get('severity')}")
                            print(f"   Equipment: {data.get('equipment_id')}")
                            print(f"   Message: {data.get('message')}")
                            print(f"   Time: {data.get('timestamp')}")
                            return True
                        elif msg_type == "anomaly":
                            print(f"\n🔍 이상감지 수신!")
                            print(f"   Score: {data.get('score')}")
                            print(f"   Equipment: {data.get('equipment_id')}")
                            return True
                        else:
                            print(f"   수신: {msg_type}")

                except asyncio.TimeoutError:
                    print(f"❌ 알람 수신 타임아웃 ({timeout}초)")
                    return False

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"❌ WebSocket 연결 끊김: {e}")
            return False
        except Exception as e:
            print(f"❌ WebSocket 오류: {e}")
            return False

    async def _delayed_send_alarm(self):
        """1초 후 알람 발송"""
        await asyncio.sleep(1)
        await self.send_test_alarm(
            severity="CRITICAL",
            equipment_id="CVD-001",
            message="[테스트] Chamber Pressure 이상 감지"
        )


async def run_full_test(api_url: str):
    """전체 E2E 테스트 실행"""
    print("=" * 60)
    print("  Manufacturing Ontology Platform - 알람 E2E 테스트")
    print("=" * 60)
    print(f"\nAPI URL: {api_url}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    tester = AlertTester(api_url)

    # Step 1: 서버 상태 확인
    print("─" * 40)
    print("Step 1: 서버 상태 확인")
    print("─" * 40)
    if not await tester.check_health():
        print("\n⛔ 서버에 연결할 수 없습니다. 테스트 중단.")
        return False

    # Step 2: WebSocket + 알람 테스트
    print("\n" + "─" * 40)
    print("Step 2: WebSocket 연결 및 알람 수신 테스트")
    print("─" * 40)
    ws_ok = await tester.test_websocket_connection()

    # 결과 요약
    print("\n" + "=" * 60)
    print("  테스트 결과")
    print("=" * 60)
    if ws_ok:
        print("✅ E2E 테스트 성공!")
        print("   → 백엔드 → WebSocket → 프론트엔드 흐름 정상")
    else:
        print("❌ E2E 테스트 실패")
        print("   → 로그를 확인하세요")

    return ws_ok


async def demo_mode(api_url: str):
    """데모 모드: 주기적으로 알람 발송"""
    print("=" * 60)
    print("  알람 데모 모드 (Ctrl+C로 종료)")
    print("=" * 60)

    tester = AlertTester(api_url)

    if not await tester.check_health():
        return

    scenarios = [
        ("CRITICAL", "CVD-001", "Chamber Pressure OOS (+3.2σ)"),
        ("WARNING", "ETCH-003", "Temperature Drift 감지"),
        ("INFO", "CVD-002", "PM 주기 도래 알림"),
        ("CRITICAL", "PVD-001", "Target Erosion 임계치 초과"),
        ("WARNING", "LITHO-002", "Focus Error 증가 추세"),
    ]

    idx = 0
    try:
        while True:
            severity, eq_id, msg = scenarios[idx % len(scenarios)]
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 알람 발송: {severity} - {eq_id}")
            await tester.send_test_alarm(severity, eq_id, msg)
            idx += 1
            await asyncio.sleep(5)
    except KeyboardInterrupt:
        print("\n\n데모 종료")


def main():
    parser = argparse.ArgumentParser(description="E2E 알람 테스트")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API 서버 URL (기본값: http://localhost:8000)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="데모 모드 (주기적 알람 발송)"
    )

    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo_mode(args.api_url))
    else:
        success = asyncio.run(run_full_test(args.api_url))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
