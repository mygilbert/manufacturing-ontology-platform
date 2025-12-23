"""
실시간 이상 감지 경보 시스템 실행 스크립트
"""

import os
import sys
import argparse

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
analytics_dir = os.path.join(script_dir, '..')
sys.path.insert(0, analytics_dir)

def main():
    parser = argparse.ArgumentParser(description='FDC Real-time Alert System')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--data', default=None, help='CSV data file path')
    parser.add_argument('--speed', type=int, default=60, help='Simulation speed (default: 60)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    # 데이터 파일 경로 설정
    if args.data:
        os.environ['FDC_DATA_PATH'] = args.data

    print("=" * 60)
    print("FDC Real-time Alert System")
    print("=" * 60)
    print(f"\nServer: http://{args.host}:{args.port}")
    print(f"Dashboard: http://localhost:{args.port}")
    print(f"Simulation Speed: {args.speed}x")
    if args.data:
        print(f"Data Source: {args.data}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # uvicorn 실행
    import uvicorn
    uvicorn.run(
        "realtime_alert_system.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
