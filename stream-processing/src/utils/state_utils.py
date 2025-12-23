"""
상태 관리 유틸리티 (Redis 기반)
"""
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import redis

from config import redis_config

logger = logging.getLogger(__name__)


class StateManager:
    """Redis 기반 상태 관리자"""

    def __init__(self):
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        """Redis 클라이언트 (지연 초기화)"""
        if self._client is None:
            self._client = redis.Redis(
                host=redis_config.host,
                port=redis_config.port,
                password=redis_config.password,
                db=redis_config.db,
                decode_responses=True,
            )
        return self._client

    # =========================================================
    # 설비 정보 캐시
    # =========================================================

    def get_equipment(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """설비 정보 조회"""
        key = f"equipment:{equipment_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set_equipment(self, equipment_id: str, data: Dict[str, Any], ttl: int = 3600):
        """설비 정보 저장"""
        key = f"equipment:{equipment_id}"
        self.client.setex(key, ttl, json.dumps(data))

    # =========================================================
    # 레시피 정보 캐시
    # =========================================================

    def get_recipe(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        """레시피 정보 조회"""
        key = f"recipe:{recipe_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set_recipe(self, recipe_id: str, data: Dict[str, Any], ttl: int = 3600):
        """레시피 정보 저장"""
        key = f"recipe:{recipe_id}"
        self.client.setex(key, ttl, json.dumps(data))

    # =========================================================
    # Lot 정보 캐시
    # =========================================================

    def get_lot(self, lot_id: str) -> Optional[Dict[str, Any]]:
        """Lot 정보 조회"""
        key = f"lot:{lot_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set_lot(self, lot_id: str, data: Dict[str, Any], ttl: int = 86400):
        """Lot 정보 저장 (24시간)"""
        key = f"lot:{lot_id}"
        self.client.setex(key, ttl, json.dumps(data))

    # =========================================================
    # SPC 히스토리 (관리도용)
    # =========================================================

    def add_spc_value(
        self,
        equipment_id: str,
        item_id: str,
        value: float,
        timestamp: datetime,
        max_size: int = 25,
    ):
        """SPC 값 추가 (FIFO)"""
        key = f"spc:history:{equipment_id}:{item_id}"

        # 값과 타임스탬프를 함께 저장
        entry = json.dumps({"value": value, "timestamp": timestamp.isoformat()})

        # 리스트 끝에 추가
        self.client.rpush(key, entry)

        # 최대 크기 유지
        self.client.ltrim(key, -max_size, -1)

        # TTL 설정 (24시간)
        self.client.expire(key, 86400)

    def get_spc_history(
        self,
        equipment_id: str,
        item_id: str,
        count: int = 25,
    ) -> List[Dict[str, Any]]:
        """SPC 히스토리 조회"""
        key = f"spc:history:{equipment_id}:{item_id}"
        entries = self.client.lrange(key, -count, -1)
        return [json.loads(e) for e in entries]

    def get_spc_values(
        self,
        equipment_id: str,
        item_id: str,
        count: int = 25,
    ) -> List[float]:
        """SPC 값만 조회"""
        history = self.get_spc_history(equipment_id, item_id, count)
        return [h["value"] for h in history]

    # =========================================================
    # CEP 이벤트 윈도우
    # =========================================================

    def add_cep_event(
        self,
        pattern_key: str,
        event: Dict[str, Any],
        window_ms: int,
    ):
        """CEP 이벤트 추가 (시간 윈도우)"""
        key = f"cep:events:{pattern_key}"
        now = datetime.utcnow()

        # 이벤트 저장 (score = timestamp)
        self.client.zadd(
            key,
            {json.dumps(event): now.timestamp() * 1000},
        )

        # 윈도우 밖의 오래된 이벤트 제거
        cutoff = (now.timestamp() * 1000) - window_ms
        self.client.zremrangebyscore(key, "-inf", cutoff)

        # TTL 설정
        self.client.expire(key, int(window_ms / 1000) + 60)

    def get_cep_events(
        self,
        pattern_key: str,
        window_ms: int,
    ) -> List[Dict[str, Any]]:
        """CEP 윈도우 내 이벤트 조회"""
        key = f"cep:events:{pattern_key}"
        now = datetime.utcnow()
        cutoff = (now.timestamp() * 1000) - window_ms

        entries = self.client.zrangebyscore(key, cutoff, "+inf")
        return [json.loads(e) for e in entries]

    def count_cep_events(
        self,
        pattern_key: str,
        window_ms: int,
    ) -> int:
        """CEP 윈도우 내 이벤트 수"""
        key = f"cep:events:{pattern_key}"
        now = datetime.utcnow()
        cutoff = (now.timestamp() * 1000) - window_ms

        return self.client.zcount(key, cutoff, "+inf")

    # =========================================================
    # 통계 집계
    # =========================================================

    def increment_stats(
        self,
        equipment_id: str,
        param_id: str,
        window_key: str,
        value: float,
        is_alarm: bool = False,
        is_warning: bool = False,
    ):
        """윈도우 통계 증분"""
        key = f"stats:{window_key}:{equipment_id}:{param_id}"

        pipe = self.client.pipeline()

        # 기본 통계
        pipe.hincrby(key, "count", 1)
        pipe.hincrbyfloat(key, "sum", value)

        # 알람/경고 카운트
        if is_alarm:
            pipe.hincrby(key, "alarm_count", 1)
        if is_warning:
            pipe.hincrby(key, "warning_count", 1)

        # Min/Max (간단 구현)
        pipe.execute()

        # 별도로 min/max 업데이트
        current = self.client.hgetall(key)
        current_min = float(current.get("min", float("inf")))
        current_max = float(current.get("max", float("-inf")))

        if value < current_min:
            self.client.hset(key, "min", value)
        if value > current_max:
            self.client.hset(key, "max", value)

        # TTL (윈도우 크기 + 여유)
        self.client.expire(key, 600)

    def get_stats(
        self,
        equipment_id: str,
        param_id: str,
        window_key: str,
    ) -> Optional[Dict[str, Any]]:
        """윈도우 통계 조회"""
        key = f"stats:{window_key}:{equipment_id}:{param_id}"
        data = self.client.hgetall(key)

        if not data:
            return None

        count = int(data.get("count", 0))
        sum_val = float(data.get("sum", 0))

        return {
            "count": count,
            "sum": sum_val,
            "min": float(data.get("min", 0)),
            "max": float(data.get("max", 0)),
            "avg": sum_val / count if count > 0 else 0,
            "alarm_count": int(data.get("alarm_count", 0)),
            "warning_count": int(data.get("warning_count", 0)),
        }


# 전역 인스턴스
state_manager = StateManager()
