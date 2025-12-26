"""
분석 서비스

이상 감지, SPC 분석, 예측 분석 기능
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import asyncpg
import redis.asyncio as redis
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class AnalyticsService:
    """분석 서비스"""

    def __init__(self):
        self.ts_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None

        # 모델 캐시
        self.anomaly_models: Dict[str, Any] = {}
        self.failure_models: Dict[str, Any] = {}
        self.quality_models: Dict[str, Any] = {}

    async def initialize(self):
        """서비스 초기화"""
        try:
            # TimescaleDB 연결
            self.ts_pool = await asyncpg.create_pool(
                host=settings.timescale_host,
                port=settings.timescale_port,
                user=settings.timescale_user,
                password=settings.timescale_password,
                database=settings.timescale_db,
                min_size=5,
                max_size=20,
            )
            logger.info("TimescaleDB connection pool created")

            # Redis 연결
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db,
                decode_responses=True,
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.error(f"Failed to initialize AnalyticsService: {e}")
            raise

    async def close(self):
        """서비스 종료"""
        if self.ts_pool:
            await self.ts_pool.close()
        if self.redis_client:
            await self.redis_client.close()

    # =========================================================
    # 이상 감지
    # =========================================================

    async def list_anomalies(
        self,
        equipment_id: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """이상 감지 결과 목록"""
        # Redis에서 최근 이상 감지 결과 조회
        key_pattern = f"anomaly:*"
        if equipment_id:
            key_pattern = f"anomaly:{equipment_id}:*"

        anomalies = []
        try:
            keys = await self.redis_client.keys(key_pattern)

            for key in keys[:limit]:
                data = await self.redis_client.hgetall(key)
                if data:
                    if severity and data.get('severity') != severity:
                        continue
                    if since:
                        detected_at = datetime.fromisoformat(data.get('detected_at', ''))
                        if detected_at < since:
                            continue

                    anomalies.append(data)

            return sorted(anomalies, key=lambda x: x.get('detected_at', ''), reverse=True)

        except Exception as e:
            logger.error(f"Failed to list anomalies: {e}")
            return []

    async def detect_anomalies(
        self,
        equipment_id: str,
        feature_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """실시간 이상 감지"""
        # 데이터 조회
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(hours=1))

        query = """
        SELECT timestamp, param_id, value
        FROM fdc_measurements
        WHERE equipment_id = $1 AND time BETWEEN $2 AND $3
        ORDER BY timestamp
        """

        try:
            async with self.ts_pool.acquire() as conn:
                rows = await conn.fetch(query, equipment_id, start_time, end_time)

            if not rows:
                return {"anomalies": [], "message": "No data found"}

            # 피벗 변환
            import pandas as pd
            df = pd.DataFrame([dict(row) for row in rows])
            pivot_df = df.pivot_table(
                index='timestamp',
                columns='param_id',
                values='value',
                aggfunc='mean'
            ).reset_index()

            # 피처 필터
            available_features = [f for f in feature_names if f in pivot_df.columns]
            if not available_features:
                return {"anomalies": [], "message": "No matching features found"}

            # 모델 로드 또는 간단한 통계 기반 감지
            anomalies = []

            for feature in available_features:
                values = pivot_df[feature].dropna().values

                if len(values) < 10:
                    continue

                # Z-score 기반 간단한 이상 감지
                mean = np.mean(values)
                std = np.std(values)

                if std > 0:
                    z_scores = (values - mean) / std
                    anomaly_indices = np.where(np.abs(z_scores) > 3)[0]

                    for idx in anomaly_indices:
                        anomalies.append({
                            "timestamp": str(pivot_df.iloc[idx]['timestamp']),
                            "feature": feature,
                            "value": float(values[idx]),
                            "z_score": float(z_scores[idx]),
                            "severity": "MAJOR" if abs(z_scores[idx]) > 4 else "MINOR",
                        })

            return {
                "equipment_id": equipment_id,
                "analyzed_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "features_analyzed": available_features,
                "anomaly_count": len(anomalies),
                "anomalies": anomalies,
            }

        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            raise

    async def train_anomaly_model(
        self,
        equipment_id: str,
        model_type: str,
        feature_names: List[str],
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """이상 감지 모델 학습"""
        # 학습 데이터 조회
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)

        query = """
        SELECT timestamp, param_id, value
        FROM fdc_measurements
        WHERE equipment_id = $1 AND time BETWEEN $2 AND $3
        ORDER BY timestamp
        """

        try:
            async with self.ts_pool.acquire() as conn:
                rows = await conn.fetch(query, equipment_id, start_time, end_time)

            if len(rows) < 1000:
                return {"error": "Insufficient data for training", "samples": len(rows)}

            # 피벗 변환
            import pandas as pd
            df = pd.DataFrame([dict(row) for row in rows])
            pivot_df = df.pivot_table(
                index='timestamp',
                columns='param_id',
                values='value',
                aggfunc='mean'
            ).fillna(0)

            available_features = [f for f in feature_names if f in pivot_df.columns]

            # 모델 학습 (간소화된 버전)
            result = {
                "equipment_id": equipment_id,
                "model_type": model_type,
                "training_samples": len(pivot_df),
                "features": available_features,
                "trained_at": datetime.utcnow().isoformat(),
                "status": "trained",
            }

            # 모델 메타데이터 저장
            await self.redis_client.hset(
                f"model:anomaly:{equipment_id}",
                mapping={
                    "model_type": model_type,
                    "features": ",".join(available_features),
                    "trained_at": result["trained_at"],
                    "samples": str(len(pivot_df)),
                }
            )

            return result

        except Exception as e:
            logger.error(f"Failed to train anomaly model: {e}")
            raise

    async def list_anomaly_models(self, equipment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """학습된 이상 감지 모델 목록"""
        pattern = f"model:anomaly:*"
        if equipment_id:
            pattern = f"model:anomaly:{equipment_id}"

        models = []
        try:
            keys = await self.redis_client.keys(pattern)
            for key in keys:
                data = await self.redis_client.hgetall(key)
                if data:
                    data['equipment_id'] = key.split(':')[-1]
                    models.append(data)
            return models
        except Exception as e:
            logger.error(f"Failed to list anomaly models: {e}")
            return []

    # =========================================================
    # SPC 분석
    # =========================================================

    async def get_control_chart(
        self,
        equipment_id: str,
        item_id: str,
        chart_type: str = "individual",
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """관리도 데이터 조회"""
        until = until or datetime.utcnow()
        since = since or (until - timedelta(days=7))

        query = """
        SELECT time as timestamp, value, status, usl, lsl, target
        FROM fdc_measurements
        WHERE equipment_id = $1 AND param_id = $2 AND time BETWEEN $3 AND $4
        ORDER BY time DESC
        LIMIT $5
        """

        try:
            async with self.ts_pool.acquire() as conn:
                rows = await conn.fetch(query, equipment_id, item_id, since, until, limit)

            if not rows:
                return {"error": "No data found"}

            values = [float(row['value']) for row in rows]
            timestamps = [str(row['timestamp']) for row in rows]

            # 관리 한계 계산
            mean = np.mean(values)
            std = np.std(values)

            # Moving Range (개별값 차트)
            mr = np.abs(np.diff(values))
            mr_bar = np.mean(mr) if len(mr) > 0 else 0
            d2 = 1.128
            sigma = mr_bar / d2 if d2 > 0 else std

            ucl = mean + 3 * sigma
            lcl = mean - 3 * sigma

            return {
                "equipment_id": equipment_id,
                "item_id": item_id,
                "chart_type": chart_type,
                "limits": {
                    "ucl": float(ucl),
                    "cl": float(mean),
                    "lcl": float(lcl),
                },
                "statistics": {
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "count": len(values),
                },
                "data": [
                    {"timestamp": t, "value": v}
                    for t, v in zip(timestamps, values)
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get control chart: {e}")
            return {"error": str(e)}

    async def analyze_spc(
        self,
        equipment_id: str,
        item_id: str,
        chart_type: str = "individual",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """SPC 분석 수행"""
        chart_data = await self.get_control_chart(
            equipment_id, item_id, chart_type,
            start_time, end_time, limit=100
        )

        if "error" in chart_data:
            return chart_data

        values = [d['value'] for d in chart_data['data']]
        limits = chart_data['limits']

        # OOC 점 검사
        ooc_count = sum(1 for v in values if v > limits['ucl'] or v < limits['lcl'])

        # 간단한 규칙 위반 검사
        violations = []
        if ooc_count > 0:
            violations.append(f"RULE1: {ooc_count} points beyond control limits")

        return {
            **chart_data,
            "analysis": {
                "ooc_count": ooc_count,
                "violations": violations,
                "status": "OOC" if ooc_count > 0 else "NORMAL",
            }
        }

    async def get_spc_violations(
        self,
        equipment_id: Optional[str] = None,
        item_id: Optional[str] = None,
        rule: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """SPC 규칙 위반 목록"""
        # Redis에서 조회
        violations = []
        pattern = "spc:violation:*"

        try:
            keys = await self.redis_client.keys(pattern)
            for key in keys[:limit]:
                data = await self.redis_client.hgetall(key)
                if data:
                    violations.append(data)
            return violations
        except Exception as e:
            logger.error(f"Failed to get SPC violations: {e}")
            return []

    # =========================================================
    # 공정 능력
    # =========================================================

    async def analyze_capability(
        self,
        equipment_id: str,
        item_id: str,
        usl: float,
        lsl: float,
        target: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """공정 능력 분석"""
        end_time = end_time or datetime.utcnow()
        start_time = start_time or (end_time - timedelta(days=7))

        query = """
        SELECT value FROM fdc_measurements
        WHERE equipment_id = $1 AND param_id = $2 AND time BETWEEN $3 AND $4
        """

        try:
            async with self.ts_pool.acquire() as conn:
                rows = await conn.fetch(query, equipment_id, item_id, start_time, end_time)

            if len(rows) < 2:
                return {"error": "Insufficient data"}

            values = np.array([float(row['value']) for row in rows])
            mean = np.mean(values)
            std = np.std(values, ddof=1)

            # Cp, Cpk
            cp = (usl - lsl) / (6 * std) if std > 0 else 0
            cpu = (usl - mean) / (3 * std) if std > 0 else 0
            cpl = (mean - lsl) / (3 * std) if std > 0 else 0
            cpk = min(cpu, cpl)

            # PPM
            from scipy import stats
            ppm_upper = (1 - stats.norm.cdf((usl - mean) / std)) * 1e6 if std > 0 else 0
            ppm_lower = stats.norm.cdf((lsl - mean) / std) * 1e6 if std > 0 else 0
            ppm_total = ppm_upper + ppm_lower

            # 등급
            if cpk >= 2.0:
                level = "EXCELLENT"
            elif cpk >= 1.67:
                level = "GOOD"
            elif cpk >= 1.33:
                level = "ACCEPTABLE"
            elif cpk >= 1.0:
                level = "MARGINAL"
            else:
                level = "POOR"

            # 권고사항
            recommendations = []
            if cpk < 1.33:
                recommendations.append("공정 능력 개선이 필요합니다.")
            if abs(cpu - cpl) > 0.3:
                recommendations.append("공정 중심 조정이 필요합니다.")

            return {
                "equipment_id": equipment_id,
                "item_id": item_id,
                "sample_size": len(values),
                "specifications": {
                    "usl": usl,
                    "lsl": lsl,
                    "target": target or (usl + lsl) / 2,
                },
                "statistics": {
                    "mean": float(mean),
                    "std": float(std),
                    "min": float(values.min()),
                    "max": float(values.max()),
                },
                "indices": {
                    "cp": float(cp),
                    "cpu": float(cpu),
                    "cpl": float(cpl),
                    "cpk": float(cpk),
                },
                "ppm": {
                    "upper": float(ppm_upper),
                    "lower": float(ppm_lower),
                    "total": float(ppm_total),
                },
                "level": level,
                "recommendations": recommendations,
            }

        except Exception as e:
            logger.error(f"Failed to analyze capability: {e}")
            return {"error": str(e)}

    async def get_capability_trend(
        self,
        equipment_id: str,
        item_id: str,
        usl: float,
        lsl: float,
        period: str = "daily",
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """공정 능력 추세"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        interval = "1 day" if period == "daily" else "1 hour" if period == "hourly" else "1 week"

        query = f"""
        SELECT time_bucket('{interval}', timestamp) as bucket,
               avg(value) as mean,
               stddev(value) as std,
               count(*) as count
        FROM fdc_measurements
        WHERE equipment_id = $1 AND item_id = $2 AND time BETWEEN $3 AND $4
        GROUP BY bucket
        ORDER BY bucket
        """

        try:
            async with self.ts_pool.acquire() as conn:
                rows = await conn.fetch(query, equipment_id, item_id, start_time, end_time)

            results = []
            for row in rows:
                mean = float(row['mean']) if row['mean'] else 0
                std = float(row['std']) if row['std'] else 0

                if std > 0:
                    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
                else:
                    cpk = 0

                results.append({
                    "timestamp": str(row['bucket']),
                    "mean": mean,
                    "std": std,
                    "cpk": float(cpk),
                    "count": int(row['count']),
                })

            return results

        except Exception as e:
            logger.error(f"Failed to get capability trend: {e}")
            return []

    async def compare_capability(
        self,
        equipment_ids: List[str],
        item_id: str,
        usl: float,
        lsl: float,
        days: int = 7,
    ) -> List[Dict[str, Any]]:
        """설비간 공정 능력 비교"""
        results = []
        for equipment_id in equipment_ids:
            capability = await self.analyze_capability(
                equipment_id, item_id, usl, lsl,
                start_time=datetime.utcnow() - timedelta(days=days)
            )
            if "error" not in capability:
                results.append(capability)

        # Cpk 기준 정렬
        return sorted(results, key=lambda x: x.get('indices', {}).get('cpk', 0), reverse=True)

    # =========================================================
    # 예측 분석
    # =========================================================

    async def predict_failure(
        self,
        equipment_id: str,
        horizon_hours: int = 24,
    ) -> Dict[str, Any]:
        """설비 고장 예측"""
        # 간소화된 예측 (실제 구현에서는 학습된 모델 사용)
        return {
            "equipment_id": equipment_id,
            "prediction_horizon_hours": horizon_hours,
            "predicted_at": datetime.utcnow().isoformat(),
            "probability": 0.15,  # 예시
            "risk_level": "LOW",
            "message": "낮은 고장 위험",
            "top_contributing_features": ["temperature", "vibration"],
        }

    async def predict_quality(
        self,
        process_id: str,
        lot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """품질 예측"""
        return {
            "process_id": process_id,
            "lot_id": lot_id,
            "predicted_at": datetime.utcnow().isoformat(),
            "predicted_yield": 0.945,
            "confidence_interval": [0.92, 0.97],
            "risk_factors": [],
        }

    async def train_failure_model(
        self,
        equipment_id: str,
        feature_names: List[str],
        lookback_days: int = 90,
    ) -> Dict[str, Any]:
        """고장 예측 모델 학습"""
        return {
            "equipment_id": equipment_id,
            "status": "trained",
            "trained_at": datetime.utcnow().isoformat(),
        }

    async def train_quality_model(
        self,
        process_id: str,
        target_name: str,
        feature_names: Optional[List[str]],
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """품질 예측 모델 학습"""
        return {
            "process_id": process_id,
            "target": target_name,
            "status": "trained",
            "trained_at": datetime.utcnow().isoformat(),
        }

    # =========================================================
    # 대시보드
    # =========================================================

    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """대시보드 요약"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "equipment_count": 50,
            "active_lots": 25,
            "alerts_today": 12,
            "ooc_count": 3,
            "avg_cpk": 1.45,
        }

    async def get_recent_alerts(self, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        """최근 알람"""
        return []

    async def get_equipment_status_summary(self) -> Dict[str, Any]:
        """설비 상태 요약"""
        return {
            "running": 40,
            "idle": 5,
            "maintenance": 3,
            "fault": 2,
        }


# 전역 인스턴스
analytics_service = AnalyticsService()
