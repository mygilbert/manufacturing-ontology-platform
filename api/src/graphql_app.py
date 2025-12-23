"""
GraphQL 애플리케이션 (Strawberry)
"""
import json
import strawberry
from datetime import datetime
from typing import List, Optional

from strawberry.fastapi import GraphQLRouter

from graphql.types import (
    Equipment, Lot, Wafer, Process, Recipe, Alarm,
    Measurement, Anomaly, SPCResult, CapabilityResult,
    Prediction, GraphNode, GraphEdge, GraphResult, PathResult
)


@strawberry.type
class Query:
    """GraphQL 쿼리"""

    # =========================================================
    # 설비
    # =========================================================

    @strawberry.field
    async def equipment(self, equipment_id: str) -> Optional[Equipment]:
        """설비 조회"""
        from services.ontology_service import ontology_service
        data = await ontology_service.get_equipment(equipment_id)
        if data:
            return Equipment(**data)
        return None

    @strawberry.field
    async def equipment_list(
        self,
        equipment_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Equipment]:
        """설비 목록"""
        from services.ontology_service import ontology_service
        data = await ontology_service.list_equipment(
            equipment_type=equipment_type,
            status=status,
            limit=limit,
        )
        return [Equipment(**d) for d in data]

    # =========================================================
    # Lot
    # =========================================================

    @strawberry.field
    async def lot(self, lot_id: str) -> Optional[Lot]:
        """Lot 조회"""
        from services.ontology_service import ontology_service
        data = await ontology_service.get_lot(lot_id)
        if data:
            return Lot(**data)
        return None

    @strawberry.field
    async def lot_list(
        self,
        product_code: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Lot]:
        """Lot 목록"""
        from services.ontology_service import ontology_service
        data = await ontology_service.list_lots(
            product_code=product_code,
            status=status,
            limit=limit,
        )
        return [Lot(**d) for d in data]

    # =========================================================
    # 그래프 탐색
    # =========================================================

    @strawberry.field
    async def traverse(
        self,
        start_type: str,
        start_id: str,
        direction: str = "both",
        depth: int = 2,
    ) -> GraphResult:
        """그래프 탐색"""
        from services.ontology_service import ontology_service
        result = await ontology_service.traverse_graph(
            start_type=start_type,
            start_id=start_id,
            direction=direction,
            depth=depth,
        )

        nodes = [
            GraphNode(
                id=str(n.get('id', '')),
                label=str(n.get('label', '')),
                properties=json.dumps(n),
            )
            for n in result.get('nodes', [])
        ]

        edges = [
            GraphEdge(
                id=str(e.get('id', '')),
                label=str(e.get('label', '')),
                start_id=str(e.get('start_id', '')),
                end_id=str(e.get('end_id', '')),
                properties=json.dumps(e),
            )
            for e in result.get('edges', [])
        ]

        return GraphResult(nodes=nodes, edges=edges)

    @strawberry.field
    async def find_path(
        self,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        max_depth: int = 5,
    ) -> PathResult:
        """경로 탐색"""
        from services.ontology_service import ontology_service
        result = await ontology_service.find_path(
            from_type=from_type,
            from_id=from_id,
            to_type=to_type,
            to_id=to_id,
            max_depth=max_depth,
        )

        nodes = [
            GraphNode(
                id=str(n.get('id', '')),
                label=str(n.get('label', '')),
                properties=json.dumps(n),
            )
            for n in result.get('nodes', [])
        ]

        edges = [
            GraphEdge(
                id=str(e.get('id', '')),
                label=str(e.get('label', '')),
                start_id=str(e.get('start_id', '')),
                end_id=str(e.get('end_id', '')),
                properties=json.dumps(e),
            )
            for e in result.get('edges', [])
        ]

        return PathResult(
            found=result.get('found', False),
            nodes=nodes,
            edges=edges,
            length=result.get('length', 0),
        )

    # =========================================================
    # 분석
    # =========================================================

    @strawberry.field
    async def anomalies(
        self,
        equipment_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[Anomaly]:
        """이상 감지 결과"""
        from services.analytics_service import analytics_service
        data = await analytics_service.list_anomalies(
            equipment_id=equipment_id,
            severity=severity,
            limit=limit,
        )
        return [
            Anomaly(
                anomaly_id=d.get('anomaly_id', ''),
                equipment_id=d.get('equipment_id', ''),
                detected_at=datetime.fromisoformat(d.get('detected_at', datetime.utcnow().isoformat())),
                severity=d.get('severity', 'INFO'),
                anomaly_score=float(d.get('anomaly_score', 0)),
                features=d.get('features'),
                message=d.get('message'),
            )
            for d in data
        ]

    @strawberry.field
    async def spc_analysis(
        self,
        equipment_id: str,
        item_id: str,
        chart_type: str = "individual",
    ) -> Optional[SPCResult]:
        """SPC 분석"""
        from services.analytics_service import analytics_service
        result = await analytics_service.analyze_spc(
            equipment_id=equipment_id,
            item_id=item_id,
            chart_type=chart_type,
        )

        if "error" in result:
            return None

        return SPCResult(
            equipment_id=equipment_id,
            item_id=item_id,
            chart_type=chart_type,
            ucl=result.get('limits', {}).get('ucl', 0),
            cl=result.get('limits', {}).get('cl', 0),
            lcl=result.get('limits', {}).get('lcl', 0),
            cpk=result.get('indices', {}).get('cpk'),
            ooc_count=result.get('analysis', {}).get('ooc_count', 0),
            status=result.get('analysis', {}).get('status', 'NORMAL'),
        )

    @strawberry.field
    async def capability_analysis(
        self,
        equipment_id: str,
        item_id: str,
        usl: float,
        lsl: float,
        target: Optional[float] = None,
    ) -> Optional[CapabilityResult]:
        """공정 능력 분석"""
        from services.analytics_service import analytics_service
        result = await analytics_service.analyze_capability(
            equipment_id=equipment_id,
            item_id=item_id,
            usl=usl,
            lsl=lsl,
            target=target,
        )

        if "error" in result:
            return None

        return CapabilityResult(
            equipment_id=equipment_id,
            item_id=item_id,
            cp=result.get('indices', {}).get('cp', 0),
            cpk=result.get('indices', {}).get('cpk', 0),
            pp=result.get('indices', {}).get('pp'),
            ppk=result.get('indices', {}).get('ppk'),
            ppm_total=result.get('ppm', {}).get('total', 0),
            level=result.get('level', 'UNKNOWN'),
        )

    @strawberry.field
    async def failure_prediction(
        self,
        equipment_id: str,
        horizon_hours: int = 24,
    ) -> Prediction:
        """설비 고장 예측"""
        from services.analytics_service import analytics_service
        result = await analytics_service.predict_failure(
            equipment_id=equipment_id,
            horizon_hours=horizon_hours,
        )

        return Prediction(
            equipment_id=equipment_id,
            prediction_type="failure",
            predicted_at=datetime.fromisoformat(result.get('predicted_at', datetime.utcnow().isoformat())),
            probability=result.get('probability'),
            risk_level=result.get('risk_level'),
        )

    @strawberry.field
    async def quality_prediction(
        self,
        process_id: str,
        lot_id: Optional[str] = None,
    ) -> Prediction:
        """품질 예측"""
        from services.analytics_service import analytics_service
        result = await analytics_service.predict_quality(
            process_id=process_id,
            lot_id=lot_id,
        )

        return Prediction(
            process_id=process_id,
            prediction_type="quality",
            predicted_at=datetime.fromisoformat(result.get('predicted_at', datetime.utcnow().isoformat())),
            predicted_value=result.get('predicted_yield'),
        )


@strawberry.type
class Mutation:
    """GraphQL 뮤테이션"""

    @strawberry.mutation
    async def create_equipment(
        self,
        equipment_id: str,
        name: str,
        equipment_type: str,
        status: str = "UNKNOWN",
        location: Optional[str] = None,
    ) -> Equipment:
        """설비 생성"""
        from services.ontology_service import ontology_service
        data = await ontology_service.create_equipment({
            "equipment_id": equipment_id,
            "name": name,
            "equipment_type": equipment_type,
            "status": status,
            "location": location,
        })
        return Equipment(**data)

    @strawberry.mutation
    async def create_relationship(
        self,
        from_type: str,
        from_id: str,
        relation: str,
        to_type: str,
        to_id: str,
    ) -> bool:
        """관계 생성"""
        from services.ontology_service import ontology_service
        try:
            await ontology_service.create_relationship(
                from_type=from_type,
                from_id=from_id,
                relation=relation,
                to_type=to_type,
                to_id=to_id,
            )
            return True
        except Exception:
            return False

    @strawberry.mutation
    async def train_anomaly_model(
        self,
        equipment_id: str,
        model_type: str,
        feature_names: List[str],
        lookback_days: int = 30,
    ) -> bool:
        """이상 감지 모델 학습"""
        from services.analytics_service import analytics_service
        try:
            await analytics_service.train_anomaly_model(
                equipment_id=equipment_id,
                model_type=model_type,
                feature_names=feature_names,
                lookback_days=lookback_days,
            )
            return True
        except Exception:
            return False


# GraphQL 스키마
schema = strawberry.Schema(query=Query, mutation=Mutation)

# GraphQL 라우터
graphql_app = GraphQLRouter(schema)
