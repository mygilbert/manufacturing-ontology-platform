"""
GraphQL 타입 정의
"""
import strawberry
from datetime import datetime
from typing import List, Optional, Dict, Any


@strawberry.type
class Equipment:
    """설비"""
    equipment_id: str
    name: str
    equipment_type: str
    status: str
    location: Optional[str] = None
    properties: Optional[str] = None

    @strawberry.field
    async def alarms(
        self,
        severity: Optional[str] = None,
        limit: int = 10,
    ) -> List["Alarm"]:
        """설비 알람"""
        from services.ontology_service import ontology_service
        alarms = await ontology_service.get_equipment_alarms(
            self.equipment_id, severity, limit=limit
        )
        return [Alarm(**a) for a in alarms]

    @strawberry.field
    async def processed_lots(self, limit: int = 10) -> List["Lot"]:
        """처리한 Lot"""
        from services.ontology_service import ontology_service
        neighbors = await ontology_service.get_neighbors(
            "Equipment", self.equipment_id, "PROCESSED_AT"
        )
        lots = []
        for n in neighbors[:limit]:
            if 'lot_id' in n.get('node', {}):
                lots.append(Lot(**n['node']))
        return lots


@strawberry.type
class Lot:
    """Lot"""
    lot_id: str
    product_code: str
    quantity: int = 1
    status: str = "CREATED"
    current_step: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wafer_count: int = 0

    @strawberry.field
    async def wafers(self, limit: int = 25) -> List["Wafer"]:
        """Lot의 웨이퍼"""
        from services.ontology_service import ontology_service
        neighbors = await ontology_service.get_neighbors(
            "Lot", self.lot_id, "CONTAINS"
        )
        wafers = []
        for n in neighbors[:limit]:
            if 'wafer_id' in n.get('node', {}):
                wafers.append(Wafer(**n['node']))
        return wafers

    @strawberry.field
    async def process_history(self) -> List["ProcessStep"]:
        """공정 이력"""
        from services.ontology_service import ontology_service
        genealogy = await ontology_service.get_lot_genealogy(self.lot_id)
        return [
            ProcessStep(
                equipment_id=g['equipment'].get('equipment_id'),
                equipment_name=g['equipment'].get('name'),
                start_time=g['relation'].get('start_time'),
                end_time=g['relation'].get('end_time'),
            )
            for g in genealogy
        ]


@strawberry.type
class Wafer:
    """웨이퍼"""
    wafer_id: str
    lot_id: str
    slot_no: int = 1
    status: str = "IN_PROCESS"

    @strawberry.field
    async def measurements(
        self,
        param_id: Optional[str] = None,
        limit: int = 100,
    ) -> List["Measurement"]:
        """웨이퍼 측정 데이터"""
        # TimescaleDB에서 조회
        return []


@strawberry.type
class Process:
    """공정"""
    process_id: str
    name: str
    sequence: int = 1
    description: Optional[str] = None


@strawberry.type
class Recipe:
    """레시피"""
    recipe_id: str
    name: str
    version: str = "1.0"
    parameters: Optional[str] = None


@strawberry.type
class Alarm:
    """알람"""
    alarm_id: str
    equipment_id: Optional[str] = None
    alarm_code: Optional[str] = None
    severity: str = "INFO"
    message: Optional[str] = None
    occurred_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@strawberry.type
class Measurement:
    """측정값"""
    measurement_id: str
    equipment_id: str
    param_id: str
    value: float
    timestamp: datetime
    status: str = "NORMAL"
    unit: Optional[str] = None


@strawberry.type
class ProcessStep:
    """공정 단계 (이력용)"""
    equipment_id: str
    equipment_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@strawberry.type
class Anomaly:
    """이상 감지 결과"""
    anomaly_id: str
    equipment_id: str
    detected_at: datetime
    severity: str
    anomaly_score: float
    features: Optional[str] = None
    message: Optional[str] = None


@strawberry.type
class SPCResult:
    """SPC 분석 결과"""
    equipment_id: str
    item_id: str
    chart_type: str
    ucl: float
    cl: float
    lcl: float
    cpk: Optional[float] = None
    ooc_count: int = 0
    status: str = "NORMAL"


@strawberry.type
class CapabilityResult:
    """공정 능력 분석 결과"""
    equipment_id: str
    item_id: str
    cp: float
    cpk: float
    pp: Optional[float] = None
    ppk: Optional[float] = None
    ppm_total: float
    level: str


@strawberry.type
class Prediction:
    """예측 결과"""
    equipment_id: Optional[str] = None
    process_id: Optional[str] = None
    prediction_type: str
    predicted_at: datetime
    probability: Optional[float] = None
    predicted_value: Optional[float] = None
    risk_level: Optional[str] = None


@strawberry.type
class GraphNode:
    """그래프 노드"""
    id: str
    label: str
    properties: str  # JSON string


@strawberry.type
class GraphEdge:
    """그래프 엣지"""
    id: str
    label: str
    start_id: str
    end_id: str
    properties: str  # JSON string


@strawberry.type
class GraphResult:
    """그래프 탐색 결과"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]


@strawberry.type
class PathResult:
    """경로 탐색 결과"""
    found: bool
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    length: int
