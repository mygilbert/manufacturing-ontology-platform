"""
온톨로지 API 라우터

설비, 공정, Lot, 웨이퍼 등 온톨로지 객체 CRUD 및 관계 탐색
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field

router = APIRouter()


# ============================================================
# Pydantic 모델 (요청/응답)
# ============================================================

class EquipmentBase(BaseModel):
    """설비 기본 모델"""
    equipment_id: str
    name: str
    equipment_type: str
    status: str = "UNKNOWN"
    location: Optional[str] = None


class EquipmentCreate(EquipmentBase):
    """설비 생성 요청"""
    properties: Optional[Dict[str, Any]] = None


class EquipmentResponse(EquipmentBase):
    """설비 응답"""
    properties: Dict[str, Any] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class LotBase(BaseModel):
    """Lot 기본 모델"""
    lot_id: str
    product_code: str
    quantity: int = 1
    status: str = "CREATED"


class LotResponse(LotBase):
    """Lot 응답"""
    current_step: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wafer_count: int = 0


class MeasurementQuery(BaseModel):
    """측정 데이터 조회 쿼리"""
    equipment_id: Optional[str] = None
    param_id: Optional[str] = None
    lot_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100


class RelationshipResponse(BaseModel):
    """관계 응답"""
    from_type: str
    from_id: str
    relation: str
    to_type: str
    to_id: str
    properties: Dict[str, Any] = {}


class GraphTraversalResponse(BaseModel):
    """그래프 탐색 응답"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


# ============================================================
# 설비 API
# ============================================================

@router.get("/equipment", response_model=List[EquipmentResponse])
async def list_equipment(
    equipment_type: Optional[str] = Query(None, description="설비 타입 필터"),
    status: Optional[str] = Query(None, description="상태 필터"),
    location: Optional[str] = Query(None, description="위치 필터"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """설비 목록 조회"""
    from services.ontology_service import ontology_service

    try:
        equipments = await ontology_service.list_equipment(
            equipment_type=equipment_type,
            status=status,
            location=location,
            limit=limit,
            offset=offset,
        )
        return equipments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/equipment/{equipment_id}", response_model=EquipmentResponse)
async def get_equipment(
    equipment_id: str = Path(..., description="설비 ID"),
):
    """설비 상세 조회"""
    from services.ontology_service import ontology_service

    equipment = await ontology_service.get_equipment(equipment_id)
    if not equipment:
        raise HTTPException(status_code=404, detail=f"Equipment {equipment_id} not found")
    return equipment


@router.post("/equipment", response_model=EquipmentResponse, status_code=201)
async def create_equipment(equipment: EquipmentCreate):
    """설비 생성"""
    from services.ontology_service import ontology_service

    try:
        result = await ontology_service.create_equipment(equipment.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/equipment/{equipment_id}/alarms")
async def get_equipment_alarms(
    equipment_id: str = Path(..., description="설비 ID"),
    severity: Optional[str] = Query(None, description="심각도 필터"),
    since: Optional[datetime] = Query(None, description="시작 시간"),
    limit: int = Query(100, ge=1, le=1000),
):
    """설비 알람 이력 조회"""
    from services.ontology_service import ontology_service

    alarms = await ontology_service.get_equipment_alarms(
        equipment_id=equipment_id,
        severity=severity,
        since=since,
        limit=limit,
    )
    return alarms


@router.get("/equipment/{equipment_id}/measurements")
async def get_equipment_measurements(
    equipment_id: str = Path(..., description="설비 ID"),
    param_id: Optional[str] = Query(None, description="파라미터 ID"),
    since: Optional[datetime] = Query(None, description="시작 시간"),
    until: Optional[datetime] = Query(None, description="종료 시간"),
    limit: int = Query(1000, ge=1, le=10000),
):
    """설비 측정 데이터 조회"""
    from services.ontology_service import ontology_service

    measurements = await ontology_service.get_equipment_measurements(
        equipment_id=equipment_id,
        param_id=param_id,
        since=since,
        until=until,
        limit=limit,
    )
    return measurements


# ============================================================
# Lot API
# ============================================================

@router.get("/lots", response_model=List[LotResponse])
async def list_lots(
    product_code: Optional[str] = Query(None, description="제품 코드"),
    status: Optional[str] = Query(None, description="상태"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Lot 목록 조회"""
    from services.ontology_service import ontology_service

    lots = await ontology_service.list_lots(
        product_code=product_code,
        status=status,
        limit=limit,
        offset=offset,
    )
    return lots


@router.get("/lots/{lot_id}", response_model=LotResponse)
async def get_lot(lot_id: str = Path(..., description="Lot ID")):
    """Lot 상세 조회"""
    from services.ontology_service import ontology_service

    lot = await ontology_service.get_lot(lot_id)
    if not lot:
        raise HTTPException(status_code=404, detail=f"Lot {lot_id} not found")
    return lot


@router.get("/lots/{lot_id}/trace")
async def trace_lot(
    lot_id: str = Path(..., description="Lot ID"),
    depth: int = Query(3, ge=1, le=10, description="탐색 깊이"),
):
    """Lot 이력 추적 (Forward Trace)"""
    from services.ontology_service import ontology_service

    trace = await ontology_service.trace_lot_forward(lot_id, depth)
    return trace


@router.get("/lots/{lot_id}/genealogy")
async def get_lot_genealogy(
    lot_id: str = Path(..., description="Lot ID"),
):
    """Lot 계보 조회 (공정 이력)"""
    from services.ontology_service import ontology_service

    genealogy = await ontology_service.get_lot_genealogy(lot_id)
    return genealogy


# ============================================================
# 그래프 탐색 API
# ============================================================

@router.get("/graph/traverse")
async def traverse_graph(
    start_type: str = Query(..., description="시작 노드 타입"),
    start_id: str = Query(..., description="시작 노드 ID"),
    direction: str = Query("both", regex="^(forward|backward|both)$"),
    depth: int = Query(2, ge=1, le=5),
    relation_types: Optional[str] = Query(None, description="관계 타입 필터 (쉼표 구분)"),
):
    """그래프 탐색"""
    from services.ontology_service import ontology_service

    relations = relation_types.split(",") if relation_types else None

    result = await ontology_service.traverse_graph(
        start_type=start_type,
        start_id=start_id,
        direction=direction,
        depth=depth,
        relation_types=relations,
    )
    return result


@router.get("/graph/path")
async def find_path(
    from_type: str = Query(..., description="시작 노드 타입"),
    from_id: str = Query(..., description="시작 노드 ID"),
    to_type: str = Query(..., description="종료 노드 타입"),
    to_id: str = Query(..., description="종료 노드 ID"),
    max_depth: int = Query(5, ge=1, le=10),
):
    """두 노드 간 경로 탐색"""
    from services.ontology_service import ontology_service

    path = await ontology_service.find_path(
        from_type=from_type,
        from_id=from_id,
        to_type=to_type,
        to_id=to_id,
        max_depth=max_depth,
    )
    return path


@router.get("/graph/neighbors/{node_type}/{node_id}")
async def get_neighbors(
    node_type: str = Path(..., description="노드 타입"),
    node_id: str = Path(..., description="노드 ID"),
    relation_type: Optional[str] = Query(None, description="관계 타입"),
):
    """이웃 노드 조회"""
    from services.ontology_service import ontology_service

    neighbors = await ontology_service.get_neighbors(
        node_type=node_type,
        node_id=node_id,
        relation_type=relation_type,
    )
    return neighbors


# ============================================================
# 관계 API
# ============================================================

@router.post("/relationships")
async def create_relationship(
    from_type: str,
    from_id: str,
    relation: str,
    to_type: str,
    to_id: str,
    properties: Optional[Dict[str, Any]] = None,
):
    """관계 생성"""
    from services.ontology_service import ontology_service

    try:
        result = await ontology_service.create_relationship(
            from_type=from_type,
            from_id=from_id,
            relation=relation,
            to_type=to_type,
            to_id=to_id,
            properties=properties or {},
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/relationships")
async def list_relationships(
    from_type: Optional[str] = None,
    from_id: Optional[str] = None,
    relation: Optional[str] = None,
    to_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """관계 목록 조회"""
    from services.ontology_service import ontology_service

    relationships = await ontology_service.list_relationships(
        from_type=from_type,
        from_id=from_id,
        relation=relation,
        to_type=to_type,
        limit=limit,
    )
    return relationships
