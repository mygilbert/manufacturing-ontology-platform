"""
데이터 모델
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class MeasurementStatus(Enum):
    """측정값 상태"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ALARM = "ALARM"
    OUTLIER = "OUTLIER"


class AlarmSeverity(Enum):
    """알람 심각도"""
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    WARNING = "WARNING"
    INFO = "INFO"


class EquipmentStatus(Enum):
    """설비 상태"""
    RUNNING = "RUNNING"
    IDLE = "IDLE"
    PM = "PM"
    DOWN = "DOWN"
    ENGINEERING = "ENGINEERING"


@dataclass
class FdcMeasurement:
    """FDC 측정값"""
    measurement_id: str
    equipment_id: str
    timestamp: datetime
    param_id: str
    value: float

    chamber_id: Optional[str] = None
    recipe_id: Optional[str] = None
    lot_id: Optional[str] = None
    wafer_id: Optional[str] = None
    slot_no: Optional[int] = None
    param_name: Optional[str] = None
    unit: Optional[str] = None
    usl: Optional[float] = None
    lsl: Optional[float] = None
    target: Optional[float] = None
    status: MeasurementStatus = MeasurementStatus.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "equipment_id": self.equipment_id,
            "chamber_id": self.chamber_id,
            "recipe_id": self.recipe_id,
            "lot_id": self.lot_id,
            "wafer_id": self.wafer_id,
            "slot_no": self.slot_no,
            "param_id": self.param_id,
            "param_name": self.param_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "usl": self.usl,
            "lsl": self.lsl,
            "target": self.target,
            "status": self.status.value,
        }


@dataclass
class EnrichedMeasurement(FdcMeasurement):
    """보강된 측정값 (설비/레시피 정보 포함)"""
    equipment_name: Optional[str] = None
    equipment_type: Optional[str] = None
    equipment_location: Optional[str] = None
    recipe_name: Optional[str] = None
    recipe_version: Optional[str] = None
    process_id: Optional[str] = None
    process_name: Optional[str] = None
    product_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "equipment_name": self.equipment_name,
            "equipment_type": self.equipment_type,
            "equipment_location": self.equipment_location,
            "recipe_name": self.recipe_name,
            "recipe_version": self.recipe_version,
            "process_id": self.process_id,
            "process_name": self.process_name,
            "product_code": self.product_code,
        })
        return base


@dataclass
class SpcMeasurement:
    """SPC 측정값"""
    measurement_id: str
    equipment_id: str
    process_id: str
    item_id: str
    value: float
    timestamp: datetime

    lot_id: Optional[str] = None
    wafer_id: Optional[str] = None
    item_name: Optional[str] = None
    unit: Optional[str] = None
    usl: Optional[float] = None
    lsl: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None
    target: Optional[float] = None
    subgroup_id: Optional[str] = None
    subgroup_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "equipment_id": self.equipment_id,
            "process_id": self.process_id,
            "lot_id": self.lot_id,
            "wafer_id": self.wafer_id,
            "item_id": self.item_id,
            "item_name": self.item_name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "usl": self.usl,
            "lsl": self.lsl,
            "ucl": self.ucl,
            "lcl": self.lcl,
            "target": self.target,
            "subgroup_id": self.subgroup_id,
            "subgroup_size": self.subgroup_size,
        }


@dataclass
class SpcAnalyzedMeasurement(SpcMeasurement):
    """SPC 분석 완료된 측정값"""
    status: str = "NORMAL"
    rule_violations: List[str] = field(default_factory=list)
    x_bar: Optional[float] = None
    range_val: Optional[float] = None
    std_dev: Optional[float] = None
    cp: Optional[float] = None
    cpk: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "status": self.status,
            "rule_violations": self.rule_violations,
            "statistics": {
                "x_bar": self.x_bar,
                "range_val": self.range_val,
                "std_dev": self.std_dev,
                "cp": self.cp,
                "cpk": self.cpk,
            },
        })
        return base


@dataclass
class CepAlert:
    """CEP 알람"""
    alert_id: str
    pattern_id: str
    pattern_name: str
    equipment_id: str
    severity: AlarmSeverity
    message: str
    detected_at: datetime

    lot_id: Optional[str] = None
    process_id: Optional[str] = None
    affected_measurements: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "pattern_id": self.pattern_id,
            "pattern_name": self.pattern_name,
            "equipment_id": self.equipment_id,
            "lot_id": self.lot_id,
            "process_id": self.process_id,
            "severity": self.severity.value,
            "message": self.message,
            "detected_at": self.detected_at.isoformat(),
            "affected_measurements": self.affected_measurements,
            "context": self.context,
            "confidence_score": self.confidence_score,
        }


@dataclass
class AggregatedStats:
    """집계 통계"""
    window_start: datetime
    window_end: datetime
    equipment_id: str
    param_id: str

    count: int = 0
    sum_value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    std_value: Optional[float] = None
    alarm_count: int = 0
    warning_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "equipment_id": self.equipment_id,
            "param_id": self.param_id,
            "count": self.count,
            "sum_value": self.sum_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "avg_value": self.avg_value,
            "std_value": self.std_value,
            "alarm_count": self.alarm_count,
            "warning_count": self.warning_count,
        }
