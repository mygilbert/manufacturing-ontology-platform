"""
Agent Tools
===========

FDC 분석 Agent가 사용하는 도구 모음

도구 종류:
1. OntologySearchTool - 온톨로지 그래프 검색
2. TimeSeriesAnalysisTool - 시계열 데이터 분석
3. RootCauseAnalysisTool - 근본원인 분석
4. AlarmHistoryTool - 알람 이력 조회
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

# 상위 모듈 import를 위한 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    data: Any
    message: str = ""
    execution_time_ms: float = 0


@dataclass
class ToolDefinition:
    """도구 정의 (LLM에게 설명용)"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required
        }


class BaseTool(ABC):
    """도구 기본 클래스"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def get_definition(self) -> ToolDefinition:
        """LLM에게 전달할 도구 정의"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self._get_parameters(),
            required=self._get_required_params()
        )

    def _get_parameters(self) -> Dict:
        return {}

    def _get_required_params(self) -> List[str]:
        return []


class OntologySearchTool(BaseTool):
    """온톨로지 그래프 검색 도구"""

    def __init__(self, sample_data_path: Optional[str] = None):
        self.sample_data_path = sample_data_path
        self._load_sample_ontology()

    def _load_sample_ontology(self):
        """샘플 온톨로지 데이터 로드"""
        # 실제 환경에서는 PostgreSQL + AGE에서 로드
        self.ontology = {
            "equipment": {
                "ETCH-001": {
                    "type": "Equipment",
                    "name": "Etcher 1",
                    "location": "FAB-A",
                    "sensors": ["TEMP_001", "PRESSURE_001", "RF_POWER_001", "FLOW_001"],
                    "processes": ["ETCH_STEP_1", "ETCH_STEP_2"]
                },
                "ETCH-002": {
                    "type": "Equipment",
                    "name": "Etcher 2",
                    "location": "FAB-A",
                    "sensors": ["TEMP_002", "PRESSURE_002", "RF_POWER_002", "FLOW_002"],
                    "processes": ["ETCH_STEP_1", "ETCH_STEP_2"]
                }
            },
            "sensors": {
                "TEMP_001": {"unit": "C", "normal_range": [20, 80], "equipment": "ETCH-001"},
                "PRESSURE_001": {"unit": "mTorr", "normal_range": [10, 100], "equipment": "ETCH-001"},
                "RF_POWER_001": {"unit": "W", "normal_range": [100, 500], "equipment": "ETCH-001"},
                "FLOW_001": {"unit": "sccm", "normal_range": [50, 200], "equipment": "ETCH-001"},
            },
            "relationships": [
                {"source": "TEMP_001", "target": "PRESSURE_001", "type": "INFLUENCES", "lag": 2},
                {"source": "RF_POWER_001", "target": "TEMP_001", "type": "CAUSES", "lag": 1},
                {"source": "FLOW_001", "target": "PRESSURE_001", "type": "INFLUENCES", "lag": 3},
            ],
            "alarms": {
                "ALM_HIGH_TEMP": {"description": "온도 상한 초과", "severity": "HIGH", "threshold": "> 80C"},
                "ALM_LOW_PRESSURE": {"description": "압력 하한 미달", "severity": "MEDIUM", "threshold": "< 10 mTorr"},
                "ALM_RF_FAULT": {"description": "RF 전원 이상", "severity": "CRITICAL", "threshold": "deviation > 20%"},
            }
        }

    @property
    def name(self) -> str:
        return "ontology_search"

    @property
    def description(self) -> str:
        return "온톨로지 그래프에서 설비, 센서, 관계 정보를 검색합니다."

    def _get_parameters(self) -> Dict:
        return {
            "query_type": {
                "type": "string",
                "enum": ["equipment", "sensor", "relationship", "alarm"],
                "description": "검색 대상 유형"
            },
            "entity_id": {
                "type": "string",
                "description": "검색할 엔티티 ID (예: ETCH-001, TEMP_001)"
            },
            "include_related": {
                "type": "boolean",
                "description": "연관 엔티티 포함 여부"
            }
        }

    def _get_required_params(self) -> List[str]:
        return ["query_type"]

    def execute(
        self,
        query_type: str,
        entity_id: Optional[str] = None,
        include_related: bool = True,
        **kwargs
    ) -> ToolResult:
        """온톨로지 검색 실행"""
        start_time = datetime.now()

        try:
            if query_type == "equipment":
                if entity_id:
                    data = self.ontology["equipment"].get(entity_id)
                    if data and include_related:
                        # 관련 센서 정보 추가
                        data["sensor_details"] = {
                            s: self.ontology["sensors"].get(s)
                            for s in data.get("sensors", [])
                        }
                else:
                    data = list(self.ontology["equipment"].keys())

            elif query_type == "sensor":
                if entity_id:
                    data = self.ontology["sensors"].get(entity_id)
                else:
                    data = list(self.ontology["sensors"].keys())

            elif query_type == "relationship":
                if entity_id:
                    # 특정 엔티티와 관련된 관계만
                    data = [
                        r for r in self.ontology["relationships"]
                        if r["source"] == entity_id or r["target"] == entity_id
                    ]
                else:
                    data = self.ontology["relationships"]

            elif query_type == "alarm":
                if entity_id:
                    data = self.ontology["alarms"].get(entity_id)
                else:
                    data = self.ontology["alarms"]

            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Unknown query_type: {query_type}"
                )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return ToolResult(
                success=True,
                data=data,
                message=f"Found {len(data) if isinstance(data, (list, dict)) else 1} results",
                execution_time_ms=execution_time
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                message=f"Search error: {str(e)}"
            )


class TimeSeriesAnalysisTool(BaseTool):
    """시계열 데이터 분석 도구"""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.data = None
        if data_path and PANDAS_AVAILABLE:
            self._load_data()

    def _load_data(self):
        """샘플 데이터 로드"""
        try:
            if os.path.exists(self.data_path):
                self.data = pd.read_csv(self.data_path)
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        except Exception as e:
            print(f"Data loading error: {e}")
            self.data = None

    def _generate_sample_data(self, sensor_id: str, hours: int = 24) -> pd.DataFrame:
        """샘플 시계열 데이터 생성"""
        if not PANDAS_AVAILABLE:
            return None

        np.random.seed(42)
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=hours * 60,  # 분 단위
            freq='1min'
        )

        # 기본 패턴 + 노이즈
        base = 50 + 10 * np.sin(np.linspace(0, 4*np.pi, len(timestamps)))
        noise = np.random.normal(0, 2, len(timestamps))

        # 이상 구간 추가
        anomaly_start = int(len(timestamps) * 0.7)
        anomaly_end = int(len(timestamps) * 0.75)
        base[anomaly_start:anomaly_end] += 20

        return pd.DataFrame({
            'timestamp': timestamps,
            'value': base + noise,
            'sensor_id': sensor_id
        })

    @property
    def name(self) -> str:
        return "time_series_analysis"

    @property
    def description(self) -> str:
        return "센서 시계열 데이터를 분석하여 이상 패턴, 통계, 트렌드를 제공합니다."

    def _get_parameters(self) -> Dict:
        return {
            "sensor_id": {
                "type": "string",
                "description": "분석할 센서 ID"
            },
            "time_range_hours": {
                "type": "integer",
                "description": "분석 시간 범위 (시간 단위)"
            },
            "analysis_type": {
                "type": "string",
                "enum": ["statistics", "anomaly", "trend", "all"],
                "description": "분석 유형"
            }
        }

    def _get_required_params(self) -> List[str]:
        return ["sensor_id"]

    def execute(
        self,
        sensor_id: str,
        time_range_hours: int = 24,
        analysis_type: str = "all",
        **kwargs
    ) -> ToolResult:
        """시계열 분석 실행"""
        if not PANDAS_AVAILABLE:
            return ToolResult(
                success=False,
                data=None,
                message="pandas not available"
            )

        start_time = datetime.now()

        # 데이터 로드 또는 생성
        if self.data is not None and sensor_id in self.data.columns:
            df = self.data[['timestamp', sensor_id]].copy()
            df.columns = ['timestamp', 'value']
        else:
            df = self._generate_sample_data(sensor_id, time_range_hours)

        if df is None or df.empty:
            return ToolResult(
                success=False,
                data=None,
                message=f"No data available for sensor: {sensor_id}"
            )

        result = {"sensor_id": sensor_id}

        # 통계 분석
        if analysis_type in ["statistics", "all"]:
            result["statistics"] = {
                "count": len(df),
                "mean": float(df['value'].mean()),
                "std": float(df['value'].std()),
                "min": float(df['value'].min()),
                "max": float(df['value'].max()),
                "median": float(df['value'].median()),
                "q1": float(df['value'].quantile(0.25)),
                "q3": float(df['value'].quantile(0.75))
            }

        # 이상 탐지
        if analysis_type in ["anomaly", "all"]:
            mean = df['value'].mean()
            std = df['value'].std()
            z_scores = np.abs((df['value'] - mean) / std)
            anomalies = df[z_scores > 3].copy()

            result["anomaly"] = {
                "anomaly_count": len(anomalies),
                "anomaly_ratio": float(len(anomalies) / len(df)),
                "threshold_upper": float(mean + 3 * std),
                "threshold_lower": float(mean - 3 * std),
                "anomaly_timestamps": anomalies['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()[:5]
            }

        # 트렌드 분석
        if analysis_type in ["trend", "all"]:
            # 간단한 선형 트렌드
            x = np.arange(len(df))
            slope, intercept = np.polyfit(x, df['value'], 1)

            result["trend"] = {
                "slope": float(slope),
                "trend_direction": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable",
                "start_value": float(df['value'].iloc[0]),
                "end_value": float(df['value'].iloc[-1]),
                "change_percent": float((df['value'].iloc[-1] - df['value'].iloc[0]) / df['value'].iloc[0] * 100)
            }

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ToolResult(
            success=True,
            data=result,
            message="Analysis completed",
            execution_time_ms=execution_time
        )


class RootCauseAnalysisTool(BaseTool):
    """근본원인 분석 도구"""

    def __init__(self):
        # 인과관계 그래프 (샘플)
        self.causal_graph = {
            "TEMP_001": {
                "causes": ["RF_POWER_001", "FLOW_001"],
                "effects": ["PRESSURE_001", "ALM_HIGH_TEMP"],
                "lag_seconds": {"RF_POWER_001": 1, "FLOW_001": 5}
            },
            "PRESSURE_001": {
                "causes": ["TEMP_001", "FLOW_001"],
                "effects": ["ALM_LOW_PRESSURE"],
                "lag_seconds": {"TEMP_001": 2, "FLOW_001": 3}
            },
            "RF_POWER_001": {
                "causes": [],
                "effects": ["TEMP_001"],
                "lag_seconds": {}
            },
            "FLOW_001": {
                "causes": [],
                "effects": ["TEMP_001", "PRESSURE_001"],
                "lag_seconds": {}
            }
        }

        # 알람별 점검 순서
        self.alarm_checklist = {
            "ALM_HIGH_TEMP": [
                {"param": "RF_POWER_001", "check": "RF 파워 급상승 확인", "priority": 1},
                {"param": "FLOW_001", "check": "냉각수 유량 감소 확인", "priority": 2},
                {"param": "TEMP_001", "check": "온도 센서 캘리브레이션", "priority": 3},
            ],
            "ALM_LOW_PRESSURE": [
                {"param": "FLOW_001", "check": "가스 유량 확인", "priority": 1},
                {"param": "PRESSURE_001", "check": "진공 펌프 상태", "priority": 2},
            ],
            "ALM_RF_FAULT": [
                {"param": "RF_POWER_001", "check": "RF 매칭 상태", "priority": 1},
                {"param": "TEMP_001", "check": "RF 안테나 온도", "priority": 2},
            ]
        }

    @property
    def name(self) -> str:
        return "root_cause_analysis"

    @property
    def description(self) -> str:
        return "알람이나 이상 현상의 근본 원인을 인과관계 그래프를 기반으로 분석합니다."

    def _get_parameters(self) -> Dict:
        return {
            "alarm_code": {
                "type": "string",
                "description": "분석할 알람 코드"
            },
            "target_param": {
                "type": "string",
                "description": "분석할 파라미터 ID"
            },
            "depth": {
                "type": "integer",
                "description": "원인 추적 깊이"
            }
        }

    def _get_required_params(self) -> List[str]:
        return []

    def _trace_causes(self, param: str, depth: int, visited: set) -> List[Dict]:
        """원인 추적 (재귀)"""
        if depth <= 0 or param in visited:
            return []

        visited.add(param)
        causes = []

        if param in self.causal_graph:
            for cause in self.causal_graph[param]["causes"]:
                lag = self.causal_graph[param]["lag_seconds"].get(cause, 0)
                causes.append({
                    "cause": cause,
                    "effect": param,
                    "lag_seconds": lag,
                    "depth": depth
                })
                # 재귀적으로 상위 원인 추적
                causes.extend(self._trace_causes(cause, depth - 1, visited))

        return causes

    def execute(
        self,
        alarm_code: Optional[str] = None,
        target_param: Optional[str] = None,
        depth: int = 3,
        **kwargs
    ) -> ToolResult:
        """근본원인 분석 실행"""
        start_time = datetime.now()

        result = {}

        # 알람 기반 분석
        if alarm_code:
            if alarm_code in self.alarm_checklist:
                result["alarm_code"] = alarm_code
                result["checklist"] = self.alarm_checklist[alarm_code]
                result["primary_suspects"] = [
                    item["param"] for item in self.alarm_checklist[alarm_code][:2]
                ]
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Unknown alarm code: {alarm_code}"
                )

        # 파라미터 기반 분석
        if target_param:
            if target_param in self.causal_graph:
                causes = self._trace_causes(target_param, depth, set())
                result["target_param"] = target_param
                result["causal_chain"] = causes
                result["direct_causes"] = self.causal_graph[target_param]["causes"]
                result["direct_effects"] = self.causal_graph[target_param]["effects"]

                # 근본 원인 (더 이상 상위 원인이 없는 것)
                root_causes = [
                    c["cause"] for c in causes
                    if not self.causal_graph.get(c["cause"], {}).get("causes")
                ]
                result["root_causes"] = list(set(root_causes))
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    message=f"Unknown parameter: {target_param}"
                )

        if not result:
            return ToolResult(
                success=False,
                data=None,
                message="Please provide alarm_code or target_param"
            )

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ToolResult(
            success=True,
            data=result,
            message="Root cause analysis completed",
            execution_time_ms=execution_time
        )


class AlarmHistoryTool(BaseTool):
    """알람 이력 조회 도구"""

    def __init__(self):
        # 샘플 알람 이력
        self.alarm_history = [
            {"timestamp": "2024-12-28 14:30:00", "alarm_code": "ALM_HIGH_TEMP", "equipment": "ETCH-001", "value": 85.2, "status": "ACTIVE"},
            {"timestamp": "2024-12-28 14:25:00", "alarm_code": "ALM_HIGH_TEMP", "equipment": "ETCH-001", "value": 82.1, "status": "CLEARED"},
            {"timestamp": "2024-12-28 10:15:00", "alarm_code": "ALM_RF_FAULT", "equipment": "ETCH-001", "value": None, "status": "CLEARED"},
            {"timestamp": "2024-12-27 16:00:00", "alarm_code": "ALM_LOW_PRESSURE", "equipment": "ETCH-002", "value": 8.5, "status": "CLEARED"},
        ]

    @property
    def name(self) -> str:
        return "alarm_history"

    @property
    def description(self) -> str:
        return "설비의 알람 발생 이력을 조회합니다."

    def _get_parameters(self) -> Dict:
        return {
            "equipment_id": {
                "type": "string",
                "description": "설비 ID"
            },
            "alarm_code": {
                "type": "string",
                "description": "알람 코드 (선택)"
            },
            "hours": {
                "type": "integer",
                "description": "조회 기간 (시간)"
            }
        }

    def _get_required_params(self) -> List[str]:
        return ["equipment_id"]

    def execute(
        self,
        equipment_id: str,
        alarm_code: Optional[str] = None,
        hours: int = 24,
        **kwargs
    ) -> ToolResult:
        """알람 이력 조회"""
        start_time = datetime.now()

        # 필터링
        filtered = [
            a for a in self.alarm_history
            if a["equipment"] == equipment_id
        ]

        if alarm_code:
            filtered = [a for a in filtered if a["alarm_code"] == alarm_code]

        result = {
            "equipment_id": equipment_id,
            "total_alarms": len(filtered),
            "alarms": filtered,
            "active_alarms": [a for a in filtered if a["status"] == "ACTIVE"]
        }

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ToolResult(
            success=True,
            data=result,
            message=f"Found {len(filtered)} alarms",
            execution_time_ms=execution_time
        )


class PatternMiningTool(BaseTool):
    """패턴 마이닝 도구"""

    def __init__(self):
        # 발견된 패턴 데이터 (샘플)
        self.patterns = {
            "TEMP_HIGH": {
                "co_occurs": ["PRESSURE_HIGH", "RF_ADJUST", "COOLANT_LOW"],
                "precedes": ["PARTICLE_ALARM", "ETCH_RATE_OOS"],
                "follows": ["RF_POWER_SPIKE", "FLOW_DROP"],
                "confidence": {
                    "PRESSURE_HIGH": 0.85,
                    "RF_ADJUST": 0.72,
                    "COOLANT_LOW": 0.91,
                    "PARTICLE_ALARM": 0.68,
                    "ETCH_RATE_OOS": 0.79,
                    "RF_POWER_SPIKE": 0.88,
                    "FLOW_DROP": 0.76
                }
            },
            "PRESSURE_HIGH": {
                "co_occurs": ["TEMP_HIGH", "VACUUM_FAULT"],
                "precedes": ["ETCH_RATE_OOS"],
                "follows": ["GAS_FLOW_CHANGE"],
                "confidence": {
                    "TEMP_HIGH": 0.85,
                    "VACUUM_FAULT": 0.67,
                    "ETCH_RATE_OOS": 0.82,
                    "GAS_FLOW_CHANGE": 0.74
                }
            },
            "ALARM_TRIGGER": {
                "co_occurs": ["TEMP_HIGH", "PRESSURE_HIGH", "RF_ADJUST"],
                "precedes": ["EQUIPMENT_STOP", "MAINTENANCE_REQUEST"],
                "follows": ["PARAMETER_DRIFT", "SENSOR_ANOMALY"],
                "confidence": {
                    "TEMP_HIGH": 0.97,
                    "PRESSURE_HIGH": 1.00,
                    "RF_ADJUST": 0.89,
                    "EQUIPMENT_STOP": 0.65,
                    "MAINTENANCE_REQUEST": 0.58,
                    "PARAMETER_DRIFT": 0.92,
                    "SENSOR_ANOMALY": 0.71
                }
            },
            "ETCH_RATE_OOS": {
                "co_occurs": ["PRESSURE_HIGH", "RF_ADJUST", "TEMP_DRIFT"],
                "precedes": ["QUALITY_FAIL", "REWORK"],
                "follows": ["GAS_COMPOSITION_CHANGE", "CHAMBER_CONDITION"],
                "confidence": {
                    "PRESSURE_HIGH": 0.92,
                    "RF_ADJUST": 0.97,
                    "TEMP_DRIFT": 0.84,
                    "QUALITY_FAIL": 0.78,
                    "REWORK": 0.45,
                    "GAS_COMPOSITION_CHANGE": 0.86,
                    "CHAMBER_CONDITION": 0.69
                }
            }
        }

        # 시퀀스 패턴
        self.sequence_patterns = [
            {
                "sequence": ["RF_POWER_SPIKE", "TEMP_HIGH", "PRESSURE_HIGH"],
                "support": 0.15,
                "confidence": 0.89,
                "avg_interval_sec": 30
            },
            {
                "sequence": ["FLOW_DROP", "TEMP_HIGH", "PARTICLE_ALARM"],
                "support": 0.12,
                "confidence": 0.76,
                "avg_interval_sec": 45
            },
            {
                "sequence": ["GAS_FLOW_CHANGE", "PRESSURE_HIGH", "ETCH_RATE_OOS"],
                "support": 0.18,
                "confidence": 0.82,
                "avg_interval_sec": 60
            }
        ]

    @property
    def name(self) -> str:
        return "pattern_mining"

    @property
    def description(self) -> str:
        return "알람 및 이벤트 간의 동시 발생, 선후 관계 패턴을 분석합니다."

    def _get_parameters(self) -> Dict:
        return {
            "event_id": {
                "type": "string",
                "description": "분석할 이벤트/알람 ID (예: TEMP_HIGH, ALARM_TRIGGER)"
            },
            "pattern_type": {
                "type": "string",
                "enum": ["co_occurrence", "sequence", "all"],
                "description": "패턴 유형 (동시발생, 시퀀스, 전체)"
            },
            "min_confidence": {
                "type": "number",
                "description": "최소 신뢰도 (0.0-1.0)"
            }
        }

    def _get_required_params(self) -> List[str]:
        return ["event_id"]

    def execute(
        self,
        event_id: str,
        pattern_type: str = "all",
        min_confidence: float = 0.5,
        related_sensors: Optional[List[str]] = None,
        alarm_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """패턴 마이닝 실행"""
        start_time = datetime.now()

        # alarm_id가 제공되면 event_id로 사용 (호환성)
        if alarm_id and not event_id:
            event_id = alarm_id

        result = {"event_id": event_id, "patterns": {}}

        # 이벤트 패턴 조회
        if event_id in self.patterns:
            pattern_data = self.patterns[event_id]

            if pattern_type in ["co_occurrence", "all"]:
                # 동시 발생 패턴
                co_occurs = []
                for event in pattern_data.get("co_occurs", []):
                    conf = pattern_data["confidence"].get(event, 0)
                    if conf >= min_confidence:
                        co_occurs.append({
                            "event": event,
                            "confidence": conf,
                            "relation": "CO_OCCURS"
                        })
                result["patterns"]["co_occurrence"] = sorted(
                    co_occurs, key=lambda x: x["confidence"], reverse=True
                )

                # 선행 이벤트 (이 이벤트 전에 발생)
                precedes = []
                for event in pattern_data.get("follows", []):
                    conf = pattern_data["confidence"].get(event, 0)
                    if conf >= min_confidence:
                        precedes.append({
                            "event": event,
                            "confidence": conf,
                            "relation": "PRECEDES"
                        })
                result["patterns"]["precedes"] = sorted(
                    precedes, key=lambda x: x["confidence"], reverse=True
                )

                # 후행 이벤트 (이 이벤트 후에 발생)
                follows = []
                for event in pattern_data.get("precedes", []):
                    conf = pattern_data["confidence"].get(event, 0)
                    if conf >= min_confidence:
                        follows.append({
                            "event": event,
                            "confidence": conf,
                            "relation": "FOLLOWS"
                        })
                result["patterns"]["follows"] = sorted(
                    follows, key=lambda x: x["confidence"], reverse=True
                )

            if pattern_type in ["sequence", "all"]:
                # 시퀀스 패턴 (해당 이벤트 포함)
                sequences = [
                    seq for seq in self.sequence_patterns
                    if event_id in seq["sequence"] and seq["confidence"] >= min_confidence
                ]
                result["patterns"]["sequences"] = sequences

            # 요약 통계
            all_related = (
                pattern_data.get("co_occurs", []) +
                pattern_data.get("precedes", []) +
                pattern_data.get("follows", [])
            )
            high_conf_count = sum(
                1 for e in all_related
                if pattern_data["confidence"].get(e, 0) >= 0.8
            )

            result["summary"] = {
                "total_related_events": len(all_related),
                "high_confidence_relations": high_conf_count,
                "top_related": sorted(
                    [(e, pattern_data["confidence"].get(e, 0)) for e in all_related],
                    key=lambda x: x[1], reverse=True
                )[:5]
            }

        else:
            # 알려지지 않은 이벤트 - 관련 센서 기반 분석
            result["patterns"]["note"] = f"No direct patterns found for {event_id}"
            if related_sensors:
                result["patterns"]["related_sensor_patterns"] = []
                for sensor in related_sensors:
                    if sensor in self.patterns:
                        result["patterns"]["related_sensor_patterns"].append({
                            "sensor": sensor,
                            "co_occurs": self.patterns[sensor].get("co_occurs", [])[:3]
                        })

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return ToolResult(
            success=True,
            data=result,
            message=f"Pattern mining completed for {event_id}",
            execution_time_ms=execution_time
        )


class ToolRegistry:
    """도구 레지스트리"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """도구 등록"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """도구 조회"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """등록된 도구 목록"""
        return list(self.tools.keys())

    def get_all_definitions(self) -> List[Dict]:
        """모든 도구 정의 (LLM 프롬프트용)"""
        return [tool.get_definition().to_dict() for tool in self.tools.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """도구 실행"""
        tool = self.get(tool_name)
        if tool:
            return tool.execute(**kwargs)
        return ToolResult(
            success=False,
            data=None,
            message=f"Tool not found: {tool_name}"
        )


def create_default_registry(data_path: Optional[str] = None) -> ToolRegistry:
    """기본 도구 레지스트리 생성"""
    registry = ToolRegistry()
    registry.register(OntologySearchTool())
    registry.register(TimeSeriesAnalysisTool(data_path=data_path))
    registry.register(RootCauseAnalysisTool())
    registry.register(AlarmHistoryTool())
    registry.register(PatternMiningTool())
    return registry
