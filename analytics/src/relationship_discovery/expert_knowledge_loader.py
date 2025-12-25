"""
Expert Knowledge Loader
=======================

엑셀 템플릿에서 전문가 지식을 로드하여 시스템에 적용

사용법:
    loader = ExpertKnowledgeLoader('expert_knowledge.xlsx')
    knowledge = loader.load_all()

    # 검증에 활용
    validator = RelationshipValidator(knowledge)
    is_valid = validator.validate(discovered_relationship)

    # Agent 프롬프트 생성
    prompt = loader.generate_agent_prompt()
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings


@dataclass
class CausalRelationship:
    """인과관계"""
    source: str
    target: str
    relation_type: str  # CAUSES, INFLUENCES, CORRELATES, INHIBITS
    direction: str      # positive, negative, complex
    lag_min: float
    lag_max: float
    confidence: float
    physics_explanation: str
    note: str = ""

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'relation_type': self.relation_type,
            'direction': self.direction,
            'lag_range': [self.lag_min, self.lag_max],
            'confidence': self.confidence,
            'physics': self.physics_explanation,
        }


@dataclass
class AlarmCause:
    """알람 원인"""
    alarm_code: str
    alarm_description: str
    cause_parameter: str
    condition: str
    probability: float
    action: str
    priority: int


@dataclass
class LeadingIndicator:
    """선행 지표"""
    target_event: str
    indicator_parameter: str
    pattern_type: str
    lead_time_seconds: float
    threshold: str
    confidence: float
    description: str


@dataclass
class ParameterGroup:
    """파라미터 그룹"""
    name: str
    description: str
    parameters: List[str]
    note: str = ""


@dataclass
class ImpossibleRelationship:
    """불가능한 관계"""
    param1: str
    param2: str
    reason: str
    note: str = ""


@dataclass
class ParameterDefinition:
    """파라미터 정의"""
    param_id: str
    name_ko: str
    unit: str
    normal_min: float
    normal_max: float
    spec_lsl: float
    spec_usl: float
    data_source: str
    description: str


@dataclass
class ExpertKnowledge:
    """전문가 지식 통합 모델"""
    causal_relationships: List[CausalRelationship] = field(default_factory=list)
    alarm_causes: List[AlarmCause] = field(default_factory=list)
    leading_indicators: List[LeadingIndicator] = field(default_factory=list)
    parameter_groups: List[ParameterGroup] = field(default_factory=list)
    impossible_relationships: List[ImpossibleRelationship] = field(default_factory=list)
    parameter_definitions: List[ParameterDefinition] = field(default_factory=list)
    loaded_at: datetime = field(default_factory=datetime.now)
    source_file: str = ""

    def get_causes_for(self, target: str) -> List[CausalRelationship]:
        """특정 파라미터에 영향을 주는 원인들"""
        return [r for r in self.causal_relationships if r.target == target]

    def get_effects_of(self, source: str) -> List[CausalRelationship]:
        """특정 파라미터가 영향을 주는 대상들"""
        return [r for r in self.causal_relationships if r.source == source]

    def get_alarm_causes(self, alarm_code: str) -> List[AlarmCause]:
        """특정 알람의 원인들"""
        causes = [c for c in self.alarm_causes if c.alarm_code == alarm_code]
        return sorted(causes, key=lambda x: x.priority)

    def get_leading_indicators_for(self, event: str) -> List[LeadingIndicator]:
        """특정 이벤트의 선행 지표들"""
        return [i for i in self.leading_indicators if i.target_event == event]

    def is_impossible_relationship(self, param1: str, param2: str) -> bool:
        """불가능한 관계인지 확인"""
        for imp in self.impossible_relationships:
            if (imp.param1 == param1 and imp.param2 == param2) or \
               (imp.param1 == param2 and imp.param2 == param1):
                return True
        return False

    def get_parameter_group(self, param: str) -> Optional[ParameterGroup]:
        """파라미터가 속한 그룹"""
        for group in self.parameter_groups:
            if param in group.parameters:
                return group
        return None

    def summary(self) -> Dict[str, int]:
        return {
            'causal_relationships': len(self.causal_relationships),
            'alarm_causes': len(self.alarm_causes),
            'leading_indicators': len(self.leading_indicators),
            'parameter_groups': len(self.parameter_groups),
            'impossible_relationships': len(self.impossible_relationships),
            'parameter_definitions': len(self.parameter_definitions),
        }


class ExpertKnowledgeLoader:
    """엑셀에서 전문가 지식 로드"""

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.knowledge = ExpertKnowledge()

    def load_all(self) -> ExpertKnowledge:
        """모든 시트 로드"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas가 필요합니다: pip install pandas openpyxl")

        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.excel_path}")

        print(f"전문가 지식 로딩: {self.excel_path}")

        # 각 시트 로드
        self._load_causal_relationships(pd)
        self._load_alarm_causes(pd)
        self._load_leading_indicators(pd)
        self._load_parameter_groups(pd)
        self._load_impossible_relationships(pd)
        self._load_parameter_definitions(pd)

        self.knowledge.source_file = self.excel_path
        self.knowledge.loaded_at = datetime.now()

        # 요약 출력
        summary = self.knowledge.summary()
        print(f"로딩 완료:")
        for key, value in summary.items():
            print(f"  - {key}: {value}개")

        return self.knowledge

    def _load_causal_relationships(self, pd) -> None:
        """인과관계 로드"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='1.인과관계', skiprows=0)
            df = df.dropna(subset=[df.columns[1], df.columns[2]])  # 필수 컬럼

            for _, row in df.iterrows():
                try:
                    rel = CausalRelationship(
                        source=str(row.iloc[1]).strip(),
                        target=str(row.iloc[2]).strip(),
                        relation_type=str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else "CAUSES",
                        direction=str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else "positive",
                        lag_min=float(row.iloc[5]) if pd.notna(row.iloc[5]) else 0,
                        lag_max=float(row.iloc[6]) if pd.notna(row.iloc[6]) else 0,
                        confidence=float(row.iloc[7]) if pd.notna(row.iloc[7]) else 0.5,
                        physics_explanation=str(row.iloc[8]) if pd.notna(row.iloc[8]) else "",
                        note=str(row.iloc[9]) if pd.notna(row.iloc[9]) else "",
                    )
                    if rel.source and rel.target:
                        self.knowledge.causal_relationships.append(rel)
                except Exception as e:
                    continue
        except Exception as e:
            warnings.warn(f"인과관계 시트 로드 실패: {e}")

    def _load_alarm_causes(self, pd) -> None:
        """알람 원인 로드"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='2.알람원인', skiprows=0)
            df = df.dropna(subset=[df.columns[1], df.columns[3]])

            for _, row in df.iterrows():
                try:
                    cause = AlarmCause(
                        alarm_code=str(row.iloc[1]).strip(),
                        alarm_description=str(row.iloc[2]) if pd.notna(row.iloc[2]) else "",
                        cause_parameter=str(row.iloc[3]).strip(),
                        condition=str(row.iloc[4]) if pd.notna(row.iloc[4]) else "",
                        probability=float(row.iloc[5]) if pd.notna(row.iloc[5]) else 0.5,
                        action=str(row.iloc[6]) if pd.notna(row.iloc[6]) else "",
                        priority=int(row.iloc[7]) if pd.notna(row.iloc[7]) else 99,
                    )
                    if cause.alarm_code and cause.cause_parameter:
                        self.knowledge.alarm_causes.append(cause)
                except Exception as e:
                    continue
        except Exception as e:
            warnings.warn(f"알람원인 시트 로드 실패: {e}")

    def _load_leading_indicators(self, pd) -> None:
        """선행지표 로드"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='3.선행지표', skiprows=0)
            df = df.dropna(subset=[df.columns[1], df.columns[2]])

            for _, row in df.iterrows():
                try:
                    indicator = LeadingIndicator(
                        target_event=str(row.iloc[1]).strip(),
                        indicator_parameter=str(row.iloc[2]).strip(),
                        pattern_type=str(row.iloc[3]) if pd.notna(row.iloc[3]) else "",
                        lead_time_seconds=float(row.iloc[4]) if pd.notna(row.iloc[4]) else 0,
                        threshold=str(row.iloc[5]) if pd.notna(row.iloc[5]) else "",
                        confidence=float(row.iloc[6]) if pd.notna(row.iloc[6]) else 0.5,
                        description=str(row.iloc[7]) if pd.notna(row.iloc[7]) else "",
                    )
                    if indicator.target_event and indicator.indicator_parameter:
                        self.knowledge.leading_indicators.append(indicator)
                except Exception as e:
                    continue
        except Exception as e:
            warnings.warn(f"선행지표 시트 로드 실패: {e}")

    def _load_parameter_groups(self, pd) -> None:
        """파라미터 그룹 로드"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='4.파라미터그룹', skiprows=0)
            df = df.dropna(subset=[df.columns[1]])

            for _, row in df.iterrows():
                try:
                    params = []
                    for i in range(3, 8):  # D~H 컬럼
                        if pd.notna(row.iloc[i]) and str(row.iloc[i]).strip():
                            params.append(str(row.iloc[i]).strip())

                    group = ParameterGroup(
                        name=str(row.iloc[1]).strip(),
                        description=str(row.iloc[2]) if pd.notna(row.iloc[2]) else "",
                        parameters=params,
                        note=str(row.iloc[8]) if pd.notna(row.iloc[8]) else "",
                    )
                    if group.name and group.parameters:
                        self.knowledge.parameter_groups.append(group)
                except Exception as e:
                    continue
        except Exception as e:
            warnings.warn(f"파라미터그룹 시트 로드 실패: {e}")

    def _load_impossible_relationships(self, pd) -> None:
        """불가능한 관계 로드"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='5.불가능한관계', skiprows=0)
            df = df.dropna(subset=[df.columns[1], df.columns[2]])

            for _, row in df.iterrows():
                try:
                    imp = ImpossibleRelationship(
                        param1=str(row.iloc[1]).strip(),
                        param2=str(row.iloc[2]).strip(),
                        reason=str(row.iloc[3]) if pd.notna(row.iloc[3]) else "",
                        note=str(row.iloc[4]) if pd.notna(row.iloc[4]) else "",
                    )
                    if imp.param1 and imp.param2:
                        self.knowledge.impossible_relationships.append(imp)
                except Exception as e:
                    continue
        except Exception as e:
            warnings.warn(f"불가능한관계 시트 로드 실패: {e}")

    def _load_parameter_definitions(self, pd) -> None:
        """파라미터 정의 로드"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='6.파라미터정의', skiprows=0)
            df = df.dropna(subset=[df.columns[1]])

            for _, row in df.iterrows():
                try:
                    param = ParameterDefinition(
                        param_id=str(row.iloc[1]).strip(),
                        name_ko=str(row.iloc[2]) if pd.notna(row.iloc[2]) else "",
                        unit=str(row.iloc[3]) if pd.notna(row.iloc[3]) else "",
                        normal_min=float(row.iloc[4]) if pd.notna(row.iloc[4]) else 0,
                        normal_max=float(row.iloc[5]) if pd.notna(row.iloc[5]) else 0,
                        spec_lsl=float(row.iloc[6]) if pd.notna(row.iloc[6]) else 0,
                        spec_usl=float(row.iloc[7]) if pd.notna(row.iloc[7]) else 0,
                        data_source=str(row.iloc[8]) if pd.notna(row.iloc[8]) else "",
                        description=str(row.iloc[9]) if pd.notna(row.iloc[9]) else "",
                    )
                    if param.param_id:
                        self.knowledge.parameter_definitions.append(param)
                except Exception as e:
                    continue
        except Exception as e:
            warnings.warn(f"파라미터정의 시트 로드 실패: {e}")

    def generate_agent_prompt(self) -> str:
        """Agent 시스템 프롬프트 생성"""
        k = self.knowledge

        # 인과관계 섹션
        causal_section = "## 검증된 인과관계\n"
        for rel in k.causal_relationships:
            direction = "↑" if rel.direction == "positive" else "↓" if rel.direction == "negative" else "~"
            causal_section += f"- {rel.source} → {rel.target} ({direction}, lag={rel.lag_min}~{rel.lag_max}초)\n"
            if rel.physics_explanation:
                causal_section += f"  이유: {rel.physics_explanation}\n"

        # 알람 원인 섹션
        alarm_section = "\n## 알람별 점검 순서\n"
        alarm_codes = set(c.alarm_code for c in k.alarm_causes)
        for code in alarm_codes:
            causes = k.get_alarm_causes(code)
            alarm_section += f"\n### {code}\n"
            for c in causes:
                alarm_section += f"{c.priority}. {c.cause_parameter}: {c.condition} (확률 {c.probability*100:.0f}%)\n"
                if c.action:
                    alarm_section += f"   → 조치: {c.action}\n"

        # 선행지표 섹션
        indicator_section = "\n## 이상 예측 선행 지표\n"
        for ind in k.leading_indicators:
            indicator_section += f"- {ind.target_event}: {ind.indicator_parameter}의 {ind.pattern_type} " \
                                f"({ind.lead_time_seconds}초 전 감지)\n"

        # 금지 관계 섹션
        forbidden_section = "\n## 추론 금지 관계\n"
        forbidden_section += "다음 관계는 물리적으로 불가능하므로 인과관계로 해석하지 마세요:\n"
        for imp in k.impossible_relationships:
            forbidden_section += f"- {imp.param1} ↔ {imp.param2}: {imp.reason}\n"

        # 파라미터 그룹 섹션
        group_section = "\n## 함께 분석할 파라미터 그룹\n"
        for group in k.parameter_groups:
            group_section += f"- {group.name}: {', '.join(group.parameters)}\n"
            if group.note:
                group_section += f"  주의: {group.note}\n"

        prompt = f"""당신은 반도체 FDC(Fault Detection & Classification) 전문가 Agent입니다.
아래의 검증된 도메인 지식을 기반으로 분석하세요.

{causal_section}
{alarm_section}
{indicator_section}
{forbidden_section}
{group_section}

## 추론 규칙
1. 검증된 인과관계를 우선적으로 사용하세요.
2. 알람 발생 시 점검 순서대로 원인을 분석하세요.
3. 불가능한 관계는 상관관계가 있어도 인과관계로 해석하지 마세요.
4. 파라미터 그룹 내 다른 파라미터도 함께 확인하세요.
5. 불확실한 경우 "추가 분석 필요"라고 명시하세요.
"""
        return prompt

    def export_to_json(self, output_path: str) -> bool:
        """JSON으로 내보내기"""
        try:
            k = self.knowledge
            data = {
                'exported_at': datetime.now().isoformat(),
                'source_file': k.source_file,
                'causal_relationships': [r.to_dict() for r in k.causal_relationships],
                'alarm_causes': [
                    {
                        'alarm_code': c.alarm_code,
                        'cause_parameter': c.cause_parameter,
                        'condition': c.condition,
                        'probability': c.probability,
                        'action': c.action,
                        'priority': c.priority,
                    }
                    for c in k.alarm_causes
                ],
                'leading_indicators': [
                    {
                        'target_event': i.target_event,
                        'indicator': i.indicator_parameter,
                        'pattern': i.pattern_type,
                        'lead_time': i.lead_time_seconds,
                    }
                    for i in k.leading_indicators
                ],
                'impossible_relationships': [
                    {'param1': i.param1, 'param2': i.param2, 'reason': i.reason}
                    for i in k.impossible_relationships
                ],
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            warnings.warn(f"JSON 내보내기 실패: {e}")
            return False


class RelationshipValidator:
    """발견된 관계를 전문가 지식으로 검증"""

    def __init__(self, knowledge: ExpertKnowledge):
        self.knowledge = knowledge

    def validate(self, source: str, target: str, relation_type: str = None) -> Dict[str, Any]:
        """
        발견된 관계 검증

        Returns:
            {
                'is_valid': bool,
                'is_verified': bool,      # 전문가가 정의한 관계인가
                'is_impossible': bool,    # 불가능한 관계인가
                'expert_relation': ...,   # 매칭되는 전문가 정의 관계
                'confidence_boost': float # 신뢰도 가중치
            }
        """
        k = self.knowledge

        # 불가능한 관계 확인
        if k.is_impossible_relationship(source, target):
            return {
                'is_valid': False,
                'is_verified': False,
                'is_impossible': True,
                'expert_relation': None,
                'confidence_boost': 0.0,
                'reason': '물리적으로 불가능한 관계'
            }

        # 전문가 정의 관계 확인
        for rel in k.causal_relationships:
            if rel.source == source and rel.target == target:
                return {
                    'is_valid': True,
                    'is_verified': True,
                    'is_impossible': False,
                    'expert_relation': rel,
                    'confidence_boost': rel.confidence,
                    'reason': rel.physics_explanation
                }

        # 정의되지 않은 관계 (데이터에서만 발견)
        return {
            'is_valid': True,
            'is_verified': False,
            'is_impossible': False,
            'expert_relation': None,
            'confidence_boost': 0.5,
            'reason': '전문가 검증 대기'
        }

    def filter_discoveries(
        self,
        discoveries: List[Dict],
        remove_impossible: bool = True,
        boost_verified: bool = True
    ) -> List[Dict]:
        """발견된 관계 목록 필터링 및 보강"""
        filtered = []

        for disc in discoveries:
            source = disc.get('source', disc.get('source_param'))
            target = disc.get('target', disc.get('target_param'))

            validation = self.validate(source, target)

            if remove_impossible and validation['is_impossible']:
                continue

            if boost_verified and validation['is_verified']:
                disc['confidence'] = min(1.0, disc.get('confidence', 0.5) + 0.2)
                disc['verified_by_expert'] = True
                disc['physics_explanation'] = validation['reason']

            disc['validation'] = validation
            filtered.append(disc)

        return filtered
