"""
Relationship Store
==================

발견된 관계를 온톨로지(PostgreSQL + Apache AGE)에 저장

기능:
- Parameter 노드 생성/조회
- 관계 Edge 생성
- 관계 이력 관리
- 검증 상태 관리
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings

from common.cypher_safe import cy_label, cy_num, cy_props, cy_str
from .config import OntologyConfig, RelationType
from .correlation_analyzer import CorrelationResult
from .causality_analyzer import CausalityResult
from .pattern_detector import PatternResult


class VerificationStatus(Enum):
    """검증 상태"""
    PENDING = "pending"           # 검증 대기
    VERIFIED = "verified"         # 전문가 검증 완료
    REJECTED = "rejected"         # 거부됨
    AUTO_APPROVED = "auto"        # 자동 승인 (높은 신뢰도)


class RelationshipOrigin(Enum):
    """관계의 출처

    출처가 다르면 신뢰도와 검증 절차가 달라야 한다.
    이 값을 남기지 않으면 몇 달 뒤에 "이 관계는 사람이 넣은 건가
    기계가 찾은 건가"를 아무도 알 수 없게 된다.
    """
    STRUCTURAL = "structural"   # 계층1: 키/시간구간 조인으로 도출된 사실
    EXPERT = "expert"           # 계층2: 엔지니어가 입력한 도메인 지식
    DISCOVERED = "discovered"   # 계층3: 통계 분석이 제시한 가설


@dataclass
class DiscoveredRelationship:
    """발견된 관계 통합 모델"""
    source: str
    target: str
    relation_type: str
    method: str
    confidence: float
    properties: Dict[str, Any]
    discovered_at: datetime
    verification_status: str = VerificationStatus.PENDING.value
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    notes: Optional[str] = None

    # --- 출처/근거/유효기간 (docs/06_표준_적용_가이드.md 참조) ---
    origin: str = RelationshipOrigin.DISCOVERED.value
    evidence: Dict[str, Any] = field(default_factory=dict)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    def to_cypher_props(self) -> str:
        """Cypher 속성 문자열 생성"""
        props = {
            'method': self.method,
            'confidence': round(self.confidence, 4),
            'discovered_at': self.discovered_at.isoformat(),
            'verification_status': self.verification_status,
            'origin': self.origin,
            'verified_by': self.verified_by,
            'verified_at': self.verified_at,
            'valid_from': self.valid_from,
            'valid_to': self.valid_to,
            # 근거는 개별 속성으로 펼쳐 저장한다 (evidence_p_value 등).
            # 그래야 그래프 질의에서 바로 필터링할 수 있다.
            **{f'evidence_{k}': v for k, v in self.evidence.items() if v is not None},
            **{k: v for k, v in self.properties.items() if v is not None}
        }

        return cy_props(props)


class RelationshipStore:
    """관계 저장소"""

    def __init__(self, config: Optional[OntologyConfig] = None):
        self.config = config or OntologyConfig()
        self.conn = None
        self.pending_relationships: List[DiscoveredRelationship] = []

    def connect(self) -> bool:
        """데이터베이스 연결"""
        try:
            import psycopg2
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            # AGE 설정
            with self.conn.cursor() as cur:
                cur.execute("LOAD 'age';")
                cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            self.conn.commit()
            return True
        except ImportError:
            warnings.warn("psycopg2가 필요합니다: pip install psycopg2-binary")
            return False
        except Exception as e:
            warnings.warn(f"데이터베이스 연결 실패: {e}")
            return False

    def disconnect(self) -> None:
        """연결 종료"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def ensure_parameter_node(self, param_name: str, param_type: str = "PV") -> bool:
        """Parameter 노드가 없으면 생성"""
        if not self.conn:
            return False

        try:
            with self.conn.cursor() as cur:
                # 노드 존재 확인
                cur.execute(f"""
                    SELECT * FROM cypher('{self.config.graph_name}', $$
                        MATCH (p:Parameter {{name: {cy_str(param_name)}}})
                        RETURN p
                    $$) AS (p agtype);
                """)

                if cur.fetchone() is None:
                    # 노드 생성
                    cur.execute(f"""
                        SELECT * FROM cypher('{self.config.graph_name}', $$
                            CREATE (p:Parameter {{
                                name: {cy_str(param_name)},
                                type: {cy_str(param_type)},
                                created_at: {cy_str(datetime.now().isoformat())}
                            }})
                            RETURN p
                        $$) AS (p agtype);
                    """)

            self.conn.commit()
            return True

        except Exception as e:
            warnings.warn(f"Parameter 노드 생성 실패 ({param_name}): {e}")
            self.conn.rollback()
            return False

    def add_correlation(self, result: CorrelationResult) -> DiscoveredRelationship:
        """상관관계 결과 추가"""
        rel = DiscoveredRelationship(
            source=result.source_param,
            target=result.target_param,
            relation_type=result.relation_type,
            method=result.method,
            confidence=result.confidence,
            properties={
                'correlation': result.correlation,
                'p_value': result.p_value,
                'n_samples': result.n_samples,
                'lag': result.lag,
                'lag_seconds': result.lag_seconds,
            },
            discovered_at=datetime.now(),
            verification_status=self._auto_verify(result.confidence)
        )
        self.pending_relationships.append(rel)
        return rel

    def add_causality(self, result: CausalityResult) -> DiscoveredRelationship:
        """인과관계 결과 추가"""
        rel = DiscoveredRelationship(
            source=result.source_param,
            target=result.target_param,
            relation_type=result.relation_type,
            method=result.method,
            confidence=result.confidence,
            properties={
                'statistic': result.statistic,
                'p_value': result.p_value,
                'optimal_lag': result.optimal_lag,
                'lag_seconds': result.lag_seconds,
                'direction_strength': result.direction_strength,
            },
            discovered_at=datetime.now(),
            verification_status=self._auto_verify(result.confidence)
        )
        self.pending_relationships.append(rel)
        return rel

    def add_pattern(self, result: PatternResult) -> DiscoveredRelationship:
        """패턴 결과 추가"""
        # 패턴은 여러 노드를 연결하므로 순차적 관계로 분해
        relationships = []

        for i in range(len(result.pattern) - 1):
            rel = DiscoveredRelationship(
                source=result.pattern[i],
                target=result.pattern[i + 1],
                relation_type=result.relation_type,
                method="pattern_mining",
                confidence=result.confidence,
                properties={
                    'support': result.support,
                    'lift': result.lift,
                    'count': result.count,
                    'avg_time_gap': result.avg_time_gap,
                    'pattern_type': result.pattern_type,
                    'full_pattern': ' → '.join(result.pattern),
                },
                discovered_at=datetime.now(),
                verification_status=self._auto_verify(result.confidence)
            )
            relationships.append(rel)
            self.pending_relationships.append(rel)

        return relationships[-1] if relationships else None

    def _auto_verify(self, confidence: float) -> str:
        """자동 검증 여부 결정

        [중요] 통계 신뢰도만으로 관계를 승격시키지 않는다.

        신뢰도가 높다는 것은 "데이터에서 강하게 보인다"이지
        "물리적으로 맞다"가 아니다. 같은 챔버의 두 파라미터는 레시피 스텝이
        동시에 바뀌기 때문에 인과가 아니어도 상관계수가 쉽게 0.9를 넘는다.
        이런 관계를 자동 승인하면 그래프가 의미 없는 엣지로 가득 찬다.

        승격 관문은 다음 순서여야 한다.
          1. 전문가 지식의 "불가능한 관계" 필터 통과 (RelationshipValidator)
          2. 물리적 타당성 (lag 방향이 공정 순서와 모순되지 않는가)
          3. 재현성 (다른 기간/다른 설비에서도 나타나는가)
          4. 엔지니어 검토

        따라서 여기서는 신뢰도가 낮은 것을 걸러낼 뿐, 승격은 하지 않는다.
        ``require_verification=False``로 명시적으로 끈 경우에만
        자동 승인하며, 이는 실데이터 운영에서는 권장하지 않는다.
        """
        if not self.config.require_verification:
            warnings.warn(
                "require_verification=False: 발견된 관계가 전문가 검증 없이 "
                "온톨로지에 승인됩니다. 운영 환경에서는 권장하지 않습니다.",
                stacklevel=2,
            )
            return VerificationStatus.AUTO_APPROVED.value

        # 신뢰도가 충분해도 pending에 머문다. 승격은 사람이 한다.
        return VerificationStatus.PENDING.value

    def save_pending(self, save_all: bool = False) -> Dict[str, int]:
        """대기 중인 관계 저장"""
        if not self.conn:
            if not self.connect():
                return {"error": "connection_failed", "saved": 0}

        stats = {"saved": 0, "skipped": 0, "errors": 0}

        for rel in self.pending_relationships:
            # 검증 대기 상태면 스킵 (save_all이 아닌 경우)
            if not save_all and rel.verification_status == VerificationStatus.PENDING.value:
                stats["skipped"] += 1
                continue

            try:
                self._save_relationship(rel)
                stats["saved"] += 1
            except Exception as e:
                warnings.warn(f"관계 저장 실패: {e}")
                stats["errors"] += 1

        # 저장 완료된 관계 제거
        if not save_all:
            self.pending_relationships = [
                r for r in self.pending_relationships
                if r.verification_status == VerificationStatus.PENDING.value
            ]
        else:
            self.pending_relationships = []

        return stats

    def _save_relationship(self, rel: DiscoveredRelationship) -> bool:
        """단일 관계 저장"""
        if not self.conn:
            return False

        # Parameter 노드 확인/생성
        self.ensure_parameter_node(rel.source)
        self.ensure_parameter_node(rel.target)

        props = rel.to_cypher_props()

        try:
            with self.conn.cursor() as cur:
                # 기존 관계 확인
                cur.execute(f"""
                    SELECT * FROM cypher('{self.config.graph_name}', $$
                        MATCH (s:Parameter {{name: {cy_str(rel.source)}}})-[r:{cy_label(rel.relation_type)}]->(t:Parameter {{name: {cy_str(rel.target)}}})
                        RETURN r
                    $$) AS (r agtype);
                """)

                existing = cur.fetchone()

                if existing:
                    # 기존 관계 업데이트
                    cur.execute(f"""
                        SELECT * FROM cypher('{self.config.graph_name}', $$
                            MATCH (s:Parameter {{name: {cy_str(rel.source)}}})-[r:{cy_label(rel.relation_type)}]->(t:Parameter {{name: {cy_str(rel.target)}}})
                            SET r += {props}
                            RETURN r
                        $$) AS (r agtype);
                    """)
                else:
                    # 새 관계 생성
                    cur.execute(f"""
                        SELECT * FROM cypher('{self.config.graph_name}', $$
                            MATCH (s:Parameter {{name: {cy_str(rel.source)}}})
                            MATCH (t:Parameter {{name: {cy_str(rel.target)}}})
                            CREATE (s)-[r:{cy_label(rel.relation_type)} {props}]->(t)
                            RETURN r
                        $$) AS (r agtype);
                    """)

            self.conn.commit()
            return True

        except Exception as e:
            warnings.warn(f"관계 저장 실패 ({rel.source}→{rel.target}): {e}")
            self.conn.rollback()
            return False

    def verify_relationship(
        self,
        source: str,
        target: str,
        relation_type: str,
        status: VerificationStatus,
        verified_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """관계 검증 상태 업데이트"""
        # 대기 목록에서 찾기
        for rel in self.pending_relationships:
            if (rel.source == source and
                rel.target == target and
                rel.relation_type == relation_type):
                rel.verification_status = status.value
                rel.verified_by = verified_by
                rel.verified_at = datetime.now()
                rel.notes = notes
                return True

        # DB에서 업데이트
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT * FROM cypher('{self.config.graph_name}', $$
                            MATCH (s:Parameter {{name: {cy_str(source)}}})-[r:{cy_label(relation_type)}]->(t:Parameter {{name: {cy_str(target)}}})
                            SET r.verification_status = {cy_str(status.value)},
                                r.verified_by = {cy_str(verified_by)},
                                r.verified_at = {cy_str(datetime.now().isoformat())}
                            RETURN r
                        $$) AS (r agtype);
                    """)
                self.conn.commit()
                return True
            except Exception as e:
                warnings.warn(f"검증 상태 업데이트 실패: {e}")
                self.conn.rollback()

        return False

    def get_relationships(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        relation_type: Optional[str] = None,
        min_confidence: float = 0.0,
        verified_only: bool = False
    ) -> List[Dict[str, Any]]:
        """관계 조회"""
        if not self.conn:
            if not self.connect():
                return []

        conditions = []
        if source:
            conditions.append(f"s.name = {cy_str(source)}")
        if target:
            conditions.append(f"t.name = {cy_str(target)}")
        if min_confidence > 0:
            conditions.append(f"r.confidence >= {cy_num(min_confidence)}")
        if verified_only:
            conditions.append(f"r.verification_status IN ['verified', 'auto']")

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        rel_pattern = f"[r:{cy_label(relation_type)}]" if relation_type else "[r]"

        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT * FROM cypher('{self.config.graph_name}', $$
                        MATCH (s:Parameter)-{rel_pattern}->(t:Parameter)
                        {where_clause}
                        RETURN s.name, type(r), t.name, properties(r)
                    $$) AS (source agtype, rel_type agtype, target agtype, props agtype);
                """)

                results = []
                for row in cur.fetchall():
                    results.append({
                        'source': str(row[0]).strip('"'),
                        'relation_type': str(row[1]).strip('"'),
                        'target': str(row[2]).strip('"'),
                        'properties': json.loads(str(row[3])) if row[3] else {}
                    })

                return results

        except Exception as e:
            warnings.warn(f"관계 조회 실패: {e}")
            return []

    def get_pending_for_verification(self) -> List[DiscoveredRelationship]:
        """검증 대기 중인 관계 목록"""
        return [
            r for r in self.pending_relationships
            if r.verification_status == VerificationStatus.PENDING.value
        ]

    def export_to_json(self, filepath: str) -> bool:
        """관계를 JSON으로 내보내기"""
        try:
            data = {
                'exported_at': datetime.now().isoformat(),
                'pending': [asdict(r) for r in self.pending_relationships],
                'from_db': self.get_relationships()
            }

            # datetime 직렬화 처리
            def serialize(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=serialize, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            warnings.warn(f"JSON 내보내기 실패: {e}")
            return False

    def import_from_json(self, filepath: str) -> int:
        """JSON에서 관계 가져오기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            count = 0
            for rel_data in data.get('pending', []):
                rel = DiscoveredRelationship(
                    source=rel_data['source'],
                    target=rel_data['target'],
                    relation_type=rel_data['relation_type'],
                    method=rel_data['method'],
                    confidence=rel_data['confidence'],
                    properties=rel_data['properties'],
                    discovered_at=datetime.fromisoformat(rel_data['discovered_at']),
                    verification_status=rel_data.get('verification_status', 'pending')
                )
                self.pending_relationships.append(rel)
                count += 1

            return count

        except Exception as e:
            warnings.warn(f"JSON 가져오기 실패: {e}")
            return 0

    def summary(self) -> Dict[str, Any]:
        """저장소 상태 요약"""
        pending_by_status = {}
        for rel in self.pending_relationships:
            status = rel.verification_status
            pending_by_status[status] = pending_by_status.get(status, 0) + 1

        return {
            "pending_total": len(self.pending_relationships),
            "pending_by_status": pending_by_status,
            "connected": self.conn is not None,
        }
