"""
온톨로지 서비스

PostgreSQL + Apache AGE 기반 그래프 데이터 관리
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import asyncpg
import redis.asyncio as redis

from config import settings

logger = logging.getLogger(__name__)


class OntologyService:
    """온톨로지 데이터 서비스"""

    def __init__(self):
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.graph_name = "manufacturing"

    async def initialize(self):
        """서비스 초기화"""
        try:
            # PostgreSQL 연결 풀
            self.pg_pool = await asyncpg.create_pool(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                min_size=5,
                max_size=20,
            )
            logger.info("PostgreSQL connection pool created")

            # AGE 확장 로드 확인
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")

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
            logger.error(f"Failed to initialize OntologyService: {e}")
            raise

    async def close(self):
        """서비스 종료"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()

    # =========================================================
    # 설비 (Equipment)
    # =========================================================

    async def list_equipment(
        self,
        equipment_type: Optional[str] = None,
        status: Optional[str] = None,
        location: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """설비 목록 조회"""
        # 캐시 확인
        cache_key = f"equipment:list:{equipment_type}:{status}:{location}:{limit}:{offset}"
        cached = await self.redis_client.get(cache_key)
        if cached:
            import json
            return json.loads(cached)

        # AGE Cypher 쿼리
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (e:Equipment)
            {"WHERE e.equipment_type = '" + equipment_type + "'" if equipment_type else ""}
            {"AND" if equipment_type and status else "WHERE" if status else ""} {"e.status = '" + status + "'" if status else ""}
            {"AND" if (equipment_type or status) and location else "WHERE" if location else ""} {"e.location = '" + location + "'" if location else ""}
            RETURN e
            ORDER BY e.equipment_id
            SKIP {offset}
            LIMIT {limit}
        $$) as (equipment agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            result = []
            for row in rows:
                equipment = self._parse_agtype(str(row['equipment']))
                result.append(self._parse_vertex(equipment))

            # 캐시 저장 (5분)
            import json
            await self.redis_client.setex(cache_key, 300, json.dumps(result))

            return result

        except Exception as e:
            logger.error(f"Failed to list equipment: {e}")
            return []

    async def get_equipment(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """설비 상세 조회"""
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})
            RETURN e
        $$) as (equipment agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                row = await conn.fetchrow(query)

            if row:
                equipment = self._parse_agtype(str(row['equipment']))
                return self._parse_vertex(equipment)
            return None

        except Exception as e:
            logger.error(f"Failed to get equipment {equipment_id}: {e}")
            return None

    async def create_equipment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """설비 생성"""
        props = ", ".join([f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}"
                          for k, v in data.items()])

        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            CREATE (e:Equipment {{{props}}})
            RETURN e
        $$) as (equipment agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                row = await conn.fetchrow(query)

            equipment = self._parse_agtype(str(row['equipment']))
            return self._parse_vertex(equipment)

        except Exception as e:
            logger.error(f"Failed to create equipment: {e}")
            raise

    async def get_equipment_alarms(
        self,
        equipment_id: str,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """설비 알람 이력 조회"""
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (e:Equipment {{equipment_id: '{equipment_id}'}})-[:GENERATES]->(a:Alarm)
            {"WHERE a.severity = '" + severity + "'" if severity else ""}
            RETURN a
            ORDER BY a.occurred_at DESC
            LIMIT {limit}
        $$) as (alarm agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            result = []
            for row in rows:
                alarm = self._parse_agtype(str(row['alarm']))
                result.append(self._parse_vertex(alarm))

            return result

        except Exception as e:
            logger.error(f"Failed to get equipment alarms: {e}")
            return []

    async def get_equipment_measurements(
        self,
        equipment_id: str,
        param_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """설비 측정 데이터 조회 (TimescaleDB)"""
        # TimescaleDB에서 조회
        from config import settings

        query = """
        SELECT * FROM fdc_measurements
        WHERE equipment_id = $1
        """
        params = [equipment_id]

        if param_id:
            query += f" AND param_id = ${len(params) + 1}"
            params.append(param_id)
        if since:
            query += f" AND timestamp >= ${len(params) + 1}"
            params.append(since)
        if until:
            query += f" AND timestamp <= ${len(params) + 1}"
            params.append(until)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        try:
            # TimescaleDB 연결
            ts_pool = await asyncpg.create_pool(
                host=settings.timescale_host,
                port=settings.timescale_port,
                user=settings.timescale_user,
                password=settings.timescale_password,
                database=settings.timescale_db,
            )

            async with ts_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            await ts_pool.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get measurements: {e}")
            return []

    # =========================================================
    # Lot
    # =========================================================

    async def list_lots(
        self,
        product_code: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Lot 목록 조회"""
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (l:Lot)
            {"WHERE l.product_code = '" + product_code + "'" if product_code else ""}
            {"AND" if product_code and status else "WHERE" if status else ""} {"l.status = '" + status + "'" if status else ""}
            RETURN l
            ORDER BY l.start_time DESC
            SKIP {offset}
            LIMIT {limit}
        $$) as (lot agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            result = []
            for row in rows:
                lot = self._parse_agtype(str(row['lot']))
                result.append(self._parse_vertex(lot))

            return result

        except Exception as e:
            logger.error(f"Failed to list lots: {e}")
            return []

    async def get_lot(self, lot_id: str) -> Optional[Dict[str, Any]]:
        """Lot 상세 조회"""
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (l:Lot {{lot_id: '{lot_id}'}})
            OPTIONAL MATCH (l)-[:CONTAINS]->(w:Wafer)
            RETURN l, count(w) as wafer_count
        $$) as (lot agtype, wafer_count agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                row = await conn.fetchrow(query)

            if row:
                lot = self._parse_agtype(str(row['lot']))
                result = self._parse_vertex(lot)
                result['wafer_count'] = int(str(row['wafer_count']))
                return result
            return None

        except Exception as e:
            logger.error(f"Failed to get lot {lot_id}: {e}")
            return None

    async def trace_lot_forward(self, lot_id: str, depth: int = 3) -> Dict[str, Any]:
        """Lot 순방향 추적"""
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH p = (l:Lot {{lot_id: '{lot_id}'}})-[*1..{depth}]->()
            RETURN p
        $$) as (path agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            nodes = {}
            edges = []

            for row in rows:
                path = self._parse_agtype(str(row['path']))
                self._parse_path(path, nodes, edges)

            return {
                "nodes": list(nodes.values()),
                "edges": edges,
            }

        except Exception as e:
            logger.error(f"Failed to trace lot {lot_id}: {e}")
            return {"nodes": [], "edges": []}

    async def get_lot_genealogy(self, lot_id: str) -> List[Dict[str, Any]]:
        """Lot 계보 (공정 이력)"""
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (l:Lot {{lot_id: '{lot_id}'}})-[r:PROCESSED_AT]->(e:Equipment)
            RETURN e, r
            ORDER BY r.start_time
        $$) as (equipment agtype, relation agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            result = []
            for row in rows:
                equipment = self._parse_agtype(str(row['equipment']))
                relation = json.loads(str(row['relation']))
                result.append({
                    "equipment": self._parse_vertex(equipment),
                    "relation": self._parse_edge(relation),
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get lot genealogy: {e}")
            return []

    # =========================================================
    # 그래프 탐색
    # =========================================================

    async def traverse_graph(
        self,
        start_type: str,
        start_id: str,
        direction: str = "both",
        depth: int = 2,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """그래프 탐색"""
        id_field = f"{start_type.lower()}_id"
        rel_filter = f"[:{('|'.join(relation_types))}]" if relation_types else ""

        if direction == "forward":
            if rel_filter:
                pattern = f"(n:{start_type} {{{id_field}: '{start_id}'}})-{rel_filter}*1..{depth}->(m)"
            else:
                pattern = f"(n:{start_type} {{{id_field}: '{start_id}'}})-[*1..{depth}]->(m)"
        elif direction == "backward":
            if rel_filter:
                pattern = f"(m)-{rel_filter}*1..{depth}->(n:{start_type} {{{id_field}: '{start_id}'}})"
            else:
                pattern = f"(m)-[*1..{depth}]->(n:{start_type} {{{id_field}: '{start_id}'}})"
        else:
            if rel_filter:
                pattern = f"(n:{start_type} {{{id_field}: '{start_id}'}})-{rel_filter}*1..{depth}-(m)"
            else:
                pattern = f"(n:{start_type} {{{id_field}: '{start_id}'}})-[*1..{depth}]-(m)"

        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH p = {pattern}
            RETURN p
        $$) as (path agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            nodes = {}
            edges = []

            for row in rows:
                path = self._parse_agtype(str(row['path']))
                self._parse_path(path, nodes, edges)

            return {
                "nodes": list(nodes.values()),
                "edges": edges,
            }

        except Exception as e:
            logger.error(f"Failed to traverse graph: {e}")
            return {"nodes": [], "edges": []}

    async def find_path(
        self,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """두 노드 간 경로 탐색"""
        from_id_field = f"{from_type.lower()}_id"
        to_id_field = f"{to_type.lower()}_id"

        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH p = shortestPath(
                (a:{from_type} {{{from_id_field}: '{from_id}'}})-[*..{max_depth}]-(b:{to_type} {{{to_id_field}: '{to_id}'}})
            )
            RETURN p
        $$) as (path agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                row = await conn.fetchrow(query)

            if row:
                path = self._parse_agtype(str(row['path']))
                nodes = {}
                edges = []
                self._parse_path(path, nodes, edges)
                return {
                    "found": True,
                    "nodes": list(nodes.values()),
                    "edges": edges,
                    "length": len(edges),
                }

            return {"found": False, "nodes": [], "edges": [], "length": 0}

        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return {"found": False, "nodes": [], "edges": [], "length": 0}

    async def get_neighbors(
        self,
        node_type: str,
        node_id: str,
        relation_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """이웃 노드 조회"""
        id_field = f"{node_type.lower()}_id"
        rel_filter = f":{relation_type}" if relation_type else ""

        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (n:{node_type} {{{id_field}: '{node_id}'}})-[r{rel_filter}]-(neighbor)
            RETURN neighbor, r, labels(neighbor) as labels
        $$) as (neighbor agtype, relation agtype, labels agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            result = []
            for row in rows:
                neighbor = self._parse_agtype(str(row['neighbor']))
                relation = self._parse_agtype(str(row['relation']))
                result.append({
                    "node": self._parse_vertex(neighbor),
                    "relation": self._parse_edge(relation),
                })

            return result

        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []

    # =========================================================
    # 관계
    # =========================================================

    async def create_relationship(
        self,
        from_type: str,
        from_id: str,
        relation: str,
        to_type: str,
        to_id: str,
        properties: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """관계 생성"""
        from_id_field = f"{from_type.lower()}_id"
        to_id_field = f"{to_type.lower()}_id"

        props = ""
        if properties:
            props = "{" + ", ".join([f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}"
                                      for k, v in properties.items()]) + "}"

        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (a:{from_type} {{{from_id_field}: '{from_id}'}}),
                  (b:{to_type} {{{to_id_field}: '{to_id}'}})
            CREATE (a)-[r:{relation} {props}]->(b)
            RETURN r
        $$) as (relation agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                row = await conn.fetchrow(query)

            rel = self._parse_agtype(str(row['relation']))
            return self._parse_edge(rel)

        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            raise

    async def list_relationships(
        self,
        from_type: Optional[str] = None,
        from_id: Optional[str] = None,
        relation: Optional[str] = None,
        to_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """관계 목록 조회"""
        # 간소화된 쿼리
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (a)-[r]->(b)
            RETURN a, r, b
            LIMIT {limit}
        $$) as (from_node agtype, relation agtype, to_node agtype);
        """

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")
                rows = await conn.fetch(query)

            result = []
            for row in rows:
                from_node = self._parse_agtype(str(row['from_node']))
                rel = self._parse_agtype(str(row['relation']))
                to_node = self._parse_agtype(str(row['to_node']))

                result.append({
                    "from": self._parse_vertex(from_node),
                    "relation": self._parse_edge(rel),
                    "to": self._parse_vertex(to_node),
                })

            return result

        except Exception as e:
            logger.error(f"Failed to list relationships: {e}")
            return []

    # =========================================================
    # 헬퍼 메서드
    # =========================================================

    def _parse_agtype(self, agtype_str: str) -> Any:
        """AGE agtype 문자열을 JSON으로 파싱"""
        import json
        import re as regex

        if agtype_str is None:
            return None

        s = str(agtype_str).strip()

        # 배열 내부의 ::vertex, ::edge 접미사 제거
        s = regex.sub(r'}::vertex', '}', s)
        s = regex.sub(r'}::edge', '}', s)

        # 외부 ::path, ::vertex, ::edge 등 접미사 제거
        s = regex.sub(r'::(vertex|edge|path|numeric|integer|float|string)$', '', s)

        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse agtype: {e}")
            return s

    def _parse_vertex(self, vertex: Any) -> Dict[str, Any]:
        """AGE vertex 파싱 - 프론트엔드 D3.js 형식에 맞게 변환"""
        if isinstance(vertex, str):
            vertex = self._parse_agtype(vertex)

        if isinstance(vertex, dict):
            props = vertex.get('properties', {})
            node_type = vertex.get('label')  # Equipment, Lot, Wafer, Alarm 등

            # 노드 타입에 따른 표시 이름 결정
            display_name = (
                props.get('name') or
                props.get('equipment_id') or
                props.get('lot_id') or
                props.get('wafer_id') or
                props.get('alarm_name') or
                props.get('alarm_id') or
                props.get('recipe_id') or
                str(vertex.get('id'))
            )

            result = {
                "id": str(vertex.get('id')),  # 문자열로 변환 (D3.js 호환)
                "type": node_type,  # 프론트엔드가 기대하는 필드
                "label": display_name,  # 표시용 이름
                "properties": props,  # 원본 속성
            }

            return result
        return {}

    def _parse_edge(self, edge: Any) -> Dict[str, Any]:
        """AGE edge 파싱 - 프론트엔드 D3.js 형식에 맞게 변환"""
        if isinstance(edge, str):
            edge = self._parse_agtype(edge)

        if isinstance(edge, dict):
            props = edge.get('properties', {})
            result = {
                "id": str(edge.get('id')),  # 문자열로 변환
                "label": edge.get('label'),
                "source": str(edge.get('start_id')),  # D3.js는 source 사용
                "target": str(edge.get('end_id')),    # D3.js는 target 사용
            }
            result.update(props)
            return result
        return {}

    def _parse_path(self, path: Any, nodes: Dict, edges: List):
        """AGE path 파싱"""
        if isinstance(path, str):
            path = self._parse_agtype(path)

        if isinstance(path, list):
            for item in path:
                parsed = item
                if isinstance(item, str):
                    parsed = self._parse_agtype(item)

                if isinstance(parsed, dict):
                    if 'start_id' in parsed:
                        edge = self._parse_edge(parsed)
                        edges.append(edge)
                    else:
                        vertex = self._parse_vertex(parsed)
                        if vertex.get('id'):
                            nodes[vertex.get('id')] = vertex


# 전역 인스턴스
ontology_service = OntologyService()
