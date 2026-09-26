"""액션 감사 기록

모든 상태 변경은 "누가, 언제, 무엇을, 왜, 무엇이 바뀌었나"를 남긴다.
이 기록이 없으면 Action 계층을 두는 의미의 절반이 사라진다.

싱크는 두 가지를 제공한다.
  - InMemoryAuditSink : 테스트/개발용
  - PostgresAuditSink : 운영용 (action_audit 테이블)
"""
from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["AuditRecord", "AuditSink", "InMemoryAuditSink", "PostgresAuditSink"]


@dataclass
class AuditRecord:
    """감사 기록 1건"""
    audit_id: str
    action_type: str
    principal_id: str
    principal_roles: List[str]
    parameters: Dict[str, Any]
    reason: Optional[str]
    dry_run: bool
    succeeded: bool
    changes: List[Dict[str, Any]]
    error: Optional[str]
    occurred_at: datetime
    on_behalf_of: Optional[str] = None
    is_agent: bool = False
    client_request_id: Optional[str] = None

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["occurred_at"] = self.occurred_at.isoformat()
        return d


class AuditSink(ABC):
    """감사 기록 저장소"""

    @abstractmethod
    def write(self, record: AuditRecord) -> None:
        ...


@dataclass
class InMemoryAuditSink(AuditSink):
    """메모리 저장 (테스트/개발용)

    운영에서 쓰면 프로세스가 죽는 순간 기록이 사라진다.
    """
    records: List[AuditRecord] = field(default_factory=list)

    def write(self, record: AuditRecord) -> None:
        self.records.append(record)

    def find(self, action_type: str) -> List[AuditRecord]:
        return [r for r in self.records if r.action_type == action_type]


class PostgresAuditSink(AuditSink):
    """PostgreSQL ``action_audit`` 테이블에 기록.

    DDL은 infra/postgres/init.sql 참조.
    커넥션은 psycopg2 호환 객체를 주입받는다.
    """

    INSERT_SQL = """
        INSERT INTO action_audit (
            audit_id, action_type, principal_id, principal_roles,
            parameters, reason, dry_run, succeeded, changes, error,
            occurred_at, on_behalf_of, is_agent, client_request_id
        ) VALUES (
            %(audit_id)s, %(action_type)s, %(principal_id)s, %(principal_roles)s,
            %(parameters)s, %(reason)s, %(dry_run)s, %(succeeded)s, %(changes)s, %(error)s,
            %(occurred_at)s, %(on_behalf_of)s, %(is_agent)s, %(client_request_id)s
        )
    """

    def __init__(self, connection):
        self.connection = connection

    def write(self, record: AuditRecord) -> None:
        params = {
            "audit_id": record.audit_id,
            "action_type": record.action_type,
            "principal_id": record.principal_id,
            "principal_roles": list(record.principal_roles),
            "parameters": json.dumps(record.parameters, ensure_ascii=False, default=str),
            "reason": record.reason,
            "dry_run": record.dry_run,
            "succeeded": record.succeeded,
            "changes": json.dumps(record.changes, ensure_ascii=False, default=str),
            "error": record.error,
            "occurred_at": record.occurred_at,
            "on_behalf_of": record.on_behalf_of,
            "is_agent": record.is_agent,
            "client_request_id": record.client_request_id,
        }
        with self.connection.cursor() as cur:
            cur.execute(self.INSERT_SQL, params)
        self.connection.commit()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
