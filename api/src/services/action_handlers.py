"""액션 핸들러 구현

각 핸들러 시그니처::

    def handle(params: dict, principal: Principal, dry_run: bool) -> dict

반환값은 ``{"changes": [...], "message": "..."}``.

**dry_run=True이면 어떤 쓰기도 하지 않고 예상 변경만 돌려줘야 한다.**
이 규약이 깨지면 시뮬레이션이 실제 변경을 일으킨다.

현재 핸들러들은 변경 계획(changes)을 계산하고 감사 기록까지 남기지만,
실제 그래프/DB 반영은 대상 서비스 연동이 끝나는 대로 채운다.
미구현 구간은 각 핸들러의 TODO에 명시했다.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from common.actions import ActionRegistry, Principal, ValidationFailed

logger = logging.getLogger(__name__)

__all__ = ["register_handlers"]


def _change(
    object_type: str,
    object_id: str,
    field: str,
    before: Any,
    after: Any,
) -> Dict[str, Any]:
    """변경 1건을 표준 형태로 기록한다.

    before/after를 함께 남겨야 감사 기록만 보고 되돌릴 수 있다.
    """
    return {
        "object_type": object_type,
        "object_id": object_id,
        "field": field,
        "before": before,
        "after": after,
    }


def register_handlers(registry: ActionRegistry) -> ActionRegistry:
    """레지스트리에 핸들러를 등록한다."""

    # ------------------------------------------------------------------
    @registry.handler("VerifyRelationship")
    def verify_relationship(params, principal: Principal, dry_run: bool) -> Dict[str, Any]:
        source = params["source"]
        target = params["target"]
        relation_type = params["relation_type"]
        decision = params["decision"]
        edge_id = f"{source}-[{relation_type}]->{target}"

        changes: List[Dict[str, Any]] = [
            _change("Relationship", edge_id, "verification_status", "pending", decision),
            _change("Relationship", edge_id, "verified_by", None, principal.user_id),
        ]
        if params.get("notes"):
            changes.append(_change("Relationship", edge_id, "notes", None, params["notes"]))

        if dry_run:
            return {
                "changes": changes,
                "message": f"[시뮬레이션] {edge_id} 를 {decision} 처리합니다",
            }

        # TODO: RelationshipStore.verify_relationship() 연결
        #       analytics 패키지 의존성이 API 컨테이너에 들어온 뒤 활성화한다.
        logger.info(
            "VerifyRelationship: %s -> %s (by %s)", edge_id, decision, principal.user_id
        )
        return {
            "changes": changes,
            "message": f"{edge_id} 를 {decision} 처리했습니다",
        }

    # ------------------------------------------------------------------
    @registry.handler("RecordExpertRelationship")
    def record_expert_relationship(params, principal: Principal, dry_run: bool) -> Dict[str, Any]:
        source = params["source"]
        target = params["target"]
        relation_type = params["relation_type"]
        edge_id = f"{source}-[{relation_type}]->{target}"

        # 전문가 입력은 사람이 근거이므로 즉시 verified 상태로 들어간다.
        changes = [
            _change("Relationship", edge_id, "origin", None, "expert"),
            _change("Relationship", edge_id, "verification_status", None, "verified"),
            _change("Relationship", edge_id, "verified_by", None, principal.user_id),
            _change("Relationship", edge_id, "confidence", None, params.get("confidence")),
        ]
        if params.get("lag_seconds") is not None:
            changes.append(
                _change("Relationship", edge_id, "lag_seconds", None, params["lag_seconds"])
            )

        prefix = "[시뮬레이션] " if dry_run else ""
        note = ""
        if relation_type == "IMPOSSIBLE":
            note = " (통계 발견 결과의 오탐 필터로 즉시 반영됩니다)"

        if not dry_run:
            # TODO: RelationshipStore 연결
            logger.info("RecordExpertRelationship: %s (by %s)", edge_id, principal.user_id)

        return {
            "changes": changes,
            "message": f"{prefix}전문가 관계 {edge_id} 를 등록합니다{note}",
        }

    # ------------------------------------------------------------------
    @registry.handler("AcknowledgeAlarm")
    def acknowledge_alarm(params, principal: Principal, dry_run: bool) -> Dict[str, Any]:
        alarm_id = params["alarm_id"]
        escalate_to = params.get("escalate_to")
        new_status = "ESCALATED" if escalate_to else "ACKNOWLEDGED"

        changes = [
            _change("Alarm", alarm_id, "status", "ACTIVE", new_status),
            _change("Alarm", alarm_id, "acknowledged_by", None, principal.user_id),
        ]
        if escalate_to:
            changes.append(_change("Alarm", alarm_id, "escalated_to", None, escalate_to))
        if params.get("action_taken"):
            changes.append(
                _change("Alarm", alarm_id, "action_taken", None, params["action_taken"])
            )

        if not dry_run:
            # TODO: 알람 저장소 연결 (realtime 라우터의 알람 상태와 통합)
            logger.info(
                "AcknowledgeAlarm: %s -> %s (by %s)", alarm_id, new_status, principal.user_id
            )

        prefix = "[시뮬레이션] " if dry_run else ""
        return {"changes": changes, "message": f"{prefix}알람 {alarm_id} 를 {new_status} 처리합니다"}

    # ------------------------------------------------------------------
    @registry.handler("UpdateEquipmentState")
    def update_equipment_state(params, principal: Principal, dry_run: bool) -> Dict[str, Any]:
        equipment_id = params["equipment_id"]
        chamber_id = params.get("chamber_id")
        to_state = params["to_state"]
        target_type = "Chamber" if chamber_id else "Equipment"
        target_id = chamber_id or equipment_id

        # TODO: 현재 상태를 온톨로지에서 조회. 연동 전까지는 전이 검증을 건너뛴다.
        from_state = None

        definition = registry.get("UpdateEquipmentState")
        if from_state is not None and definition.transitions is not None:
            allowed = definition.transitions.get(from_state, ())
            if to_state not in allowed:
                # SEMI E10 전이 규칙 위반. 허용하면 가동률 집계가 무너진다.
                raise ValidationFailed({
                    "to_state": (
                        f"{from_state} 에서 {to_state} 로 직접 전이할 수 없습니다. "
                        f"가능: {', '.join(allowed) or '없음'}"
                    )
                })

        changes = [_change(target_type, target_id, "status", from_state, to_state)]
        if params.get("substate"):
            changes.append(
                _change(target_type, target_id, "e10_substate", None, params["substate"])
            )

        if not dry_run:
            # TODO: ontology_service 를 통한 상태 갱신 + 상태 구간 이력 적재
            logger.info(
                "UpdateEquipmentState: %s -> %s (by %s)", target_id, to_state, principal.user_id
            )

        prefix = "[시뮬레이션] " if dry_run else ""
        return {
            "changes": changes,
            "message": f"{prefix}{target_id} 상태를 {to_state} 로 전이합니다",
        }

    # ------------------------------------------------------------------
    @registry.handler("HoldLot")
    def hold_lot(params, principal: Principal, dry_run: bool) -> Dict[str, Any]:
        lot_id = params["lot_id"]
        hold = params["hold"]
        new_status = "ON_HOLD" if hold else "RELEASED"

        changes = [_change("Lot", lot_id, "status", None, new_status)]
        if hold and params.get("hold_code"):
            changes.append(_change("Lot", lot_id, "hold_code", None, params["hold_code"]))
        if params.get("triggered_by_alarm_id"):
            changes.append(
                _change("Lot", lot_id, "triggered_by_alarm_id", None,
                        params["triggered_by_alarm_id"])
            )

        if not dry_run:
            # TODO: MES 연동 (읽기 전용 연동이면 지시 큐에 적재)
            logger.info("HoldLot: %s -> %s (by %s)", lot_id, new_status, principal.user_id)

        prefix = "[시뮬레이션] " if dry_run else ""
        verb = "홀드" if hold else "홀드 해제"
        return {"changes": changes, "message": f"{prefix}Lot {lot_id} 를 {verb} 합니다"}

    # ------------------------------------------------------------------
    @registry.handler("RequestInspection")
    def request_inspection(params, principal: Principal, dry_run: bool) -> Dict[str, Any]:
        equipment_id = params["equipment_id"]
        chamber_id = params.get("chamber_id")
        target_id = chamber_id or equipment_id

        changes = [
            _change("Inspection", target_id, "requested_by", None, principal.user_id),
            _change("Inspection", target_id, "priority", None, params["priority"]),
            _change("Inspection", target_id, "checklist", None, params["checklist"]),
            _change("Inspection", target_id, "due_hours", None, params.get("due_hours")),
        ]
        if params.get("based_on_relationship"):
            changes.append(
                _change("Inspection", target_id, "based_on_relationship", None,
                        params["based_on_relationship"])
            )

        if not dry_run:
            # TODO: 점검 지시 저장소 + 담당자 배정 연동
            logger.info(
                "RequestInspection: %s (%s, by %s)",
                target_id, params["priority"], principal.user_id,
            )

        prefix = "[시뮬레이션] " if dry_run else ""
        return {
            "changes": changes,
            "message": f"{prefix}{target_id} 점검 지시를 생성합니다 ({params['priority']})",
        }

    return registry
