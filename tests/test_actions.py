"""Action 계층 테스트

검증하는 것:
  - 파라미터 검증 (타입, 필수, enum, 범위, 패턴)
  - 권한 (역할 기반)
  - 사유(reason) 요구
  - dry_run 시 쓰기 없음
  - 감사 기록 (성공/실패/거부 모두)
  - 실제 액션 스키마 YAML이 전부 로드되는지
"""
import pytest

from common.actions import (
    ActionExecutor,
    ActionRegistry,
    ActionRequest,
    ActionTypeDef,
    InMemoryAuditSink,
    ParameterDef,
    PermissionDenied,
    Principal,
    UnknownActionType,
    ValidationFailed,
    load_action_types,
)


# --- 픽스처 ---

def _definition(**overrides) -> ActionTypeDef:
    base = dict(
        action_type="TestAction",
        parameters={
            "name": ParameterDef("name", "string", required=True),
            "count": ParameterDef("count", "integer", minimum=1, maximum=10),
            "mode": ParameterDef("mode", "string", values=("a", "b")),
            "flag": ParameterDef("flag", "boolean"),
        },
        allowed_roles=("engineer",),
        reason_required=False,
    )
    base.update(overrides)
    return ActionTypeDef(**base)


@pytest.fixture
def sink():
    return InMemoryAuditSink()


@pytest.fixture
def executor(sink):
    registry = ActionRegistry({"TestAction": _definition()})

    @registry.handler("TestAction")
    def handle(params, principal, dry_run):
        if dry_run:
            return {"changes": [{"field": "name", "after": params["name"]}],
                    "message": "simulated"}
        return {"changes": [{"field": "name", "after": params["name"]}], "message": "done"}

    return ActionExecutor(registry, sink)


@pytest.fixture
def engineer():
    return Principal(user_id="hong", roles=("engineer",), display_name="홍길동")


@pytest.fixture
def operator():
    return Principal(user_id="kim", roles=("operator",))


# --- Principal ---

class TestPrincipal:
    def test_requires_user_id(self):
        with pytest.raises(ValueError):
            Principal(user_id="")

    def test_roles_default_empty(self):
        assert Principal(user_id="a").roles == ()


# --- 권한 ---

class TestAuthorization:
    def test_allowed_role_passes(self, executor, engineer):
        result = executor.execute(
            ActionRequest("TestAction", {"name": "x"}), engineer
        )
        assert result.succeeded

    def test_wrong_role_denied(self, executor, operator):
        with pytest.raises(PermissionDenied):
            executor.execute(ActionRequest("TestAction", {"name": "x"}), operator)

    def test_empty_allowed_roles_permits_anyone(self, sink):
        registry = ActionRegistry({"Open": _definition(action_type="Open", allowed_roles=())})
        registry.register_handler("Open", lambda p, pr, d: {"message": "ok"})
        ex = ActionExecutor(registry, sink)
        assert ex.execute(ActionRequest("Open", {"name": "x"}), Principal("anyone")).succeeded

    def test_denied_attempt_is_audited(self, executor, operator, sink):
        # 거부된 시도도 남아야 한다
        with pytest.raises(PermissionDenied):
            executor.execute(ActionRequest("TestAction", {"name": "x"}), operator)
        assert len(sink.records) == 1
        assert sink.records[0].succeeded is False
        assert sink.records[0].principal_id == "kim"


# --- 파라미터 검증 ---

class TestValidation:
    def test_missing_required(self, executor, engineer):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(ActionRequest("TestAction", {}), engineer)
        assert "name" in exc.value.errors

    def test_wrong_type(self, executor, engineer):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(ActionRequest("TestAction", {"name": 123}), engineer)
        assert "name" in exc.value.errors

    def test_bool_is_not_accepted_as_integer(self, executor, engineer):
        # bool은 int의 하위 타입이라 명시적으로 막지 않으면 통과해버린다
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest("TestAction", {"name": "x", "count": True}), engineer
            )
        assert "count" in exc.value.errors

    def test_out_of_range(self, executor, engineer):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest("TestAction", {"name": "x", "count": 99}), engineer
            )
        assert "count" in exc.value.errors

    def test_invalid_enum(self, executor, engineer):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest("TestAction", {"name": "x", "mode": "z"}), engineer
            )
        assert "mode" in exc.value.errors

    def test_unknown_parameter_rejected(self, executor, engineer):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest("TestAction", {"name": "x", "bogus": 1}), engineer
            )
        assert "bogus" in exc.value.errors

    def test_all_errors_collected_at_once(self, executor, engineer):
        # 하나씩 알려주면 호출자가 여러 번 왕복해야 한다
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest("TestAction", {"count": 99, "mode": "z"}), engineer
            )
        assert set(exc.value.errors) == {"name", "count", "mode"}

    def test_pattern(self):
        p = ParameterDef("rel", "string", pattern="^[A-Z_]+$")
        assert p.validate("CAUSES") == "CAUSES"
        with pytest.raises(ValueError):
            p.validate("bad-value")

    def test_default_applied_when_absent(self):
        p = ParameterDef("n", "integer", default=7)
        assert p.validate(None) == 7

    def test_validation_failure_is_audited(self, executor, engineer, sink):
        with pytest.raises(ValidationFailed):
            executor.execute(ActionRequest("TestAction", {}), engineer)
        assert len(sink.records) == 1
        assert sink.records[0].succeeded is False


# --- 사유 요구 ---

class TestReasonRequired:
    @pytest.fixture
    def executor(self, sink):
        registry = ActionRegistry({"Risky": _definition(action_type="Risky",
                                                        reason_required=True)})
        registry.register_handler("Risky", lambda p, pr, d: {"message": "ok"})
        return ActionExecutor(registry, sink)

    def test_missing_reason_rejected(self, executor, engineer):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(ActionRequest("Risky", {"name": "x"}), engineer)
        assert "reason" in exc.value.errors

    def test_blank_reason_rejected(self, executor, engineer):
        with pytest.raises(ValidationFailed):
            executor.execute(ActionRequest("Risky", {"name": "x"}, reason="   "), engineer)

    def test_reason_present_passes(self, executor, engineer):
        result = executor.execute(
            ActionRequest("Risky", {"name": "x"}, reason="고온 알람 확인"), engineer
        )
        assert result.succeeded


# --- dry run / 시뮬레이션 ---

class TestDryRun:
    def test_handler_receives_dry_run_flag(self, sink, engineer):
        seen = {}
        registry = ActionRegistry({"TestAction": _definition()})
        registry.register_handler(
            "TestAction",
            lambda p, pr, d: seen.update(dry_run=d) or {"message": "ok"},
        )
        ex = ActionExecutor(registry, sink)
        ex.execute(ActionRequest("TestAction", {"name": "x"}, dry_run=True), engineer)
        assert seen["dry_run"] is True

    def test_simulate_forces_dry_run(self, executor, engineer):
        result = executor.simulate(
            ActionRequest("TestAction", {"name": "x"}, dry_run=False), engineer
        )
        assert result.dry_run is True
        assert result.message == "simulated"

    def test_simulate_still_checks_permission(self, executor, operator):
        with pytest.raises(PermissionDenied):
            executor.simulate(ActionRequest("TestAction", {"name": "x"}), operator)

    def test_simulate_still_validates(self, executor, engineer):
        with pytest.raises(ValidationFailed):
            executor.simulate(ActionRequest("TestAction", {}), engineer)

    def test_dry_run_recorded_in_audit(self, executor, engineer, sink):
        executor.simulate(ActionRequest("TestAction", {"name": "x"}), engineer)
        assert sink.records[0].dry_run is True


# --- 레지스트리 ---

class TestRegistry:
    def test_unknown_action_type(self, executor, engineer):
        with pytest.raises(UnknownActionType):
            executor.execute(ActionRequest("NoSuchAction", {}), engineer)

    def test_declared_but_unimplemented_is_rejected(self, sink, engineer):
        registry = ActionRegistry({"TestAction": _definition()})  # 핸들러 없음
        ex = ActionExecutor(registry, sink)
        with pytest.raises(UnknownActionType):
            ex.execute(ActionRequest("TestAction", {"name": "x"}), engineer)

    def test_cannot_register_handler_without_definition(self):
        registry = ActionRegistry({})
        with pytest.raises(UnknownActionType):
            registry.register_handler("Ghost", lambda *a: None)


# --- 감사 기록 ---

class TestAudit:
    def test_success_recorded_with_context(self, executor, engineer, sink):
        result = executor.execute(
            ActionRequest("TestAction", {"name": "x"}, reason="테스트",
                          client_request_id="req-1"),
            engineer,
        )
        rec = sink.records[0]
        assert rec.audit_id == result.audit_id
        assert rec.action_type == "TestAction"
        assert rec.principal_id == "hong"
        assert rec.principal_roles == ["engineer"]
        assert rec.reason == "테스트"
        assert rec.client_request_id == "req-1"
        assert rec.succeeded is True

    def test_agent_context_recorded(self, executor, sink):
        agent = Principal(
            user_id="fdc-agent", roles=("engineer",), is_agent=True, on_behalf_of="hong"
        )
        executor.execute(ActionRequest("TestAction", {"name": "x"}), agent)
        rec = sink.records[0]
        assert rec.is_agent is True
        assert rec.on_behalf_of == "hong"

    def test_handler_exception_is_audited_and_reraised(self, sink, engineer):
        registry = ActionRegistry({"TestAction": _definition()})

        def boom(p, pr, d):
            raise RuntimeError("DB 연결 실패")

        registry.register_handler("TestAction", boom)
        ex = ActionExecutor(registry, sink)
        with pytest.raises(RuntimeError):
            ex.execute(ActionRequest("TestAction", {"name": "x"}), engineer)
        assert sink.records[0].succeeded is False
        assert "DB 연결 실패" in sink.records[0].error

    def test_audit_sink_failure_does_not_break_action(self, engineer):
        class BrokenSink(InMemoryAuditSink):
            def write(self, record):
                raise IOError("디스크 가득 참")

        registry = ActionRegistry({"TestAction": _definition()})
        registry.register_handler("TestAction", lambda p, pr, d: {"message": "ok"})
        ex = ActionExecutor(registry, BrokenSink())
        # 감사 실패가 액션 결과를 뒤집지는 않는다 (로그로만 남는다)
        assert ex.execute(ActionRequest("TestAction", {"name": "x"}), engineer).succeeded


# --- 실제 스키마 YAML ---

@pytest.fixture(scope="module")
def definitions():
    return load_action_types()


@pytest.fixture(scope="module")
def real_registry(definitions):
    """실제 스키마 + 실제 핸들러로 구성한 레지스트리"""
    import sys
    import pathlib as _pathlib

    sys.path.insert(0, str(_pathlib.Path("api/src").resolve()))
    from services.action_handlers import register_handlers

    return register_handlers(ActionRegistry(definitions))


class TestRealSchemas:

    def test_schemas_load(self, definitions):
        assert definitions, "ontology/schemas/actions 에서 액션 타입을 로드하지 못했습니다"

    @pytest.mark.parametrize(
        "action_type",
        [
            "VerifyRelationship",
            "RecordExpertRelationship",
            "AcknowledgeAlarm",
            "UpdateEquipmentState",
            "HoldLot",
            "RequestInspection",
        ],
    )
    def test_expected_actions_exist(self, definitions, action_type):
        assert action_type in definitions

    def test_state_changing_actions_require_reason(self, definitions):
        # 생산에 영향을 주는 액션은 사유 없이 수행될 수 없어야 한다
        for name in ("HoldLot", "UpdateEquipmentState", "RequestInspection",
                     "VerifyRelationship"):
            assert definitions[name].reason_required, f"{name}은 reasonRequired 여야 합니다"

    def test_every_action_has_permissions(self, definitions):
        for name, d in definitions.items():
            assert d.allowed_roles, f"{name}에 permissions.roles 가 없습니다"

    def test_e10_transitions_are_defined(self, definitions):
        transitions = definitions["UpdateEquipmentState"].transitions
        assert transitions is not None
        # 정비/고장 종료 후 곧바로 생산으로 갈 수 없어야 한다
        assert "PRODUCTIVE" not in transitions["SCHEDULED_DOWNTIME"]
        assert "PRODUCTIVE" not in transitions["UNSCHEDULED_DOWNTIME"]
        assert "STANDBY" in transitions["UNSCHEDULED_DOWNTIME"]

    def test_e10_states_match_equipment_schema(self, definitions):
        import yaml
        eq = yaml.safe_load(
            open("ontology/schemas/objects/equipment.yaml", encoding="utf-8")
        )
        schema_states = set(eq["properties"]["status"]["values"])
        action_states = set(definitions["UpdateEquipmentState"].parameters["to_state"].values)
        assert schema_states == action_states, "설비 스키마와 액션의 E10 상태 목록이 다릅니다"

    def test_handlers_cover_all_declared_actions(self, definitions, real_registry):
        # 선언만 되고 구현이 없는 액션이 방치되지 않도록 확인
        missing = set(definitions) - set(real_registry.implemented())
        assert not missing, f"핸들러가 없는 액션: {sorted(missing)}"


class TestRealActionsEndToEnd:
    """실제 스키마 + 실제 핸들러로 수행해보는 통합 테스트"""

    @pytest.fixture
    def executor(self, real_registry):
        return ActionExecutor(real_registry, InMemoryAuditSink())

    @pytest.fixture
    def pe(self):
        return Principal(user_id="hong", roles=("process_engineer",))

    def test_verify_relationship(self, executor, pe):
        result = executor.execute(
            ActionRequest(
                "VerifyRelationship",
                {
                    "source": "flow_rate",
                    "target": "pressure",
                    "relation_type": "CAUSES",
                    "decision": "verified",
                },
                reason="스텝 구간 내 재현 확인, 물리적으로 타당",
            ),
            pe,
        )
        assert result.succeeded
        fields = {c["field"]: c["after"] for c in result.changes}
        assert fields["verification_status"] == "verified"
        assert fields["verified_by"] == "hong"

    def test_verify_relationship_rejects_bad_relation_type(self, executor, pe):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest(
                    "VerifyRelationship",
                    {
                        "source": "a",
                        "target": "b",
                        "relation_type": "X'}) DETACH DELETE (n) //",
                        "decision": "verified",
                    },
                    reason="테스트",
                ),
                pe,
            )
        assert "relation_type" in exc.value.errors

    def test_hold_lot_requires_reason(self, executor, pe):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest("HoldLot", {"lot_id": "LOT-001", "hold": True}), pe
            )
        assert "reason" in exc.value.errors

    def test_hold_lot_denied_for_operator(self, executor):
        with pytest.raises(PermissionDenied):
            executor.execute(
                ActionRequest(
                    "HoldLot", {"lot_id": "LOT-001", "hold": True}, reason="품질 이상"
                ),
                Principal(user_id="kim", roles=("operator",)),
            )

    def test_acknowledge_alarm_allows_operator(self, executor):
        result = executor.execute(
            ActionRequest("AcknowledgeAlarm", {"alarm_id": "ALM-001"}),
            Principal(user_id="kim", roles=("operator",)),
        )
        assert result.succeeded
        fields = {c["field"]: c["after"] for c in result.changes}
        assert fields["status"] == "ACKNOWLEDGED"
        # 하드코딩된 'current_user'가 아니라 실제 수행자가 기록된다
        assert fields["acknowledged_by"] == "kim"

    def test_acknowledge_alarm_escalation(self, executor):
        result = executor.execute(
            ActionRequest(
                "AcknowledgeAlarm",
                {"alarm_id": "ALM-001", "escalate_to": "supervisor.park"},
            ),
            Principal(user_id="kim", roles=("operator",)),
        )
        fields = {c["field"]: c["after"] for c in result.changes}
        assert fields["status"] == "ESCALATED"
        assert fields["escalated_to"] == "supervisor.park"

    def test_update_equipment_state_rejects_unknown_state(self, executor):
        with pytest.raises(ValidationFailed) as exc:
            executor.execute(
                ActionRequest(
                    "UpdateEquipmentState",
                    {"equipment_id": "EQP-001", "to_state": "RUNNING"},
                    reason="복구 완료",
                ),
                Principal(user_id="lee", roles=("equipment_engineer",)),
            )
        # 레거시 용어(RUNNING)는 더 이상 허용되지 않는다
        assert "to_state" in exc.value.errors

    def test_request_inspection_simulate_writes_nothing(self, executor, pe):
        result = executor.simulate(
            ActionRequest(
                "RequestInspection",
                {
                    "equipment_id": "EQP-001",
                    "chamber_id": "EQP-001.CH2",
                    "priority": "HIGH",
                    "checklist": "1. RF 정합 확인 2. 유량계 점검",
                },
                reason="flow_rate 이상 후속",
            ),
            pe,
        )
        assert result.dry_run is True
        assert result.message.startswith("[시뮬레이션]")

    def test_every_attempt_is_audited(self, executor, pe):
        executor.execute(
            ActionRequest("AcknowledgeAlarm", {"alarm_id": "ALM-001"}), pe
        )
        with pytest.raises(ValidationFailed):
            executor.execute(ActionRequest("HoldLot", {"lot_id": "L1", "hold": True}), pe)
        assert len(executor.audit_sink.records) == 2
        assert [r.succeeded for r in executor.audit_sink.records] == [True, False]
