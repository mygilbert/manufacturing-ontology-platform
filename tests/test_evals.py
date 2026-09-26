"""평가 프레임 테스트

평가 장치 자체가 틀리면 모든 숫자가 무의미해진다.
특히 세 가지를 지킨다.

  1. NOT_IMPLEMENTED 가 분모에 들어간다 (빠지면 진척이 부풀려진다)
  2. 채점기가 없는 케이스는 통과하지 않는다 (무엇을 검증하는지 적지 않은 케이스)
  3. Action 타겟은 항상 dry_run 이다 (평가가 상태를 바꾸면 한 번밖에 못 쓴다)
"""
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

yaml = pytest.importorskip("yaml")

from common.evals import (  # noqa: E402
    CaseResult,
    EvalRunner,
    GraderSpec,
    Priority,
    Status,
    SuiteResult,
    TargetSpec,
    TestCase,
    load_suite,
    run_grader,
)

SUITE_PATH = REPO / "evals/suites/competency_battery.yaml"


def make_case(cid="X1", target=None, graders=(), priority="P1") -> TestCase:
    return TestCase(
        id=cid, question="테스트",
        target=target or TargetSpec("not_implemented", {"note": "n/a"}),
        graders=graders, priority=Priority(priority),
    )


# ----------------------------------------------------------------------
class TestStatusSemantics:
    def test_not_implemented_counts_in_denominator(self):
        # 분모에서 빼면 "아직 못 한다"가 점수에 드러나지 않는다
        assert Status.NOT_IMPLEMENTED.counts_in_denominator is True

    def test_skipped_does_not_count(self):
        assert Status.SKIPPED.counts_in_denominator is False

    def test_only_passed_is_pass(self):
        assert Status.PASSED.is_pass
        for s in (Status.FAILED, Status.NOT_IMPLEMENTED, Status.ERROR, Status.SKIPPED):
            assert not s.is_pass


class TestAggregation:
    def _suite(self, statuses) -> SuiteResult:
        r = SuiteResult(suite="s", version="1", dataset="d")
        for i, st in enumerate(statuses):
            r.results.append(CaseResult(case=make_case(f"C{i}"), status=st))
        return r

    def test_pass_rate_includes_not_implemented(self):
        r = self._suite([Status.PASSED, Status.PASSED, Status.NOT_IMPLEMENTED])
        assert (r.passed, r.total) == (2, 3)
        assert round(r.pass_rate, 1) == 66.7

    def test_skipped_excluded_from_total(self):
        r = self._suite([Status.PASSED, Status.SKIPPED])
        assert (r.passed, r.total) == (1, 1)
        assert r.pass_rate == 100.0

    def test_empty_suite_does_not_divide_by_zero(self):
        assert self._suite([]).pass_rate == 0.0

    def test_by_priority_groups(self):
        r = SuiteResult(suite="s", version="1", dataset="d")
        r.results = [
            CaseResult(case=make_case("A", priority="P0"), status=Status.PASSED),
            CaseResult(case=make_case("B", priority="P0"), status=Status.FAILED),
            CaseResult(case=make_case("C", priority="P1"), status=Status.PASSED),
        ]
        assert r.by_priority() == {"P0": {"passed": 1, "total": 2},
                                   "P1": {"passed": 1, "total": 1}}


# ----------------------------------------------------------------------
class TestGraders:
    ROWS = {"rows": [{"a": 1, "b": "x"}, {"a": 5, "b": "y"}],
            "columns": ["a", "b"]}
    EMPTY = {"rows": [], "columns": ["a"]}

    def test_not_empty(self):
        assert run_grader("not_empty", {}, self.ROWS).passed
        assert not run_grader("not_empty", {}, self.EMPTY).passed

    def test_no_rows(self):
        assert run_grader("no_rows", {}, self.EMPTY).passed
        assert not run_grader("no_rows", {}, self.ROWS).passed

    def test_row_count_bounds(self):
        assert run_grader("row_count", {"min": 2}, self.ROWS).passed
        assert not run_grader("row_count", {"min": 3}, self.ROWS).passed
        assert not run_grader("row_count", {"max": 1}, self.ROWS).passed

    def test_columns_present_reports_missing(self):
        r = run_grader("columns_present", {"columns": ["a", "zz"]}, self.ROWS)
        assert not r.passed and "zz" in r.detail

    def test_top_row_equals(self):
        assert run_grader("top_row_equals", {"column": "b", "value": "x"}, self.ROWS).passed
        assert not run_grader("top_row_equals", {"column": "b", "value": "y"}, self.ROWS).passed

    def test_top_row_equals_on_empty_fails(self):
        assert not run_grader("top_row_equals", {"column": "a", "value": 1}, self.EMPTY).passed

    def test_value_between(self):
        assert run_grader("value_between", {"column": "a", "min": 0, "max": 2}, self.ROWS).passed
        assert not run_grader("value_between", {"column": "a", "min": 3}, self.ROWS).passed

    def test_value_between_null_fails(self):
        out = {"rows": [{"a": None}], "columns": ["a"]}
        assert not run_grader("value_between", {"column": "a", "min": 0}, out).passed

    def test_all_rows_match(self):
        assert run_grader("all_rows_match", {"column": "b", "allowed": ["x", "y"]}, self.ROWS).passed
        r = run_grader("all_rows_match", {"column": "b", "allowed": ["x"]}, self.ROWS)
        assert not r.passed and "y" in r.detail

    def test_column_values_include(self):
        assert run_grader("column_values_include",
                          {"column": "b", "values": ["x"]}, self.ROWS).passed
        assert not run_grader("column_values_include",
                              {"column": "b", "values": ["z"]}, self.ROWS).passed

    def test_unknown_grader_fails_loudly(self):
        r = run_grader("does_not_exist", {}, self.ROWS)
        assert not r.passed and "알 수 없는" in r.detail

    def test_grader_exception_becomes_failure(self):
        # 채점기가 터져도 스위트 전체가 멈추면 안 된다
        r = run_grader("value_between", {"column": "b", "min": 0}, self.ROWS)
        assert not r.passed


class TestActionGraders:
    OK = {"action": {"succeeded": True, "dry_run": True, "message": "ok",
                     "changes": [{"field": "status", "after": "STANDBY"}], "error": None}}
    DENIED = {"action": {"succeeded": False, "dry_run": True, "changes": [],
                         "error": "'X' 수행 권한이 없습니다"}}

    def test_action_succeeded(self):
        assert run_grader("action_succeeded", {}, self.OK).passed
        assert not run_grader("action_succeeded", {}, self.DENIED).passed

    def test_action_rejected(self):
        assert run_grader("action_rejected", {}, self.DENIED).passed
        assert not run_grader("action_rejected", {}, self.OK).passed

    def test_action_rejected_with_error_match(self):
        assert run_grader("action_rejected", {"error_contains": "권한"}, self.DENIED).passed
        assert not run_grader("action_rejected", {"error_contains": "사유"}, self.DENIED).passed

    def test_changes_include(self):
        assert run_grader("changes_include",
                          {"field": "status", "after": "STANDBY"}, self.OK).passed
        assert not run_grader("changes_include",
                              {"field": "status", "after": "PRODUCTIVE"}, self.OK).passed

    def test_changes_include_reports_available_fields(self):
        r = run_grader("changes_include", {"field": "nope"}, self.OK)
        assert not r.passed and "status" in r.detail

    def test_dry_run_wrote_nothing(self):
        assert run_grader("dry_run_wrote_nothing", {}, self.OK).passed
        wet = {"action": {"succeeded": True, "dry_run": False, "changes": []}}
        assert not run_grader("dry_run_wrote_nothing", {}, wet).passed


# ----------------------------------------------------------------------
class TestRunnerBehaviour:
    @pytest.fixture
    def runner(self):
        return EvalRunner(REPO / "landing")

    def test_not_implemented_target(self, runner):
        r = runner._run_case(make_case("N1"))
        assert r.status is Status.NOT_IMPLEMENTED
        assert r.evidence == "n/a"

    def test_case_without_graders_does_not_pass(self, runner):
        # 무엇을 검증하는지 적지 않은 케이스가 통과하면 점수가 거짓이 된다
        case = make_case("S1", TargetSpec("sql", {"query": "SELECT 1 AS a"}), graders=())
        r = runner._run_case(case)
        assert r.status is Status.FAILED
        assert r.graders[0].grader == "no_grader"

    def test_bad_sql_becomes_error_not_failure(self, runner):
        case = make_case("S2", TargetSpec("sql", {"query": "SELECT * FROM nope_x"}),
                         graders=(GraderSpec("not_empty"),))
        r = runner._run_case(case)
        assert r.status is Status.ERROR
        assert r.error

    def test_unknown_target_type_is_error(self, runner):
        case = make_case("S3", TargetSpec("telepathy", {}), graders=(GraderSpec("not_empty"),))
        assert runner._run_case(case).status is Status.ERROR

    def test_sql_case_passes_and_counts_rows(self, runner):
        case = make_case("S4", TargetSpec("sql", {"query": "SELECT 1 AS a UNION ALL SELECT 2"}),
                         graders=(GraderSpec("row_count", {"min": 2}),))
        r = runner._run_case(case)
        assert r.status is Status.PASSED and r.rows == 2


class TestActionTargetAlwaysDryRun:
    """평가가 실제 상태를 바꾸면 그 평가는 한 번밖에 못 쓴다"""

    def test_dry_run_is_forced(self):
        captured = {}

        class FakeExecutor:
            def execute(self, request, principal):
                captured["dry_run"] = request.dry_run
                captured["principal"] = principal.user_id

                class R:
                    succeeded, dry_run, changes, message = True, True, [], "ok"
                return R()

        runner = EvalRunner(REPO / "landing", action_executor=FakeExecutor())
        case = make_case("D9", TargetSpec("action", {
            "action_type": "UpdateEquipmentState",
            "parameters": {"equipment_id": "E1", "to_state": "STANDBY"},
            "reason": "테스트",
            "principal_roles": ["equipment_engineer"],
        }), graders=(GraderSpec("action_succeeded"),))
        runner._run_case(case)
        assert captured["dry_run"] is True


# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE_PATH)


class TestRealSuite:
    def test_suite_loads(self, suite):
        assert suite["cases"]

    def test_case_ids_unique(self, suite):
        ids = [c["id"] for c in suite["cases"]]
        dup = {i for i in ids if ids.count(i) > 1}
        assert not dup, f"중복 케이스 ID: {dup}"

    def test_every_case_parses(self, suite):
        for raw in suite["cases"]:
            TestCase.from_dict(raw)

    def test_runnable_cases_declare_graders(self, suite):
        # not_implemented 를 제외한 모든 케이스는 무엇을 검증하는지 밝혀야 한다
        missing = [c["id"] for c in suite["cases"]
                   if c["target"]["type"] != "not_implemented" and not c.get("graders")]
        assert not missing, f"채점기가 없는 케이스: {missing}"

    def test_not_implemented_cases_explain_why(self, suite):
        # "아직 못 한다"는 이유가 남아야 로드맵으로 이어진다
        bad = [c["id"] for c in suite["cases"]
               if c["target"]["type"] == "not_implemented"
               and not (c["target"].get("note") or c.get("note"))]
        assert not bad, f"사유가 없는 미구현 케이스: {bad}"

    def test_action_cases_never_declare_dry_run_false(self, suite):
        # 러너가 강제하지만 스위트에서도 혼동을 남기지 않는다
        bad = [c["id"] for c in suite["cases"]
               if c["target"]["type"] == "action" and c["target"].get("dry_run") is False]
        assert not bad

    def test_p0_cases_exist(self, suite):
        p0 = [c for c in suite["cases"] if c.get("priority") == "P0"]
        assert len(p0) >= 10, "P0 케이스가 너무 적으면 점수가 의미를 잃는다"


@pytest.mark.skipif(not (REPO / "landing").is_dir(),
                    reason="landing/ 없음 — generate_battery_genealogy.py 먼저 실행")
class TestEndToEnd:
    def test_full_suite_runs_and_scores(self):
        suite = load_suite(SUITE_PATH)
        result = EvalRunner(REPO / "landing").run(suite)
        assert result.total == len(suite["cases"])
        assert result.passed > 0
        # 합성 데이터에서 P0 계보 케이스는 반드시 통과해야 한다
        by_id = {r.case.id: r for r in result.results}
        for cid in ("A1", "A2", "A3", "B2", "B3"):
            assert by_id[cid].status is Status.PASSED, (
                f"{cid} 실패: {[g.detail for g in by_id[cid].failed_graders]}")

    def test_no_case_errors_on_synthetic_data(self):
        suite = load_suite(SUITE_PATH)
        result = EvalRunner(REPO / "landing").run(suite)
        errors = [(r.case.id, r.error) for r in result.results if r.status is Status.ERROR]
        assert not errors, f"합성 데이터에서 오류가 난 케이스: {errors}"
