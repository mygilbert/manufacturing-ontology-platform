"""cypher_safe 유틸리티 테스트"""
import pytest

from common.cypher_safe import (
    CypherSafetyError,
    cy_bool,
    cy_ident,
    cy_int,
    cy_label,
    cy_num,
    cy_props,
    cy_rel_types,
    cy_str,
    cy_value,
)


class TestCyStr:
    def test_plain_value_is_quoted(self):
        assert cy_str("LOT-001") == "'LOT-001'"

    def test_none_becomes_null(self):
        assert cy_str(None) == "null"

    def test_single_quote_is_escaped(self):
        # 이스케이프하지 않으면 실제 현장 데이터에서 쿼리가 깨진다.
        assert cy_str("O'Brien") == r"'O\'Brien'"

    def test_backslash_is_escaped_before_quote(self):
        # 역슬래시를 나중에 치환하면 이중 이스케이프가 되어 깨진다.
        assert cy_str("a\\b") == r"'a\\b'"

    def test_injection_attempt_stays_inside_literal(self):
        malicious = "x'}) DETACH DELETE (n) //"
        result = cy_str(malicious)
        assert result.startswith("'") and result.endswith("'")
        # 따옴표가 이스케이프되어 리터럴을 빠져나가지 못한다.
        assert r"\'" in result
        assert result.count("'") - result.count(r"\'") == 2

    def test_newline_is_escaped(self):
        assert cy_str("a\nb") == r"'a\nb'"

    def test_control_characters_are_stripped(self):
        assert cy_str("a\x00b") == "'ab'"

    def test_dollar_quote_is_rejected(self):
        # $$ 는 AGE 쿼리 본문의 종료 구분자라 이스케이프가 불가능하다.
        with pytest.raises(CypherSafetyError):
            cy_str("abc$$def")

    def test_too_long_is_rejected(self):
        with pytest.raises(CypherSafetyError):
            cy_str("x" * 10, max_len=5)

    def test_datetime_uses_isoformat(self):
        from datetime import datetime

        assert cy_str(datetime(2026, 9, 26, 14, 3, 22)) == "'2026-09-26T14:03:22'"


class TestCyNum:
    def test_int_and_float(self):
        assert cy_num(42) == "42"
        assert cy_num(1.5) == "1.5"

    def test_numeric_string_is_converted(self):
        assert cy_num("3.5") == "3.5"

    def test_bool_is_rejected(self):
        with pytest.raises(CypherSafetyError):
            cy_num(True)

    def test_non_numeric_is_rejected(self):
        with pytest.raises(CypherSafetyError):
            cy_num("1) RETURN 1 //")

    def test_nan_and_inf_are_rejected(self):
        with pytest.raises(CypherSafetyError):
            cy_num(float("nan"))
        with pytest.raises(CypherSafetyError):
            cy_num(float("inf"))


class TestCyInt:
    def test_clamps_to_bounds(self):
        assert cy_int(-5) == "0"
        assert cy_int(999_999, maximum=1000) == "1000"

    def test_normal_value_passes(self):
        assert cy_int(50) == "50"

    def test_non_integer_is_rejected(self):
        with pytest.raises(CypherSafetyError):
            cy_int("10; DROP")


class TestCyBool:
    def test_values(self):
        assert cy_bool(True) == "true"
        assert cy_bool(False) == "false"


class TestCyValue:
    def test_dispatches_by_type(self):
        assert cy_value(None) == "null"
        assert cy_value(True) == "true"
        assert cy_value(7) == "7"
        assert cy_value("a") == "'a'"

    def test_list(self):
        assert cy_value([1, "a"]) == "[1, 'a']"


class TestIdentifiers:
    def test_valid_identifiers(self):
        assert cy_ident("equipment_id") == "equipment_id"
        assert cy_label("Equipment") == "Equipment"

    @pytest.mark.parametrize(
        "bad",
        [
            "equipment id",       # 공백
            "1abc",               # 숫자 시작
            "e'}) DELETE (n) //",  # 인젝션 시도
            "",                   # 빈 문자열
            "x" * 65,             # 길이 초과
            "prop-name",          # 하이픈
        ],
    )
    def test_invalid_identifiers_are_rejected(self, bad):
        with pytest.raises(CypherSafetyError):
            cy_ident(bad)


class TestCyRelTypes:
    def test_empty_returns_empty_string(self):
        assert cy_rel_types(None) == ""
        assert cy_rel_types([]) == ""

    def test_joins_with_pipe(self):
        assert cy_rel_types(["PROCESSED_AT", "BELONGS_TO"]) == "PROCESSED_AT|BELONGS_TO"

    def test_invalid_member_is_rejected(self):
        with pytest.raises(CypherSafetyError):
            cy_rel_types(["PROCESSED_AT", "X] DETACH DELETE (n) //"])


class TestCyProps:
    def test_empty_returns_empty_string(self):
        assert cy_props(None) == ""
        assert cy_props({}) == ""

    def test_builds_property_map(self):
        assert cy_props({"a": 1, "b": "x"}) == "{a: 1, b: 'x'}"

    def test_none_values_are_skipped_by_default(self):
        assert cy_props({"a": 1, "b": None}) == "{a: 1}"

    def test_none_values_kept_when_requested(self):
        assert cy_props({"a": None}, skip_none=False) == "{a: null}"

    def test_all_none_returns_empty_string(self):
        assert cy_props({"a": None}) == ""

    def test_quotes_in_values_are_escaped(self):
        assert cy_props({"name": "O'Brien"}) == r"{name: 'O\'Brien'}"

    def test_malicious_key_is_rejected(self):
        with pytest.raises(CypherSafetyError):
            cy_props({"a}) DETACH DELETE (n) //": 1})
