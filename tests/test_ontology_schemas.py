"""온톨로지 스키마 일관성 테스트

스키마는 사람이 손으로 고치는 파일이라 조용히 어긋나기 쉽다.
특히 다음 세 가지가 어긋나면 런타임에 드러나지 않고 데이터만 오염된다.

  - 코어에 도메인 고유 개념이 새어 들어옴
  - 계보 체인이 끊김 (Roll -> ElectrodeLot -> Cell -> Module -> Pack)
  - 상태/enum 목록이 스키마와 액션 사이에서 불일치

코드가 아직 이 YAML 을 읽지 않으므로, 테스트가 유일한 방어선이다.
"""
import glob
import pathlib

import pytest

yaml = pytest.importorskip("yaml")

SCHEMA_ROOT = pathlib.Path("ontology/schemas")


def _load(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_dir(rel: str) -> dict:
    out = {}
    for p in sorted((SCHEMA_ROOT / rel).glob("*.yaml")):
        out[p.stem] = _load(p)
    return out


@pytest.fixture(scope="module")
def core_objects():
    return _load_dir("core/objects")


@pytest.fixture(scope="module")
def core_links():
    return _load_dir("core/links")


@pytest.fixture(scope="module")
def battery_objects():
    return _load_dir("profiles/battery/objects")


@pytest.fixture(scope="module")
def battery_links():
    return _load_dir("profiles/battery/links")


@pytest.fixture(scope="module")
def battery_profile():
    return _load(SCHEMA_ROOT / "profiles/battery/profile.yaml")


class TestAllSchemasParse:
    def test_every_yaml_parses(self):
        files = glob.glob(str(SCHEMA_ROOT / "**" / "*.yaml"), recursive=True)
        assert files, "스키마 파일을 찾지 못했습니다"
        for f in files:
            _load(pathlib.Path(f))

    def test_objects_declare_primary_key(self, core_objects, battery_objects):
        for name, d in {**core_objects, **battery_objects}.items():
            if d.get("deprecated"):
                continue
            assert d.get("primaryKey"), f"{name}: primaryKey 누락"

    def test_links_declare_source_and_target(self, core_links, battery_links):
        for name, d in {**core_links, **battery_links}.items():
            if d.get("deprecated"):
                continue
            assert d.get("source", {}).get("objectType"), f"{name}: source 누락"
            assert d.get("target", {}).get("objectType"), f"{name}: target 누락"


class TestCoreIsDomainNeutral:
    """코어에 도메인 고유 용어가 새어 들어오지 않아야 한다"""

    # 코어가 특정 도메인에 묶이면 다음 도메인에서 통째로 다시 만들어야 한다
    BATTERY_TERMS = ["Roll", "ElectrodeLot", "Cell", "Cathode", "Anode"]
    SEMI_TERMS = ["Wafer", "SEMI-E10", "SEMI-E120", "DRY_ETCH"]

    def test_core_has_no_semi_standard_reference(self, core_objects):
        for name, d in core_objects.items():
            ids = [s.get("id") for s in (d.get("standards") or [])]
            leaked = [i for i in ids if i and i.startswith("SEMI")]
            assert not leaked, f"core/{name}: 반도체 전용 표준 참조 {leaked}"

    def test_core_object_names_are_neutral(self, core_objects):
        names = {d.get("objectType") for d in core_objects.values()}
        for term in self.BATTERY_TERMS + self.SEMI_TERMS:
            assert term not in names, f"코어에 도메인 고유 객체 {term} 이 있습니다"

    def test_equipment_type_enum_moved_to_profile(self, core_objects):
        # 설비 유형은 도메인마다 다르므로 코어에서 enum 으로 고정하면 안 된다
        eq_type = core_objects["equipment"]["properties"]["type"]
        assert eq_type["type"] != "enum", "설비 type 은 프로파일에서 정의해야 합니다"


class TestEquipmentStateModel:
    EXPECTED = {
        "PRODUCTIVE", "STANDBY", "ENGINEERING",
        "SCHEDULED_DOWNTIME", "UNSCHEDULED_DOWNTIME", "NON_SCHEDULED",
    }

    def test_six_states(self, core_objects):
        states = set(core_objects["equipment"]["properties"]["status"]["values"])
        assert states == self.EXPECTED

    def test_every_state_maps_to_iso22400_time_element(self, core_objects):
        status = core_objects["equipment"]["properties"]["status"]
        mapping = status.get("iso22400Mapping")
        assert mapping, "ISO 22400 시간 요소 매핑이 없으면 OEE 집계가 사내와 어긋납니다"
        assert set(mapping) == self.EXPECTED
        assert mapping["PRODUCTIVE"] == "APT"
        assert mapping["SCHEDULED_DOWNTIME"] == "ADOT"
        assert mapping["UNSCHEDULED_DOWNTIME"] == "ADOT"

    def test_legacy_mapping_covers_old_codes(self, core_objects):
        legacy = core_objects["equipment"]["properties"]["status"]["legacyMapping"]
        assert legacy["RUNNING"] == "PRODUCTIVE"
        assert legacy["DOWN"] == "UNSCHEDULED_DOWNTIME"
        assert set(legacy.values()) <= self.EXPECTED

    def test_module_states_match_equipment(self, core_objects):
        mod = set(core_objects["equipment_module"]["properties"]["status"]["values"])
        assert mod == self.EXPECTED, "설비와 모듈의 상태 목록이 다릅니다"


class TestBatteryGenealogyChain:
    """연속 -> 이산 계보 체인이 끊기지 않아야 한다

    Roll --PRODUCES_LOT--> ElectrodeLot --SUPPLIES_CELL--> Cell --> Module --> Pack

    이 체인이 끊기면 근본원인 역추적도, 영향 범위 순추적도,
    배터리 여권의 셀 단위 추적도 전부 성립하지 않는다.
    """

    def test_chain_objects_exist(self, battery_objects):
        names = {d.get("objectType") for d in battery_objects.values()}
        for required in ("Roll", "WebSegment", "ElectrodeLot", "Cell", "Module", "Pack"):
            assert required in names, f"계보 객체 {required} 누락"

    def test_produces_lot_connects_roll_to_electrode_lot(self, battery_links):
        link = battery_links["produces_lot"]
        assert link["source"]["objectType"] == "Roll"
        assert link["target"]["objectType"] == "ElectrodeLot"

    def test_supplies_cell_connects_lot_to_cell(self, battery_links):
        link = battery_links["supplies_cell"]
        assert link["source"]["objectType"] == "ElectrodeLot"
        assert link["target"]["objectType"] == "Cell"

    def test_position_range_is_required_on_produces_lot(self, battery_links):
        # 이 두 필드가 없으면 셀에서 코팅 조건으로 역추적할 수 없다
        props = battery_links["produces_lot"]["properties"]
        for field in ("position_start_m", "position_end_m"):
            assert props[field]["required"] is True, f"{field} 는 필수여야 합니다"

    def test_electrode_lot_carries_position_range(self, battery_objects):
        props = battery_objects["electrode_lot"]["properties"]
        assert props["position_start_m"]["required"] is True
        assert props["position_end_m"]["required"] is True
        assert props["roll_id"]["required"] is True

    def test_electrode_role_is_cathode_or_anode(self, battery_objects, battery_links):
        lot_roles = set(battery_objects["electrode_lot"]["properties"]["electrode_role"]["values"])
        link_roles = set(battery_links["supplies_cell"]["properties"]["electrode_role"]["values"])
        assert lot_roles == {"CATHODE", "ANODE"}
        assert link_roles == lot_roles

    def test_cell_no_longer_assumes_single_source_roll(self, battery_objects):
        # 셀 하나에는 양극 롤과 음극 롤이 각각 들어간다.
        # 단일 source_roll_id 필드는 이 사실을 표현하지 못한다.
        props = battery_objects["cell"]["properties"]
        assert "source_roll_id" not in props, "단일 원본 롤 가정이 남아 있습니다"
        assert "cathode_lot_id" in props
        assert "anode_lot_id" in props

    def test_lot_boundary_cells_are_flagged(self, battery_links):
        # 경계 셀은 서로 다른 조건이 섞여 있어 분석에서 특별 취급이 필요하다
        assert "spans_lot_boundary" in battery_links["supplies_cell"]["properties"]

    def test_deprecated_direct_roll_to_cell_link(self, battery_links):
        produces_cell = battery_links.get("produces_cell")
        if produces_cell is None:
            return  # 이미 삭제됨
        assert produces_cell.get("deprecated") is True
        assert set(produces_cell.get("replacedBy", [])) == {"PRODUCES_LOT", "SUPPLIES_CELL"}


class TestTraceabilityLevel:
    """추적 신뢰도가 관계에 명시돼야 한다

    현실에서 셀<->롤위치 연결은 완벽하게 기록되지 않는 경우가 많다.
    레벨을 표시하지 않으면 추정 계보와 실측 계보가 섞여 분석이 오염된다.
    """

    LEVELS = {"MEASURED", "INFERRED", "ROLL_ONLY", "UNKNOWN"}

    def test_electrode_lot_declares_level(self, battery_objects):
        prop = battery_objects["electrode_lot"]["properties"]["traceability_level"]
        assert prop["required"] is True
        assert set(prop["values"]) == self.LEVELS

    def test_produces_lot_declares_level(self, battery_links):
        prop = battery_links["produces_lot"]["properties"]["traceability_level"]
        assert prop["required"] is True
        assert set(prop["values"]) == self.LEVELS

    def test_supplies_cell_declares_level(self, battery_links):
        prop = battery_links["supplies_cell"]["properties"]["traceability_level"]
        assert prop["required"] is True
        assert set(prop["values"]) <= self.LEVELS

    def test_position_confidence_on_web_data(self, battery_objects):
        # 시간->위치 변환이 깨질 수 있는 구간을 표시해야 한다
        for obj in ("web_segment", "web_measurement"):
            prop = battery_objects[obj]["properties"]["position_confidence"]
            assert set(prop["values"]) == {"HIGH", "MEDIUM", "LOW"}


class TestStorageDiscipline:
    """대량 시계열을 그래프에 넣지 않는다"""

    def test_measurements_are_not_graph_nodes(self, core_objects, battery_objects):
        for name, d in (("measurement", core_objects["measurement"]),
                        ("web_measurement", battery_objects["web_measurement"])):
            storage = d.get("storage") or {}
            assert storage.get("graphNode") is False, (
                f"{name}: 측정값을 그래프 노드로 두면 노드 수가 폭발합니다"
            )
            assert storage.get("primary") == "timescaledb"


class TestBatteryProfile:
    def test_declares_transition_point(self, battery_profile):
        units = battery_profile["traceabilityUnits"]
        by_stage = {u["stage"]: u for u in units}
        # 전극 단계에는 이산 ID 가 없고, 절단 단계에서 처음 부여된다
        assert by_stage["ELECTRODE"]["idAssigned"] is False
        assert by_stage["ELECTRODE_CUTTING"]["idAssigned"] is True
        assert by_stage["ELECTRODE_CUTTING"]["unit"] == "ElectrodeLot"

    def test_equipment_types_are_battery_specific(self, battery_profile):
        types = set(battery_profile["equipmentTypes"])
        assert {"COATER", "CALENDER", "SLITTER", "NOTCHER", "FORMATION"} <= types
        # 반도체 설비 유형이 섞여 있으면 프로파일 분리가 실패한 것이다
        assert not ({"DRY_ETCH", "CVD", "LITHO"} & types)

    def test_regulatory_link_to_battery_passport(self, battery_profile):
        reg = {r["id"]: r for r in battery_profile["regulatory"]}
        assert "EU-2023/1542" in reg
        assert reg["EU-2023/1542"]["appliesFrom"] == "2027-02-18"

    def test_profile_objects_exist_as_files(self, battery_profile, battery_objects):
        declared = set(battery_profile["objects"])
        actual = {d.get("objectType") for d in battery_objects.values()}
        missing = declared - actual
        assert not missing, f"프로파일이 선언했으나 스키마 파일이 없는 객체: {missing}"
