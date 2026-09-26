#!/usr/bin/env python3
"""배터리 계보 합성 데이터 생성기

실데이터가 들어오기 전에 온톨로지 설계를 검증하기 위한 픽스처.
사내 데이터를 전혀 쓰지 않으므로 반출 경계와 무관하다.

생성 구조는 실제 수집 형태와 동일하게 맞춘다.
  - Parquet
  - landing/<table>/dt=YYYY-MM-DD/hour=HH/ 파티션
  - 시간 배치(1시간 단위)로 나뉘어 도착

생성 대상
  rolls                전극 롤 (연속체)
  web_measurements     위치 인덱스 측정값 (코팅 두께 등)
  roll_position_map    시각 <-> 위치 변환 구간
  electrode_lots       전극 Lot  <- 연속/이산 전환점
  cells                셀 (양극/음극 Lot 이 수렴)
  cell_tests           EOL 용량 측정
  modules, packs       조립 계보

[의도적으로 심어 둔 것]
  특정 롤의 특정 레인, 특정 위치 구간에 코팅 두께 이상을 주입한다.
  그 구간에서 나온 전극 Lot 을 쓴 셀만 용량이 떨어지게 만든다.
  demo_genealogy_queries.py 가 이 사실을 **모르는 상태에서**
  계보를 타고 되짚어 찾아낼 수 있는지가 설계 검증의 핵심이다.

사용법
  python scripts/generate_battery_genealogy.py [--out landing] [--seed 42]
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 공정 파라미터 (합성. 실제 값 아님)
# ----------------------------------------------------------------------

ROLL_LENGTH_M = 500.0          # 롤 길이
MEASURE_PITCH_M = 0.5          # 측정 간격
LANES = [1, 2]                 # 폭 방향 레인
LOT_SPAN_M = 40.0              # 전극 Lot 하나가 덮는 롤 구간
CELLS_PER_LOT = 25             # Lot 하나에서 나오는 셀 수
CELLS_PER_MODULE = 12
MODULES_PER_PACK = 8
LINE_SPEED_MPM = 30.0          # 라인 속도 (m/min)

# 코팅 두께 기준 (um)
THICKNESS_TARGET = 120.0
THICKNESS_USL = 126.0
THICKNESS_LSL = 114.0
THICKNESS_SIGMA = 1.6

# 셀 용량 기준 (Ah)
CAPACITY_TARGET = 52.0
CAPACITY_SIGMA = 0.35
CAPACITY_SPEC_MIN = 50.5

# ★ 주입할 이상: (롤, 레인, 시작 m, 끝 m, 두께 변화량)
INJECTED_ANOMALY = {
    "roll_id": "R-CA-20260901-001",
    "lane_no": 1,
    "start_m": 120.0,
    "end_m": 135.0,
    "thickness_shift_um": 7.5,   # 두께가 두꺼워짐 -> 활물질 과다 -> 용량 편차
}
# 두께 이상이 용량에 미치는 영향 (도메인 인과: COATING_THICKNESS -> CELL_CAPACITY)
CAPACITY_PENALTY_PER_UM = 0.22


@dataclass
class RollSpec:
    roll_id: str
    electrode_role: str          # CATHODE / ANODE
    material_lot: str
    started_at: datetime


def build_rolls(base_time: datetime) -> list[RollSpec]:
    return [
        RollSpec("R-CA-20260901-001", "CATHODE", "MAT-NCM-2608-11", base_time),
        RollSpec("R-CA-20260901-002", "CATHODE", "MAT-NCM-2608-11",
                 base_time + timedelta(hours=4)),
        RollSpec("R-AN-20260901-001", "ANODE", "MAT-GRP-2608-07", base_time),
        RollSpec("R-AN-20260901-002", "ANODE", "MAT-GRP-2608-07",
                 base_time + timedelta(hours=4)),
    ]


# ----------------------------------------------------------------------
# 1. 웹 측정값 (위치 인덱스) + 위치 변환 맵
# ----------------------------------------------------------------------

def generate_web_measurements(rolls: list[RollSpec], rng: np.random.Generator):
    """코팅 두께를 롤 위치의 함수로 생성한다.

    실제 코팅 공정처럼 저주파 드리프트 + 레인 편차 + 노이즈를 섞고,
    지정한 구간에만 이상을 주입한다.
    """
    meas_rows = []
    pos_map_rows = []

    positions = np.arange(0.0, ROLL_LENGTH_M, MEASURE_PITCH_M)

    for roll in rolls:
        # 라인 속도는 완전히 일정하지 않다. 가감속 구간을 만든다.
        speed = np.full(positions.shape, LINE_SPEED_MPM)
        speed[:40] = np.linspace(8.0, LINE_SPEED_MPM, 40)          # 초기 가속
        speed[-40:] = np.linspace(LINE_SPEED_MPM, 8.0, 40)         # 말단 감속
        stop_start, stop_end = 640, 660                            # 320~330m 부근 정지
        speed[stop_start:stop_end] = 0.5

        # 위치 -> 시각 (속도 적분의 역산)
        dt_min = MEASURE_PITCH_M / np.maximum(speed, 0.1)
        elapsed_min = np.cumsum(dt_min)
        times = [roll.started_at + timedelta(minutes=float(m)) for m in elapsed_min]

        # 위치 신뢰도: 가감속/정지 구간은 시간->위치 변환이 깨진다
        confidence = np.full(positions.shape, "MEDIUM", dtype=object)
        confidence[:40] = "LOW"
        confidence[-40:] = "LOW"
        confidence[stop_start:stop_end] = "LOW"

        # 위치 변환 맵 (구간별 근거)
        for label, sl, seg_type, conf in [
            ("accel", slice(0, 40), "ACCEL", "LOW"),
            ("steady1", slice(40, stop_start), "STEADY", "MEDIUM"),
            ("stopped", slice(stop_start, stop_end), "STOPPED", "LOW"),
            ("steady2", slice(stop_end, len(positions) - 40), "STEADY", "MEDIUM"),
            ("decel", slice(len(positions) - 40, len(positions)), "DECEL", "LOW"),
        ]:
            seg_pos = positions[sl]
            seg_time = times[sl]
            if len(seg_pos) == 0:
                continue
            pos_map_rows.append({
                "roll_id": roll.roll_id,
                "time_start": seg_time[0],
                "time_end": seg_time[-1],
                "position_start_m": float(seg_pos[0]),
                "position_end_m": float(seg_pos[-1]),
                "avg_speed_mpm": float(np.mean(speed[sl])),
                "segment_type": seg_type,
                "confidence": conf,
            })

        for lane in LANES:
            # 저주파 드리프트: 코팅 공정의 실제 거동
            drift = 1.1 * np.sin(2 * np.pi * positions / 180.0 + lane)
            # 레인 편차: 레인 2가 약간 두껍다 (다이 갭 차이)
            lane_bias = 0.0 if lane == 1 else 0.9
            noise = rng.normal(0, THICKNESS_SIGMA, positions.shape)

            thickness = THICKNESS_TARGET + drift + lane_bias + noise

            # ★ 이상 주입
            a = INJECTED_ANOMALY
            if roll.roll_id == a["roll_id"] and lane == a["lane_no"]:
                mask = (positions >= a["start_m"]) & (positions < a["end_m"])
                # 계단이 아니라 완만한 진입/이탈 (실제 공정 거동)
                ramp = np.zeros_like(positions)
                ramp[mask] = a["thickness_shift_um"]
                ramp = np.convolve(ramp, np.ones(6) / 6, mode="same")
                thickness = thickness + ramp

            for i, pos in enumerate(positions):
                meas_rows.append({
                    "time": times[i],
                    "roll_id": roll.roll_id,
                    "position_m": float(pos),
                    "lane_no": lane,
                    "sensor_id": f"COATER-01.LANE{lane}.THICKNESS",
                    "observed_property": "coating_thickness",
                    "param_id": f"TH_L{lane}",
                    "value": float(thickness[i]),
                    "unit": "um",
                    "usl": THICKNESS_USL,
                    "lsl": THICKNESS_LSL,
                    "target": THICKNESS_TARGET,
                    "line_speed_mpm": float(speed[i]),
                    "position_confidence": confidence[i],
                    "equipment_id": "COATER-01",
                    "recipe_id": "COAT-NCM-A1" if roll.electrode_role == "CATHODE" else "COAT-GRP-B2",
                    "source_system": "FDC",
                })

    return pd.DataFrame(meas_rows), pd.DataFrame(pos_map_rows)


# ----------------------------------------------------------------------
# 2. 전극 Lot (연속 -> 이산 전환점)
# ----------------------------------------------------------------------

def generate_electrode_lots(rolls: list[RollSpec], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for roll in rolls:
        role_tag = "CA" if roll.electrode_role == "CATHODE" else "AN"
        seq = 0
        for lane in LANES:
            start = 0.0
            while start < ROLL_LENGTH_M:
                end = min(start + LOT_SPAN_M, ROLL_LENGTH_M)
                seq += 1

                # 현실 반영: 일부 Lot 은 위치가 실측되지 않고 추정된다
                level = rng.choice(
                    ["MEASURED", "INFERRED", "ROLL_ONLY"], p=[0.75, 0.20, 0.05]
                )
                rows.append({
                    "electrode_lot_id": f"ELOT-{role_tag}-{roll.roll_id[-3:]}-{seq:03d}",
                    "electrode_role": roll.electrode_role,
                    "roll_id": roll.roll_id,
                    "position_start_m": float(start),
                    "position_end_m": float(end),
                    "lane_no": lane,
                    "traceability_level": level,
                    "sheet_count": int(round((end - start) * 6)),
                    "slitting_equipment_id": "SLITTER-01",
                    "notching_equipment_id": "NOTCHER-01",
                    "created_at": roll.started_at + timedelta(hours=6, minutes=seq * 3),
                    "status": "CONSUMED",
                    "material_lot": roll.material_lot,
                })
                start = end
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. 셀 (양극 Lot + 음극 Lot 수렴) + EOL 용량
# ----------------------------------------------------------------------

def generate_cells(lots: pd.DataFrame, rng: np.random.Generator):
    """양극 Lot 과 음극 Lot 을 짝지어 셀을 만든다.

    용량은 투입된 양극 두께의 함수로 만든다 (도메인 인과 반영).
    단, 생성기는 "어느 Lot 이 이상인지"를 셀에 직접 표시하지 않는다.
    질의가 계보를 타고 스스로 찾아내야 한다.
    """
    cathode = lots[lots.electrode_role == "CATHODE"].reset_index(drop=True)
    anode = lots[lots.electrode_role == "ANODE"].reset_index(drop=True)
    n_pairs = min(len(cathode), len(anode))

    a = INJECTED_ANOMALY
    cell_rows, link_rows, test_rows = [], [], []
    cell_seq = 0

    for i in range(n_pairs):
        ca = cathode.iloc[i]
        an = anode.iloc[i]

        # 이 양극 Lot 이 이상 구간과 겹치는 비율 (생성기만 아는 사실)
        overlap = 0.0
        if ca.roll_id == a["roll_id"] and ca.lane_no == a["lane_no"]:
            lo = max(ca.position_start_m, a["start_m"])
            hi = min(ca.position_end_m, a["end_m"])
            if hi > lo:
                overlap = (hi - lo) / (ca.position_end_m - ca.position_start_m)

        for k in range(CELLS_PER_LOT):
            cell_seq += 1
            cell_id = f"CELL-2609-{cell_seq:05d}"
            # Lot 경계에 걸친 셀 (전체의 약 4%)
            spans_boundary = (k == CELLS_PER_LOT - 1) and (rng.random() < 0.4)

            penalty = overlap * a["thickness_shift_um"] * CAPACITY_PENALTY_PER_UM
            capacity = rng.normal(CAPACITY_TARGET - penalty, CAPACITY_SIGMA)

            cell_rows.append({
                "cell_id": cell_id,
                "cell_type": "POUCH",
                "cell_model": "NCM-52Ah",
                "cathode_lot_id": ca.electrode_lot_id,
                "anode_lot_id": an.electrode_lot_id,
                "genealogy_complete": bool(
                    ca.traceability_level != "ROLL_ONLY"
                    and an.traceability_level != "ROLL_ONLY"
                ),
                "assembled_at": ca.created_at + timedelta(minutes=k * 2),
                "stacking_equipment_id": "STACKER-02",
            })

            for lot, role in ((ca, "CATHODE"), (an, "ANODE")):
                link_rows.append({
                    "electrode_lot_id": lot.electrode_lot_id,
                    "cell_id": cell_id,
                    "electrode_role": role,
                    "sheet_count": 42,
                    "spans_lot_boundary": bool(spans_boundary),
                    "stacking_equipment_id": "STACKER-02",
                    "assembled_at": ca.created_at + timedelta(minutes=k * 2),
                    "traceability_level": "MEASURED",
                })

            test_rows.append({
                "cell_id": cell_id,
                "test_type": "EOL_CAPACITY",
                "value": float(capacity),
                "unit": "Ah",
                "spec_min": CAPACITY_SPEC_MIN,
                "result": "PASS" if capacity >= CAPACITY_SPEC_MIN else "FAIL",
                "tested_at": ca.created_at + timedelta(days=3, minutes=k * 2),
                "formation_channel": f"FORMATION-01.CH{(cell_seq % 64) + 1}",
            })

    return (pd.DataFrame(cell_rows), pd.DataFrame(link_rows), pd.DataFrame(test_rows))


# ----------------------------------------------------------------------
# 4. 모듈 / 팩
# ----------------------------------------------------------------------

def generate_modules_packs(cells: pd.DataFrame):
    mod_rows, mod_links, pack_rows, pack_links = [], [], [], []
    cell_ids = cells.cell_id.tolist()

    for mi in range(len(cell_ids) // CELLS_PER_MODULE):
        module_id = f"MOD-2609-{mi + 1:04d}"
        mod_rows.append({"module_id": module_id, "module_model": "MOD-12S"})
        for cid in cell_ids[mi * CELLS_PER_MODULE:(mi + 1) * CELLS_PER_MODULE]:
            mod_links.append({"cell_id": cid, "module_id": module_id})

    module_ids = [r["module_id"] for r in mod_rows]
    for pi in range(len(module_ids) // MODULES_PER_PACK):
        pack_id = f"PACK-2609-{pi + 1:04d}"
        pack_rows.append({"pack_id": pack_id, "pack_model": "PACK-96S"})
        for mid in module_ids[pi * MODULES_PER_PACK:(pi + 1) * MODULES_PER_PACK]:
            pack_links.append({"module_id": mid, "pack_id": pack_id})

    return (pd.DataFrame(mod_rows), pd.DataFrame(mod_links),
            pd.DataFrame(pack_rows), pd.DataFrame(pack_links))


# ----------------------------------------------------------------------
# 저장: 실제 수집 형태(시간 파티션 Parquet)와 동일하게
# ----------------------------------------------------------------------

def write_partitioned(df: pd.DataFrame, out: pathlib.Path, table: str,
                      time_col: str | None) -> int:
    """시간 컬럼이 있으면 dt/hour 파티션으로, 없으면 단일 파일로 저장한다."""
    base = out / table
    if time_col is None or time_col not in df.columns:
        base.mkdir(parents=True, exist_ok=True)
        df.to_parquet(base / "part-000.parquet", index=False)
        (base / "_SUCCESS").touch()
        return 1

    ts = pd.to_datetime(df[time_col])
    parts = 0
    for (d, h), chunk in df.groupby([ts.dt.strftime("%Y-%m-%d"), ts.dt.hour]):
        folder = base / f"dt={d}" / f"hour={h:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        chunk.to_parquet(folder / "part-000.parquet", index=False)
        # 부분 쓰기 파일을 읽지 않도록 완료 마커를 마지막에 만든다
        (folder / "_SUCCESS").touch()
        parts += 1
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="landing", help="랜딩 폴더 (기본: landing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = pathlib.Path(args.out)
    base_time = datetime(2026, 9, 1, 6, 0, 0)

    rolls = build_rolls(base_time)
    roll_df = pd.DataFrame([{
        "roll_id": r.roll_id,
        "electrode_role": r.electrode_role,
        "material_lot": r.material_lot,
        "length_m": ROLL_LENGTH_M,
        "coating_recipe": "COAT-NCM-A1" if r.electrode_role == "CATHODE" else "COAT-GRP-B2",
        "started_at": r.started_at,
        "equipment_id": "COATER-01",
    } for r in rolls])

    print("전극 롤 생성...")
    web, pos_map = generate_web_measurements(rolls, rng)
    print(f"  롤 {len(roll_df)}개, 웹 측정값 {len(web):,}행, 위치맵 {len(pos_map)}구간")

    print("전극 Lot 생성 (연속 -> 이산 전환)...")
    lots = generate_electrode_lots(rolls, rng)
    print(f"  전극 Lot {len(lots)}개")

    print("셀 조립 (양극/음극 Lot 수렴)...")
    cells, supplies, tests = generate_cells(lots, rng)
    print(f"  셀 {len(cells):,}개, 계보 엣지 {len(supplies):,}개, EOL 시험 {len(tests):,}건")

    print("모듈/팩 조립...")
    modules, mod_links, packs, pack_links = generate_modules_packs(cells)
    print(f"  모듈 {len(modules)}개, 팩 {len(packs)}개")

    tables = [
        ("rolls", roll_df, "started_at"),
        ("web_measurements", web, "time"),
        ("roll_position_map", pos_map, None),
        ("electrode_lots", lots, "created_at"),
        ("cells", cells, "assembled_at"),
        ("supplies_cell", supplies, "assembled_at"),
        ("cell_tests", tests, "tested_at"),
        ("modules", modules, None),
        ("module_cells", mod_links, None),
        ("packs", packs, None),
        ("pack_modules", pack_links, None),
    ]

    print(f"\nParquet 저장 -> {out}/")
    total = 0
    for name, df, tcol in tables:
        n = write_partitioned(df, out, name, tcol)
        total += n
        print(f"  {name:<20} {len(df):>8,}행  ({n} 파티션)")

    print(f"\n완료. 파티션 {total}개.")
    print("다음: python scripts/demo_genealogy_queries.py")


if __name__ == "__main__":
    main()
