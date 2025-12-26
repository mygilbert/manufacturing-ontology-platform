"""
?? 諛泥?FAB ?⑹ ?곗???깃?
? ?곗??
- Equipment: 50+ ?ㅻ (Etch, CVD, PVD, Litho, CMP, Metrology)
- Process: 20+ 怨듭 ?④
- Recipe: 30+ ???- Lot/Wafer: 100+ 濡, 2500+ ?⑥??- FDC 痢≪: 100留? ?怨???곗???ъ??- Alarm: 500+ ? ?대
- SPC: ?/寃쎄/OOC ?⑦ ?ы
"""

import random
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import asyncio
import asyncpg

# ============================================================
# ?ㅼ
# ============================================================

POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "ontology",
    "password": "ontology123",
    "database": "manufacturing"
}

TIMESCALE_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "user": "timescale",
    "password": "timescale123",
    "database": "measurements"
}

# ?媛 踰 (理洹 30??
END_TIME = datetime.now()
START_TIME = END_TIME - timedelta(days=30)

# ============================================================
# ============================================================

EQUIPMENT_TYPES = {
    "DRY_ETCH": {
        "prefix": "ETCH",
        "count": 12,
        "manufacturers": ["Applied Materials", "LAM Research", "TEL"],
        "models": ["Centura 5200", "Kiyo FX", "Tactras Vigus"],
        "chambers": [2, 4],
        "params": ["RF_Power", "Chamber_Pressure", "Gas_Flow_CF4", "Gas_Flow_O2", "Chuck_Temp", "ESC_Voltage"]
    },
    "CVD": {
        "prefix": "CVD",
        "count": 8,
        "manufacturers": ["Applied Materials", "LAM Research", "ASM"],
        "models": ["Producer SE", "Vector Express", "Epsilon 3200"],
        "chambers": [4, 6],
        "params": ["Heater_Temp", "Chamber_Pressure", "Gas_Flow_SiH4", "Gas_Flow_N2O", "RF_Power", "Thickness"]
    },
    "PVD": {
        "prefix": "PVD",
        "count": 6,
        "manufacturers": ["Applied Materials", "Ulvac", "Canon Anelva"],
        "models": ["Endura", "SME-200", "C-7100"],
        "chambers": [2, 4],
        "params": ["DC_Power", "Chamber_Pressure", "Ar_Flow", "Target_Voltage", "Substrate_Temp"]
    },
    "LITHO": {
        "prefix": "LITHO",
        "count": 8,
        "manufacturers": ["ASML", "Nikon", "Canon"],
        "models": ["TWINSCAN NXE:3400C", "NSR-S631E", "FPA-6300ES6a"],
        "chambers": [1],
        "params": ["Focus", "Dose", "Overlay_X", "Overlay_Y", "CD", "Stage_Temp"]
    },
    "CMP": {
        "prefix": "CMP",
        "count": 6,
        "manufacturers": ["Applied Materials", "Ebara", "KCTECH"],
        "models": ["Reflexion LK", "F-REX300", "POLI-400L"],
        "chambers": [2, 3],
        "params": ["Down_Force", "Platen_Speed", "Head_Speed", "Slurry_Flow", "Pad_Temp", "Removal_Rate"]
    },
    "DIFFUSION": {
        "prefix": "DIFF",
        "count": 4,
        "manufacturers": ["TEL", "ASM", "Kokusai"],
        "models": ["Alpha 8SE", "A412", "Vertex"],
        "chambers": [1],
        "params": ["Zone1_Temp", "Zone2_Temp", "Zone3_Temp", "Gas_Flow_N2", "Gas_Flow_O2", "Pressure"]
    },
    "IMPLANT": {
        "prefix": "IMP",
        "count": 4,
        "manufacturers": ["Applied Materials", "Axcelis", "AIBT"],
        "models": ["VIISta 900", "Purion H", "iPulsar"],
        "chambers": [1],
        "params": ["Beam_Current", "Beam_Energy", "Dose", "Tilt_Angle", "Twist_Angle", "Wafer_Temp"]
    },
    "METROLOGY": {
        "prefix": "METRO",
        "count": 6,
        "manufacturers": ["KLA", "ASML", "Hitachi"],
        "models": ["SpectraFilm F1", "YieldStar S-250D", "CG6300"],
        "chambers": [1],
        "params": ["Measurement_Value", "Repeatability", "TIS"]
    }
}

# 怨듭 ? (DRAM 怨듭 ?由)
PROCESS_FLOW = [
    {"id": "PHOTO-01", "name": "Active Photo", "category": "PHOTO", "equipment_type": "LITHO", "target_ct": 45},
    {"id": "ETCH-01", "name": "Active Etch", "category": "ETCH", "equipment_type": "DRY_ETCH", "target_ct": 30},
    {"id": "CLEAN-01", "name": "Post Etch Clean", "category": "WET", "equipment_type": "WET_CLEAN", "target_ct": 15},
    {"id": "DIFF-01", "name": "Gate Oxidation", "category": "DIFF", "equipment_type": "DIFFUSION", "target_ct": 120},
    {"id": "CVD-01", "name": "Poly Si Deposition", "category": "CVD", "equipment_type": "CVD", "target_ct": 60},
    {"id": "PHOTO-02", "name": "Gate Photo", "category": "PHOTO", "equipment_type": "LITHO", "target_ct": 45},
    {"id": "ETCH-02", "name": "Gate Etch", "category": "ETCH", "equipment_type": "DRY_ETCH", "target_ct": 35},
    {"id": "IMP-01", "name": "LDD Implant", "category": "IMPLANT", "equipment_type": "IMPLANT", "target_ct": 25},
    {"id": "CVD-02", "name": "Spacer Deposition", "category": "CVD", "equipment_type": "CVD", "target_ct": 45},
    {"id": "ETCH-03", "name": "Spacer Etch", "category": "ETCH", "equipment_type": "DRY_ETCH", "target_ct": 25},
    {"id": "IMP-02", "name": "S/D Implant", "category": "IMPLANT", "equipment_type": "IMPLANT", "target_ct": 30},
    {"id": "DIFF-02", "name": "Anneal", "category": "DIFF", "equipment_type": "DIFFUSION", "target_ct": 60},
    {"id": "PVD-01", "name": "Contact Metal", "category": "PVD", "equipment_type": "PVD", "target_ct": 40},
    {"id": "CVD-03", "name": "ILD Deposition", "category": "CVD", "equipment_type": "CVD", "target_ct": 55},
    {"id": "CMP-01", "name": "ILD CMP", "category": "CMP", "equipment_type": "CMP", "target_ct": 35},
    {"id": "PHOTO-03", "name": "Contact Photo", "category": "PHOTO", "equipment_type": "LITHO", "target_ct": 45},
    {"id": "ETCH-04", "name": "Contact Etch", "category": "ETCH", "equipment_type": "DRY_ETCH", "target_ct": 40},
    {"id": "PVD-02", "name": "Metal 1 Deposition", "category": "PVD", "equipment_type": "PVD", "target_ct": 50},
    {"id": "METRO-01", "name": "CD Measurement", "category": "METRO", "equipment_type": "METROLOGY", "target_ct": 15},
    {"id": "METRO-02", "name": "Overlay Measurement", "category": "METRO", "equipment_type": "METROLOGY", "target_ct": 15},
]

# ? ?
PRODUCTS = [
    {"code": "DRAM-8Gb-DDR5", "name": "8Gb DDR5 SDRAM", "priority_weight": 0.4},
    {"code": "DRAM-16Gb-DDR5", "name": "16Gb DDR5 SDRAM", "priority_weight": 0.3},
    {"code": "DRAM-4Gb-LPDDR5", "name": "4Gb LPDDR5", "priority_weight": 0.2},
    {"code": "DRAM-8Gb-LPDDR5", "name": "8Gb LPDDR5", "priority_weight": 0.1},
]

# ? 肄 ?
ALARM_CODES = {
    "FDC": [
        {"code": "FDC-PRESS-HIGH", "name": "Chamber Pressure High", "severity": "MAJOR"},
        {"code": "FDC-PRESS-LOW", "name": "Chamber Pressure Low", "severity": "MAJOR"},
        {"code": "FDC-TEMP-HIGH", "name": "Temperature High", "severity": "CRITICAL"},
        {"code": "FDC-TEMP-LOW", "name": "Temperature Low", "severity": "WARNING"},
        {"code": "FDC-FLOW-DEV", "name": "Gas Flow Deviation", "severity": "MAJOR"},
        {"code": "FDC-RF-UNSTABLE", "name": "RF Power Unstable", "severity": "MAJOR"},
        {"code": "FDC-DRIFT", "name": "Parameter Drift Detected", "severity": "WARNING"},
    ],
    "SPC": [
        {"code": "SPC-RULE1", "name": "Point Beyond Control Limit", "severity": "MAJOR"},
        {"code": "SPC-RULE2", "name": "9 Points Same Side", "severity": "WARNING"},
        {"code": "SPC-RULE3", "name": "6 Points Trending", "severity": "WARNING"},
        {"code": "SPC-RULE4", "name": "14 Points Alternating", "severity": "WARNING"},
        {"code": "SPC-CPK-LOW", "name": "Cpk Below Threshold", "severity": "MAJOR"},
    ],
    "EQP": [
        {"code": "EQP-PM-DUE", "name": "PM Due", "severity": "WARNING"},
        {"code": "EQP-PART-LIFE", "name": "Part Life Exceeded", "severity": "MAJOR"},
        {"code": "EQP-INTERLOCK", "name": "Safety Interlock", "severity": "CRITICAL"},
        {"code": "EQP-COMM-FAIL", "name": "Communication Failure", "severity": "CRITICAL"},
    ]
}


# ============================================================
# ?곗??? ?⑥
# ============================================================

def generate_equipment() -> List[Dict]:
    """?ㅻ ?곗???"""
    equipment_list = []

    for eq_type, config in EQUIPMENT_TYPES.items():
        for i in range(1, config["count"] + 1):
            eq_id = f"EQP-{config['prefix']}-{i:03d}"

            # ? 遺: RUNNING 70%, IDLE 15%, MAINTENANCE 10%, ERROR 5%
            status_rand = random.random()
            if status_rand < 0.70:
                status = "RUNNING"
            elif status_rand < 0.85:
                status = "IDLE"
            elif status_rand < 0.95:
                status = "MAINTENANCE"
            else:
                status = "ERROR"

            equipment = {
                "equipment_id": eq_id,
                "name": f"{config['prefix']}-{i:02d}",
                "type": eq_type,
                "status": status,
                "location": f"FAB1-ZONE{random.randint(1, 4)}",
                "manufacturer": random.choice(config["manufacturers"]),
                "model": random.choice(config["models"]),
                "chamber_count": random.choice(config["chambers"]),
                "install_date": (datetime.now() - timedelta(days=random.randint(365, 1825))).isoformat(),
                "last_pm_date": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                "pm_cycle_days": random.choice([30, 60, 90]),
                "parameters": config["params"]
            }
            equipment_list.append(equipment)

    return equipment_list


def generate_lots_and_wafers(equipment_list: List[Dict], num_lots: int = 150) -> Tuple[List[Dict], List[Dict]]:
    """Lot 諛?Wafer ?곗???"""
    lots = []
    wafers = []

    for i in range(num_lots):
        lot_date = START_TIME + timedelta(days=random.randint(0, 25))
        lot_id = f"LOT{lot_date.strftime('%Y%m%d')}{i+1:04d}"

        # ? ? (媛以移 湲곕)
        product = random.choices(PRODUCTS, weights=[p["priority_weight"] for p in PRODUCTS])[0]

        # ?곗?
        priority_rand = random.random()
        if priority_rand < 0.6:
            priority = "NORMAL"
        elif priority_rand < 0.85:
            priority = "HOT"
        else:
            priority = "SUPER_HOT"

        # ? 諛?? 怨듭
        status_rand = random.random()
        if status_rand < 0.5:
            status = "RUN"
            current_step_idx = random.randint(0, len(PROCESS_FLOW) - 1)
        elif status_rand < 0.75:
            status = "WAIT"
            current_step_idx = random.randint(0, len(PROCESS_FLOW) - 1)
        elif status_rand < 0.9:
            status = "COMPLETED"
            current_step_idx = len(PROCESS_FLOW) - 1
        else:
            status = "HOLD"
            current_step_idx = random.randint(0, len(PROCESS_FLOW) - 1)

        wafer_count = random.choice([25, 24, 23, 22])

        lot = {
            "lot_id": lot_id,
            "product_code": product["code"],
            "product_name": product["name"],
            "quantity": wafer_count,
            "priority": priority,
            "status": status,
            "current_step": PROCESS_FLOW[current_step_idx]["id"],
            "fab_id": "FAB1",
            "route_id": "ROUTE-DRAM-001",
            "start_time": lot_date.isoformat() + "Z",
            "end_time": (lot_date + timedelta(days=random.randint(3, 7))).isoformat() + "Z" if status == "COMPLETED" else None
        }
        lots.append(lot)

        # ?⑥???
        for slot in range(1, wafer_count + 1):
            wafer_status_rand = random.random()
            if wafer_status_rand < 0.92:
                wafer_status = "GOOD"
            elif wafer_status_rand < 0.98:
                wafer_status = "REJECT"
            else:
                wafer_status = "SCRAP"

            wafer = {
                "wafer_id": f"{lot_id}-{slot:02d}",
                "lot_id": lot_id,
                "slot_no": slot,
                "status": wafer_status,
                "wafer_size": "300mm"
            }
            wafers.append(wafer)

    return lots, wafers


def generate_fdc_data(equipment_list: List[Dict], days: int = 30) -> List[Dict]:
    """FDC 측정 데이터 생성 (시계열 패턴 포함)"""
    measurements = []

    # 파라미터별 기준값
    param_specs = {
        "RF_Power": {"mean": 500, "std": 10, "usl": 530, "lsl": 470},
        "Chamber_Pressure": {"mean": 50, "std": 2, "usl": 56, "lsl": 44},
        "Gas_Flow_CF4": {"mean": 100, "std": 3, "usl": 109, "lsl": 91},
        "Gas_Flow_O2": {"mean": 50, "std": 2, "usl": 56, "lsl": 44},
        "Chuck_Temp": {"mean": 60, "std": 1, "usl": 63, "lsl": 57},
        "ESC_Voltage": {"mean": 300, "std": 5, "usl": 315, "lsl": 285},
        "Heater_Temp": {"mean": 400, "std": 5, "usl": 415, "lsl": 385},
        "Gas_Flow_SiH4": {"mean": 200, "std": 5, "usl": 215, "lsl": 185},
        "Gas_Flow_N2O": {"mean": 500, "std": 10, "usl": 530, "lsl": 470},
        "Thickness": {"mean": 1000, "std": 20, "usl": 1060, "lsl": 940},
        "DC_Power": {"mean": 1000, "std": 20, "usl": 1060, "lsl": 940},
        "Ar_Flow": {"mean": 30, "std": 1, "usl": 33, "lsl": 27},
        "Target_Voltage": {"mean": 400, "std": 10, "usl": 430, "lsl": 370},
        "Substrate_Temp": {"mean": 200, "std": 5, "usl": 215, "lsl": 185},
        "Focus": {"mean": 0, "std": 0.02, "usl": 0.06, "lsl": -0.06},
        "Dose": {"mean": 30, "std": 0.5, "usl": 31.5, "lsl": 28.5},
        "Overlay_X": {"mean": 0, "std": 2, "usl": 6, "lsl": -6},
        "Overlay_Y": {"mean": 0, "std": 2, "usl": 6, "lsl": -6},
        "CD": {"mean": 45, "std": 1, "usl": 48, "lsl": 42},
        "Stage_Temp": {"mean": 22, "std": 0.1, "usl": 22.3, "lsl": 21.7},
        "Down_Force": {"mean": 3, "std": 0.1, "usl": 3.3, "lsl": 2.7},
        "Platen_Speed": {"mean": 100, "std": 2, "usl": 106, "lsl": 94},
        "Head_Speed": {"mean": 100, "std": 2, "usl": 106, "lsl": 94},
        "Slurry_Flow": {"mean": 200, "std": 5, "usl": 215, "lsl": 185},
        "Pad_Temp": {"mean": 30, "std": 1, "usl": 33, "lsl": 27},
        "Removal_Rate": {"mean": 3000, "std": 100, "usl": 3300, "lsl": 2700},
        "Zone1_Temp": {"mean": 900, "std": 2, "usl": 906, "lsl": 894},
        "Zone2_Temp": {"mean": 900, "std": 2, "usl": 906, "lsl": 894},
        "Zone3_Temp": {"mean": 900, "std": 2, "usl": 906, "lsl": 894},
        "Gas_Flow_N2": {"mean": 1000, "std": 20, "usl": 1060, "lsl": 940},
        "Pressure": {"mean": 760, "std": 5, "usl": 775, "lsl": 745},
        "Beam_Current": {"mean": 10, "std": 0.2, "usl": 10.6, "lsl": 9.4},
        "Beam_Energy": {"mean": 50, "std": 1, "usl": 53, "lsl": 47},
        "Tilt_Angle": {"mean": 7, "std": 0.1, "usl": 7.3, "lsl": 6.7},
        "Twist_Angle": {"mean": 0, "std": 0.5, "usl": 1.5, "lsl": -1.5},
        "Wafer_Temp": {"mean": 25, "std": 1, "usl": 28, "lsl": 22},
        "Measurement_Value": {"mean": 100, "std": 2, "usl": 106, "lsl": 94},
        "Repeatability": {"mean": 0.5, "std": 0.1, "usl": 0.8, "lsl": 0.2},
        "TIS": {"mean": 0, "std": 0.5, "usl": 1.5, "lsl": -1.5},
    }

    # ?留?(紐⑤ ?ㅻ ????쇰?留?
    sampled_equipment = random.sample(equipment_list, min(20, len(equipment_list)))

    for equipment in sampled_equipment:
        eq_type = equipment["type"]
        if eq_type not in EQUIPMENT_TYPES:
            continue

        params = EQUIPMENT_TYPES[eq_type]["params"]

        for param in params:
            if param not in param_specs:
                continue

            spec = param_specs[param]

            # ???ㅻ-?쇰명 議고???뱀 寃곗
            # ?(80%), ?由?(10%), ?댁(5%), 二쇨린??5%)
            pattern_rand = random.random()
            if pattern_rand < 0.80:
                pattern = "normal"
            elif pattern_rand < 0.90:
                pattern = "drift"
            elif pattern_rand < 0.95:
                pattern = "anomaly"
            else:
                pattern = "periodic"

            # 1遺?媛寃?쇰 ?곗??? (?猷??1440媛?
            # ?깅??? 10遺?媛寃?쇰 以
            interval_minutes = 10
            points_per_day = 24 * 60 // interval_minutes

            current_time = START_TIME
            drift_offset = 0

            for day in range(days):
                for point in range(points_per_day):
                    timestamp = current_time + timedelta(minutes=point * interval_minutes)

                    # 湲곕낯 媛??
                    value = random.gauss(spec["mean"], spec["std"])

                    # ?⑦ ?
                    if pattern == "drift":
                        # 泥泥??利?? ?由?
                        drift_offset += random.gauss(0.001, 0.0005) * spec["std"]
                        value += drift_offset
                    elif pattern == "anomaly" and day >= days - 5:
                        # 留?留?5?쇱 ?댁 諛
                        if random.random() < 0.1:
                            value = spec["mean"] + random.choice([-1, 1]) * random.uniform(3, 5) * spec["std"]
                    elif pattern == "periodic":
                        # 24?媛 二쇨린 ?⑦
                        hour_of_day = point * interval_minutes / 60
                        value += spec["std"] * math.sin(2 * math.pi * hour_of_day / 24)

                    # ? 寃곗
                    if value > spec["usl"] or value < spec["lsl"]:
                        status = "OOC"
                    elif value > spec["mean"] + 2 * spec["std"] or value < spec["mean"] - 2 * spec["std"]:
                        status = "WARNING"
                    else:
                        status = "NORMAL"

                    measurements.append({
                        "equipment_id": equipment["equipment_id"],
                        "param_id": param,
                        "timestamp": timestamp.isoformat(),
                        "value": round(value, 4),
                        "status": status,
                        "usl": spec["usl"],
                        "lsl": spec["lsl"],
                        "target": spec["mean"]
                    })

                current_time += timedelta(days=1)

    return measurements


def generate_alarms(equipment_list: List[Dict], lots: List[Dict], num_alarms: int = 500) -> List[Dict]:
    """? ?곗???"""
    alarms = []

    for i in range(num_alarms):
        # ?媛 ? ?
        alarm_time = START_TIME + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        # ? ? ?
        source = random.choice(list(ALARM_CODES.keys()))
        alarm_def = random.choice(ALARM_CODES[source])

        equipment = random.choice(equipment_list)

        # Lot ?곌껐 (50% ?瑜)
        lot = random.choice(lots) if random.random() < 0.5 else None

        # ? ?
        status_rand = random.random()
        if status_rand < 0.3:
            status = "ACTIVE"
            resolved_at = None
        elif status_rand < 0.7:
            status = "ACKNOWLEDGED"
            resolved_at = None
        else:
            status = "RESOLVED"
            resolved_at = (alarm_time + timedelta(minutes=random.randint(5, 120))).isoformat() + "Z"

        alarm = {
            "alarm_id": f"ALM-{alarm_time.strftime('%Y%m%d')}-{i+1:05d}",
            "alarm_code": alarm_def["code"],
            "alarm_name": alarm_def["name"],
            "source_system": source,
            "severity": alarm_def["severity"],
            "category": "OOC" if source == "SPC" else "FAULT",
            "equipment_id": equipment["equipment_id"],
            "lot_id": lot["lot_id"] if lot else None,
            "message": f"{alarm_def['name']} on {equipment['equipment_id']}",
            "triggered_value": round(random.uniform(90, 110), 2),
            "threshold_value": 100.0,
            "status": status,
            "occurred_at": alarm_time.isoformat() + "Z",
            "resolved_at": resolved_at
        }
        alarms.append(alarm)

    return alarms


def generate_process_history(lots: List[Dict], equipment_list: List[Dict]) -> List[Dict]:
    """怨듭 ?대 ?곗???"""
    history = []

    # 설비 타입별 그룹화
    equipment_by_type = {}
    for eq in equipment_list:
        eq_type = eq["type"]
        if eq_type not in equipment_by_type:
            equipment_by_type[eq_type] = []
        equipment_by_type[eq_type].append(eq)

    for lot in lots:
        if lot["status"] == "COMPLETED":
            steps_completed = len(PROCESS_FLOW)
        else:
            current_idx = next((i for i, p in enumerate(PROCESS_FLOW) if p["id"] == lot["current_step"]), 0)
            steps_completed = current_idx + 1

        process_time = datetime.fromisoformat(lot["start_time"].replace("Z", ""))

        for i in range(steps_completed):
            process = PROCESS_FLOW[i]
            eq_type = process["equipment_type"]

            # ?대 ?? ?ㅻ ?
            if eq_type in equipment_by_type and equipment_by_type[eq_type]:
                equipment = random.choice(equipment_by_type[eq_type])
            else:
                continue

            # 怨듭 ?媛 怨
            actual_ct = process["target_ct"] * random.uniform(0.8, 1.2)
            end_time = process_time + timedelta(minutes=actual_ct)

            history.append({
                "lot_id": lot["lot_id"],
                "equipment_id": equipment["equipment_id"],
                "process_id": process["id"],
                "recipe_id": f"RCP-{process['id']}-V1.0",
                "start_time": process_time.isoformat() + "Z",
                "end_time": end_time.isoformat() + "Z",
                "status": "COMPLETED" if i < steps_completed - 1 else ("IN_PROGRESS" if lot["status"] == "RUN" else "COMPLETED"),
                "cycle_time": round(actual_ct, 1)
            })

            # ?湲??媛 異?
            process_time = end_time + timedelta(minutes=random.randint(5, 60))

    return history


# ============================================================
# ?곗?곕?댁 濡 ?⑥
# ============================================================

async def load_to_graph_db(equipment: List[Dict], lots: List[Dict], wafers: List[Dict],
                           alarms: List[Dict], history: List[Dict]):
    """PostgreSQL + AGE 洹몃??DB???곗??濡"""

    conn = await asyncpg.connect(**POSTGRES_CONFIG)

    try:
        # AGE ?ㅼ
        await conn.execute("LOAD 'age'")
        await conn.execute("SET search_path = ag_catalog, '$user', public")

        # 湲곗〈 ?곗????
        print("Clearing existing graph data...")
        await conn.execute("""
            SELECT * FROM cypher('manufacturing', $$
                MATCH (n) DETACH DELETE n
            $$) as (result agtype)
        """)

        # Equipment ?
        print(f"Creating {len(equipment)} equipment nodes...")
        for eq in equipment:
            props = ", ".join([f"{k}: '{v}'" if isinstance(v, str) else f"{k}: {v}"
                              for k, v in eq.items() if k != "parameters"])
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    CREATE (:Equipment {{{props}}})
                $$) as (v agtype)
            """)

        # Process ?
        print(f"Creating {len(PROCESS_FLOW)} process nodes...")
        for proc in PROCESS_FLOW:
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    CREATE (:Process {{
                        process_id: '{proc["id"]}',
                        name: '{proc["name"]}',
                        category: '{proc["category"]}',
                        equipment_type: '{proc["equipment_type"]}',
                        target_cycle_time: {proc["target_ct"]}
                    }})
                $$) as (v agtype)
            """)

        # Process ? 愿怨?        print("Creating process flow relationships...")
        for i in range(len(PROCESS_FLOW) - 1):
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    MATCH (p1:Process {{process_id: '{PROCESS_FLOW[i]["id"]}'}}),
                          (p2:Process {{process_id: '{PROCESS_FLOW[i+1]["id"]}'}})
                    CREATE (p1)-[:NEXT_STEP {{sequence: {i+1}}}]->(p2)
                $$) as (r agtype)
            """)

        # Lot ?
        print(f"Creating {len(lots)} lot nodes...")
        for lot in lots:
            end_time_prop = f", end_time: '{lot['end_time']}'" if lot['end_time'] else ""
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    CREATE (:Lot {{
                        lot_id: '{lot["lot_id"]}',
                        product_code: '{lot["product_code"]}',
                        product_name: '{lot["product_name"]}',
                        quantity: {lot["quantity"]},
                        priority: '{lot["priority"]}',
                        status: '{lot["status"]}',
                        current_step: '{lot["current_step"]}',
                        fab_id: '{lot["fab_id"]}',
                        route_id: '{lot["route_id"]}',
                        start_time: '{lot["start_time"]}'{end_time_prop}
                    }})
                $$) as (v agtype)
            """)

        # Wafer ? (?留?
        sampled_wafers = random.sample(wafers, min(500, len(wafers)))
        print(f"Creating {len(sampled_wafers)} wafer nodes (sampled)...")
        for wafer in sampled_wafers:
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    CREATE (:Wafer {{
                        wafer_id: '{wafer["wafer_id"]}',
                        lot_id: '{wafer["lot_id"]}',
                        slot_no: {wafer["slot_no"]},
                        status: '{wafer["status"]}',
                        wafer_size: '{wafer["wafer_size"]}'
                    }})
                $$) as (v agtype)
            """)

        # Wafer-Lot 愿怨?        print("Creating wafer-lot relationships...")
        for wafer in sampled_wafers:
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    MATCH (w:Wafer {{wafer_id: '{wafer["wafer_id"]}'}}),
                          (l:Lot {{lot_id: '{wafer["lot_id"]}'}})
                    CREATE (w)-[:BELONGS_TO {{slot_no: {wafer["slot_no"]}}}]->(l)
                $$) as (r agtype)
            """)

        # Alarm ?
        sampled_alarms = random.sample(alarms, min(200, len(alarms)))
        print(f"Creating {len(sampled_alarms)} alarm nodes (sampled)...")
        for alarm in sampled_alarms:
            lot_prop = f", lot_id: '{alarm['lot_id']}'" if alarm['lot_id'] else ""
            resolved_prop = f", resolved_at: '{alarm['resolved_at']}'" if alarm['resolved_at'] else ""
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    CREATE (:Alarm {{
                        alarm_id: '{alarm["alarm_id"]}',
                        alarm_code: '{alarm["alarm_code"]}',
                        alarm_name: '{alarm["alarm_name"]}',
                        source_system: '{alarm["source_system"]}',
                        severity: '{alarm["severity"]}',
                        category: '{alarm["category"]}',
                        equipment_id: '{alarm["equipment_id"]}',
                        message: '{alarm["message"].replace("'", "''")}',
                        triggered_value: {alarm["triggered_value"]},
                        threshold_value: {alarm["threshold_value"]},
                        status: '{alarm["status"]}',
                        occurred_at: '{alarm["occurred_at"]}'{lot_prop}{resolved_prop}
                    }})
                $$) as (v agtype)
            """)

        # Equipment-Alarm 愿怨?        print("Creating equipment-alarm relationships...")
        for alarm in sampled_alarms:
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    MATCH (e:Equipment {{equipment_id: '{alarm["equipment_id"]}'}}),
                          (a:Alarm {{alarm_id: '{alarm["alarm_id"]}'}})
                    CREATE (e)-[:GENERATES_ALARM]->(a)
                $$) as (r agtype)
            """)

        # 怨듭 ?대 愿怨?(PROCESSED_AT)
        sampled_history = random.sample(history, min(500, len(history)))
        print(f"Creating {len(sampled_history)} process history relationships (sampled)...")
        for h in sampled_history:
            await conn.execute(f"""
                SELECT * FROM cypher('manufacturing', $$
                    MATCH (l:Lot {{lot_id: '{h["lot_id"]}'}}),
                          (e:Equipment {{equipment_id: '{h["equipment_id"]}'}})
                    CREATE (l)-[:PROCESSED_AT {{
                        process_id: '{h["process_id"]}',
                        recipe_id: '{h["recipe_id"]}',
                        start_time: '{h["start_time"]}',
                        end_time: '{h["end_time"]}',
                        status: '{h["status"]}',
                        cycle_time: {h["cycle_time"]}
                    }}]->(e)
                $$) as (r agtype)
            """)

        print("Graph data loaded successfully!")

    finally:
        await conn.close()


async def load_to_timescale(measurements: List[Dict]):
    """TimescaleDB???怨???곗??濡"""

    conn = await asyncpg.connect(**TIMESCALE_CONFIG)

    try:
        # ?대??
        print("Creating TimescaleDB tables...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS fdc_measurements (
                time TIMESTAMPTZ NOT NULL,
                equipment_id TEXT NOT NULL,
                param_id TEXT NOT NULL,
                value DOUBLE PRECISION,
                status TEXT,
                usl DOUBLE PRECISION,
                lsl DOUBLE PRECISION,
                target DOUBLE PRECISION
            )
        """)

        # Hypertable 蹂??(?대? 議댁?硫 臾댁)
        try:
            await conn.execute("""
                SELECT create_hypertable('fdc_measurements', 'time', if_not_exists => TRUE)
            """)
        except:
            pass

        # 湲곗〈 ?곗????
        await conn.execute("TRUNCATE fdc_measurements")

        # 諛곗 ?쎌
        print(f"Inserting {len(measurements)} FDC measurements...")
        batch_size = 1000
        for i in range(0, len(measurements), batch_size):
            batch = measurements[i:i + batch_size]
            values = [
                (
                    datetime.fromisoformat(m["timestamp"]),
                    f'FDC-{m["equipment_id"]}-{m["param_id"]}-{i+j}',  # measurement_id
                    m["equipment_id"],
                    m["param_id"],
                    m["value"],
                    m["usl"],
                    m["lsl"],
                    m.get("target", (m["usl"] + m["lsl"]) / 2),
                    m["status"]
                )
                for j, m in enumerate(batch)
            ]
            await conn.executemany("""
                INSERT INTO fdc_measurements (time, measurement_id, equipment_id, param_id, value, usl, lsl, target, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, values)

            if (i + batch_size) % 10000 == 0:
                print(f"  Inserted {min(i + batch_size, len(measurements))} / {len(measurements)}")

        # ?몃???
        print("Creating indexes...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fdc_equipment_time
            ON fdc_measurements (equipment_id, time DESC)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fdc_param_time
            ON fdc_measurements (param_id, time DESC)
        """)

        print("TimescaleDB data loaded successfully!")

    finally:
        await conn.close()


# ============================================================
# 硫
# ============================================================

async def main():
    print("=" * 60)
    print("  반도체 FAB 합성 데이터 생성기")
    print("=" * 60)
    print()

    # 1. 데이터 생성
    print("[1/6] Generating equipment data...")
    equipment = generate_equipment()
    print(f"  Created {len(equipment)} equipment")

    print("[2/6] Generating lots and wafers...")
    lots, wafers = generate_lots_and_wafers(equipment, num_lots=150)
    print(f"  Created {len(lots)} lots, {len(wafers)} wafers")

    print("[3/6] Generating FDC measurements...")
    measurements = generate_fdc_data(equipment, days=30)
    print(f"  Created {len(measurements)} measurements")

    print("[4/6] Generating alarms...")
    alarms = generate_alarms(equipment, lots, num_alarms=500)
    print(f"  Created {len(alarms)} alarms")

    print("[5/6] Generating process history...")
    history = generate_process_history(lots, equipment)
    print(f"  Created {len(history)} process history records")

    # 2. 데이터베이스 로드
    print()
    print("[6/6] Loading data to databases...")

    print("\n--- Loading to Graph DB (PostgreSQL + AGE) ---")
    await load_to_graph_db(equipment, lots, wafers, alarms, history)

    print("\n--- Loading to TimescaleDB ---")
    await load_to_timescale(measurements)

    print()
    print("=" * 60)
    print("  데이터 생성 완료!")
    print("=" * 60)
    print()
    print("생성된 데이터:")
    print(f"  - Equipment: {len(equipment)}개")
    print(f"  - Process: {len(PROCESS_FLOW)}개")
    print(f"  - Lot: {len(lots)}개")
    print(f"  - Wafer: {len(wafers)}개")
    print(f"  - Alarm: {len(alarms)}개")
    print(f"  - FDC Measurements: {len(measurements)}개")
    print(f"  - Process History: {len(history)}개")


if __name__ == "__main__":
    asyncio.run(main())
