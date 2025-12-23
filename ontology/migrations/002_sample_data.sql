-- ============================================================
-- Manufacturing Ontology - 샘플 데이터
-- ============================================================

-- AGE 로드
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- ============================================================
-- 1. Equipment (설비) 생성
-- ============================================================

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Equipment {
    equipment_id: 'EQP-ETCH-001',
    name: 'Etcher-01',
    type: 'DRY_ETCH',
    status: 'RUNNING',
    location: 'FAB1-ZONE2',
    manufacturer: 'Applied Materials',
    model: 'Centura 5200',
    chamber_count: 2
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Equipment {
    equipment_id: 'EQP-ETCH-002',
    name: 'Etcher-02',
    type: 'DRY_ETCH',
    status: 'IDLE',
    location: 'FAB1-ZONE2',
    manufacturer: 'Applied Materials',
    model: 'Centura 5200',
    chamber_count: 2
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Equipment {
    equipment_id: 'EQP-CVD-001',
    name: 'CVD-01',
    type: 'CVD',
    status: 'RUNNING',
    location: 'FAB1-ZONE3',
    manufacturer: 'LAM Research',
    model: 'Vector Express',
    chamber_count: 4
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Equipment {
    equipment_id: 'EQP-LITHO-001',
    name: 'Litho-01',
    type: 'LITHO',
    status: 'RUNNING',
    location: 'FAB1-ZONE1',
    manufacturer: 'ASML',
    model: 'TWINSCAN NXE:3400C',
    chamber_count: 1
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Equipment {
    equipment_id: 'EQP-METRO-001',
    name: 'Metrology-01',
    type: 'METROLOGY',
    status: 'RUNNING',
    location: 'FAB1-ZONE4',
    manufacturer: 'KLA',
    model: 'SpectraFilm F1',
    chamber_count: 1
  })
$$) as (v agtype);

-- ============================================================
-- 2. Process (공정) 생성
-- ============================================================

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Process {
    process_id: 'PROC-LITHO-001',
    name: 'Photo Lithography',
    category: 'PHOTO',
    sequence: 1,
    route_id: 'ROUTE-DRAM-001',
    target_cycle_time: 45.0,
    is_critical: true,
    spc_enabled: true
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Process {
    process_id: 'PROC-ETCH-001',
    name: 'Oxide Etch',
    category: 'ETCH',
    sequence: 2,
    route_id: 'ROUTE-DRAM-001',
    target_cycle_time: 30.0,
    is_critical: true,
    spc_enabled: true
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Process {
    process_id: 'PROC-CVD-001',
    name: 'TEOS CVD',
    category: 'THIN_FILM',
    sequence: 3,
    route_id: 'ROUTE-DRAM-001',
    target_cycle_time: 60.0,
    is_critical: false,
    spc_enabled: true
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Process {
    process_id: 'PROC-METRO-001',
    name: 'Film Thickness Measurement',
    category: 'METROLOGY',
    sequence: 4,
    route_id: 'ROUTE-DRAM-001',
    target_cycle_time: 15.0,
    is_critical: false,
    spc_enabled: true
  })
$$) as (v agtype);

-- ============================================================
-- 3. Recipe (레시피) 생성
-- ============================================================

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Recipe {
    recipe_id: 'RCP-ETCH-OXIDE-V2.1',
    recipe_name: 'OXIDE_ETCH_STD',
    version: '2.1',
    status: 'ACTIVE',
    equipment_type: 'DRY_ETCH',
    process_type: 'OXIDE_ETCH',
    total_duration: 1800.0
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Recipe {
    recipe_id: 'RCP-CVD-TEOS-V1.5',
    recipe_name: 'TEOS_DEP_STD',
    version: '1.5',
    status: 'ACTIVE',
    equipment_type: 'CVD',
    process_type: 'TEOS_DEP',
    total_duration: 3600.0
  })
$$) as (v agtype);

-- ============================================================
-- 4. Lot 생성
-- ============================================================

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Lot {
    lot_id: 'LOT20231201001',
    product_code: 'DRAM-8Gb-DDR5',
    product_name: '8Gb DDR5 SDRAM',
    quantity: 25,
    priority: 'NORMAL',
    status: 'RUN',
    current_step: 'PROC-ETCH-001',
    fab_id: 'FAB1',
    route_id: 'ROUTE-DRAM-001',
    start_time: '2023-12-01T09:00:00Z'
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Lot {
    lot_id: 'LOT20231201002',
    product_code: 'DRAM-8Gb-DDR5',
    product_name: '8Gb DDR5 SDRAM',
    quantity: 25,
    priority: 'HOT',
    status: 'WAIT',
    current_step: 'PROC-LITHO-001',
    fab_id: 'FAB1',
    route_id: 'ROUTE-DRAM-001',
    start_time: '2023-12-01T10:00:00Z'
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Lot {
    lot_id: 'LOT20231201003',
    product_code: 'DRAM-16Gb-DDR5',
    product_name: '16Gb DDR5 SDRAM',
    quantity: 25,
    priority: 'SUPER_HOT',
    status: 'RUN',
    current_step: 'PROC-CVD-001',
    fab_id: 'FAB1',
    route_id: 'ROUTE-DRAM-001',
    start_time: '2023-12-01T08:00:00Z'
  })
$$) as (v agtype);

-- ============================================================
-- 5. Wafer 생성 (LOT20231201001의 웨이퍼들)
-- ============================================================

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Wafer {
    wafer_id: 'LOT20231201001-01',
    lot_id: 'LOT20231201001',
    slot_no: 1,
    status: 'GOOD',
    wafer_size: '300mm'
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Wafer {
    wafer_id: 'LOT20231201001-02',
    lot_id: 'LOT20231201001',
    slot_no: 2,
    status: 'GOOD',
    wafer_size: '300mm'
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Wafer {
    wafer_id: 'LOT20231201001-03',
    lot_id: 'LOT20231201001',
    slot_no: 3,
    status: 'IN_PROCESS',
    wafer_size: '300mm'
  })
$$) as (v agtype);

-- ============================================================
-- 6. Alarm 생성
-- ============================================================

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Alarm {
    alarm_id: 'ALM-20231201-001',
    alarm_code: 'FDC-PRESS-HIGH',
    alarm_name: 'Chamber Pressure High',
    source_system: 'FDC',
    severity: 'MAJOR',
    category: 'OOC',
    equipment_id: 'EQP-ETCH-001',
    lot_id: 'LOT20231201001',
    message: 'Chamber pressure exceeded UCL: 105.3 mTorr > 100 mTorr',
    triggered_value: 105.3,
    threshold_value: 100.0,
    status: 'ACTIVE',
    occurred_at: '2023-12-01T10:30:00Z'
  })
$$) as (v agtype);

SELECT * FROM cypher('manufacturing', $$
  CREATE (:Alarm {
    alarm_id: 'ALM-20231201-002',
    alarm_code: 'SPC-RULE1',
    alarm_name: 'SPC Rule 1 Violation',
    source_system: 'SPC',
    severity: 'WARNING',
    category: 'OOC',
    equipment_id: 'EQP-CVD-001',
    process_id: 'PROC-CVD-001',
    message: 'Film thickness out of control limit',
    triggered_value: 1520.5,
    threshold_value: 1500.0,
    status: 'ACKNOWLEDGED',
    occurred_at: '2023-12-01T09:45:00Z'
  })
$$) as (v agtype);

-- ============================================================
-- 7. 관계(Edge) 생성
-- ============================================================

-- Lot → Equipment (PROCESSED_AT)
SELECT * FROM cypher('manufacturing', $$
  MATCH (l:Lot {lot_id: 'LOT20231201001'}), (e:Equipment {equipment_id: 'EQP-ETCH-001'})
  CREATE (l)-[:PROCESSED_AT {
    process_id: 'PROC-ETCH-001',
    recipe_id: 'RCP-ETCH-OXIDE-V2.1',
    start_time: '2023-12-01T10:00:00Z',
    status: 'IN_PROGRESS'
  }]->(e)
$$) as (r agtype);

SELECT * FROM cypher('manufacturing', $$
  MATCH (l:Lot {lot_id: 'LOT20231201003'}), (e:Equipment {equipment_id: 'EQP-CVD-001'})
  CREATE (l)-[:PROCESSED_AT {
    process_id: 'PROC-CVD-001',
    recipe_id: 'RCP-CVD-TEOS-V1.5',
    start_time: '2023-12-01T09:00:00Z',
    status: 'IN_PROGRESS'
  }]->(e)
$$) as (r agtype);

-- Wafer → Lot (BELONGS_TO)
SELECT * FROM cypher('manufacturing', $$
  MATCH (w:Wafer {wafer_id: 'LOT20231201001-01'}), (l:Lot {lot_id: 'LOT20231201001'})
  CREATE (w)-[:BELONGS_TO {slot_no: 1}]->(l)
$$) as (r agtype);

SELECT * FROM cypher('manufacturing', $$
  MATCH (w:Wafer {wafer_id: 'LOT20231201001-02'}), (l:Lot {lot_id: 'LOT20231201001'})
  CREATE (w)-[:BELONGS_TO {slot_no: 2}]->(l)
$$) as (r agtype);

SELECT * FROM cypher('manufacturing', $$
  MATCH (w:Wafer {wafer_id: 'LOT20231201001-03'}), (l:Lot {lot_id: 'LOT20231201001'})
  CREATE (w)-[:BELONGS_TO {slot_no: 3}]->(l)
$$) as (r agtype);

-- Equipment → Alarm (GENERATES_ALARM)
SELECT * FROM cypher('manufacturing', $$
  MATCH (e:Equipment {equipment_id: 'EQP-ETCH-001'}), (a:Alarm {alarm_id: 'ALM-20231201-001'})
  CREATE (e)-[:GENERATES_ALARM {chamber_id: 'CH1'}]->(a)
$$) as (r agtype);

SELECT * FROM cypher('manufacturing', $$
  MATCH (e:Equipment {equipment_id: 'EQP-CVD-001'}), (a:Alarm {alarm_id: 'ALM-20231201-002'})
  CREATE (e)-[:GENERATES_ALARM {chamber_id: 'CH2'}]->(a)
$$) as (r agtype);

-- Process → Recipe (USES_RECIPE)
SELECT * FROM cypher('manufacturing', $$
  MATCH (p:Process {process_id: 'PROC-ETCH-001'}), (r:Recipe {recipe_id: 'RCP-ETCH-OXIDE-V2.1'})
  CREATE (p)-[:USES_RECIPE {is_default: true, is_active: true}]->(r)
$$) as (r agtype);

SELECT * FROM cypher('manufacturing', $$
  MATCH (p:Process {process_id: 'PROC-CVD-001'}), (r:Recipe {recipe_id: 'RCP-CVD-TEOS-V1.5'})
  CREATE (p)-[:USES_RECIPE {is_default: true, is_active: true}]->(r)
$$) as (r agtype);

-- Process → Process (NEXT_STEP) - 공정 순서
SELECT * FROM cypher('manufacturing', $$
  MATCH (p1:Process {process_id: 'PROC-LITHO-001'}), (p2:Process {process_id: 'PROC-ETCH-001'})
  CREATE (p1)-[:NEXT_STEP {route_id: 'ROUTE-DRAM-001', is_default: true}]->(p2)
$$) as (r agtype);

SELECT * FROM cypher('manufacturing', $$
  MATCH (p1:Process {process_id: 'PROC-ETCH-001'}), (p2:Process {process_id: 'PROC-CVD-001'})
  CREATE (p1)-[:NEXT_STEP {route_id: 'ROUTE-DRAM-001', is_default: true}]->(p2)
$$) as (r agtype);

SELECT * FROM cypher('manufacturing', $$
  MATCH (p1:Process {process_id: 'PROC-CVD-001'}), (p2:Process {process_id: 'PROC-METRO-001'})
  CREATE (p1)-[:NEXT_STEP {route_id: 'ROUTE-DRAM-001', is_default: true}]->(p2)
$$) as (r agtype);

-- Alarm → Lot (AFFECTS_LOT)
SELECT * FROM cypher('manufacturing', $$
  MATCH (a:Alarm {alarm_id: 'ALM-20231201-001'}), (l:Lot {lot_id: 'LOT20231201001'})
  CREATE (a)-[:AFFECTS_LOT {
    impact_level: 'MEDIUM',
    affected_wafer_count: 3,
    disposition: 'HOLD'
  }]->(l)
$$) as (r agtype);

-- ============================================================
-- 완료 메시지
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE '샘플 데이터 생성 완료';
    RAISE NOTICE '- Equipment: 5개';
    RAISE NOTICE '- Process: 4개';
    RAISE NOTICE '- Recipe: 2개';
    RAISE NOTICE '- Lot: 3개';
    RAISE NOTICE '- Wafer: 3개';
    RAISE NOTICE '- Alarm: 2개';
    RAISE NOTICE '- 관계: 12개';
END $$;
