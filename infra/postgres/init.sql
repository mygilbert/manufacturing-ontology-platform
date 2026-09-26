-- ============================================================
-- Manufacturing Ontology Platform
-- PostgreSQL + Apache AGE 초기화 스크립트
-- ============================================================

-- Apache AGE 확장 설치
CREATE EXTENSION IF NOT EXISTS age;

-- AGE 로드 (세션별로 필요)
LOAD 'age';

-- ag_catalog 스키마를 search_path에 추가
SET search_path = ag_catalog, "$user", public;

-- ============================================================
-- 1. 그래프 생성
-- ============================================================

-- manufacturing 그래프 생성 (이미 존재하면 건너뜀)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'manufacturing') THEN
        PERFORM create_graph('manufacturing');
    END IF;
END $$;

-- ============================================================
-- 2. Vertex Labels (Object Types) 생성
-- ============================================================

-- Equipment 레이블
SELECT create_vlabel('manufacturing', 'Equipment');

-- Process 레이블
SELECT create_vlabel('manufacturing', 'Process');

-- Lot 레이블
SELECT create_vlabel('manufacturing', 'Lot');

-- Wafer 레이블
SELECT create_vlabel('manufacturing', 'Wafer');

-- Recipe 레이블
SELECT create_vlabel('manufacturing', 'Recipe');

-- Measurement 레이블
SELECT create_vlabel('manufacturing', 'Measurement');

-- Alarm 레이블
SELECT create_vlabel('manufacturing', 'Alarm');

-- ============================================================
-- 3. Edge Labels (Link Types) 생성
-- ============================================================

-- PROCESSED_AT: Lot → Equipment
SELECT create_elabel('manufacturing', 'PROCESSED_AT');

-- MEASURED_BY: Wafer → Measurement
SELECT create_elabel('manufacturing', 'MEASURED_BY');

-- BELONGS_TO: Wafer → Lot
SELECT create_elabel('manufacturing', 'BELONGS_TO');

-- GENERATES_ALARM: Equipment → Alarm
SELECT create_elabel('manufacturing', 'GENERATES_ALARM');

-- USES_RECIPE: Process/Equipment → Recipe
SELECT create_elabel('manufacturing', 'USES_RECIPE');

-- FOLLOWS_ROUTE: Lot → Process
SELECT create_elabel('manufacturing', 'FOLLOWS_ROUTE');

-- NEXT_STEP: Process → Process
SELECT create_elabel('manufacturing', 'NEXT_STEP');

-- AFFECTS_LOT: Alarm → Lot
SELECT create_elabel('manufacturing', 'AFFECTS_LOT');

-- CONTAINS_WAFER: Lot → Wafer
SELECT create_elabel('manufacturing', 'CONTAINS_WAFER');

-- ============================================================
-- 4. 인덱스 생성 (AGE는 property 인덱스를 CREATE INDEX로 생성)
-- ============================================================

-- Equipment 인덱스
CREATE INDEX IF NOT EXISTS idx_equipment_id
ON manufacturing."Equipment" USING btree ((properties->>'equipment_id'));

CREATE INDEX IF NOT EXISTS idx_equipment_status
ON manufacturing."Equipment" USING btree ((properties->>'status'));

CREATE INDEX IF NOT EXISTS idx_equipment_type
ON manufacturing."Equipment" USING btree ((properties->>'type'));

-- Lot 인덱스
CREATE INDEX IF NOT EXISTS idx_lot_id
ON manufacturing."Lot" USING btree ((properties->>'lot_id'));

CREATE INDEX IF NOT EXISTS idx_lot_status
ON manufacturing."Lot" USING btree ((properties->>'status'));

CREATE INDEX IF NOT EXISTS idx_lot_product_code
ON manufacturing."Lot" USING btree ((properties->>'product_code'));

-- Wafer 인덱스
CREATE INDEX IF NOT EXISTS idx_wafer_id
ON manufacturing."Wafer" USING btree ((properties->>'wafer_id'));

CREATE INDEX IF NOT EXISTS idx_wafer_lot_id
ON manufacturing."Wafer" USING btree ((properties->>'lot_id'));

-- Process 인덱스
CREATE INDEX IF NOT EXISTS idx_process_id
ON manufacturing."Process" USING btree ((properties->>'process_id'));

CREATE INDEX IF NOT EXISTS idx_process_route_id
ON manufacturing."Process" USING btree ((properties->>'route_id'));

-- Recipe 인덱스
CREATE INDEX IF NOT EXISTS idx_recipe_id
ON manufacturing."Recipe" USING btree ((properties->>'recipe_id'));

-- Alarm 인덱스
CREATE INDEX IF NOT EXISTS idx_alarm_id
ON manufacturing."Alarm" USING btree ((properties->>'alarm_id'));

CREATE INDEX IF NOT EXISTS idx_alarm_equipment_id
ON manufacturing."Alarm" USING btree ((properties->>'equipment_id'));

CREATE INDEX IF NOT EXISTS idx_alarm_severity
ON manufacturing."Alarm" USING btree ((properties->>'severity'));

-- Measurement 인덱스
CREATE INDEX IF NOT EXISTS idx_measurement_id
ON manufacturing."Measurement" USING btree ((properties->>'measurement_id'));

CREATE INDEX IF NOT EXISTS idx_measurement_equipment_id
ON manufacturing."Measurement" USING btree ((properties->>'equipment_id'));

CREATE INDEX IF NOT EXISTS idx_measurement_timestamp
ON manufacturing."Measurement" USING btree ((properties->>'timestamp'));

-- ============================================================
-- 5. TimescaleDB 설정 (측정 데이터용 하이퍼테이블)
-- ============================================================

-- 측정 데이터를 위한 관계형 테이블 (시계열 최적화)
CREATE TABLE IF NOT EXISTS measurements_timeseries (
    time                TIMESTAMPTZ NOT NULL,
    measurement_id      TEXT NOT NULL,
    equipment_id        TEXT NOT NULL,
    wafer_id            TEXT,
    lot_id              TEXT,
    process_id          TEXT,
    param_id            TEXT NOT NULL,
    param_name          TEXT,
    value               DOUBLE PRECISION NOT NULL,
    unit                TEXT,
    usl                 DOUBLE PRECISION,
    lsl                 DOUBLE PRECISION,
    ucl                 DOUBLE PRECISION,
    lcl                 DOUBLE PRECISION,
    target              DOUBLE PRECISION,
    status              TEXT DEFAULT 'NORMAL',
    source_system       TEXT,

    PRIMARY KEY (time, measurement_id)
);

-- TimescaleDB 하이퍼테이블로 변환 (TimescaleDB 설치 시)
-- SELECT create_hypertable('measurements_timeseries', 'time', if_not_exists => TRUE);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_meas_ts_equipment
ON measurements_timeseries (equipment_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_meas_ts_wafer
ON measurements_timeseries (wafer_id, time DESC) WHERE wafer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_meas_ts_param
ON measurements_timeseries (param_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_meas_ts_status
ON measurements_timeseries (status, time DESC) WHERE status != 'NORMAL';

-- ============================================================
-- 6. 알람 히스토리 테이블 (관계형)
-- ============================================================

CREATE TABLE IF NOT EXISTS alarm_history (
    time                TIMESTAMPTZ NOT NULL,
    alarm_id            TEXT NOT NULL,
    alarm_code          TEXT NOT NULL,
    alarm_name          TEXT,
    severity            TEXT NOT NULL,
    category            TEXT,
    source_system       TEXT,
    equipment_id        TEXT NOT NULL,
    chamber_id          TEXT,
    process_id          TEXT,
    lot_id              TEXT,
    wafer_id            TEXT,
    message             TEXT,
    triggered_value     DOUBLE PRECISION,
    threshold_value     DOUBLE PRECISION,
    status              TEXT DEFAULT 'ACTIVE',
    acknowledged_at     TIMESTAMPTZ,
    resolved_at         TIMESTAMPTZ,
    action_taken        TEXT,

    PRIMARY KEY (time, alarm_id)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_alarm_hist_equipment
ON alarm_history (equipment_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_alarm_hist_severity
ON alarm_history (severity, status, time DESC);

-- ============================================================
-- 7. 유틸리티 함수
-- ============================================================

-- Cypher 쿼리 실행을 위한 헬퍼 함수
CREATE OR REPLACE FUNCTION cypher_query(query text)
RETURNS SETOF agtype
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY EXECUTE format('SELECT * FROM cypher(''manufacturing'', $$ %s $$) AS (result agtype)', query);
END;
$$;

-- Object 생성 헬퍼 함수
CREATE OR REPLACE FUNCTION create_object(label text, properties jsonb)
RETURNS agtype
LANGUAGE plpgsql
AS $$
DECLARE
    result agtype;
    props_str text;
BEGIN
    props_str := properties::text;
    EXECUTE format(
        'SELECT * FROM cypher(''manufacturing'', $$ CREATE (n:%I %s) RETURN n $$) AS (n agtype)',
        label, props_str
    ) INTO result;
    RETURN result;
END;
$$;

-- Link 생성 헬퍼 함수
CREATE OR REPLACE FUNCTION create_link(
    from_label text, from_key text, from_value text,
    link_type text, link_props jsonb,
    to_label text, to_key text, to_value text
)
RETURNS agtype
LANGUAGE plpgsql
AS $$
DECLARE
    result agtype;
    props_str text;
BEGIN
    props_str := COALESCE(link_props::text, '{}');
    EXECUTE format(
        'SELECT * FROM cypher(''manufacturing'', $$
            MATCH (a:%I {%I: %L}), (b:%I {%I: %L})
            CREATE (a)-[r:%I %s]->(b)
            RETURN r
        $$) AS (r agtype)',
        from_label, from_key, from_value,
        to_label, to_key, to_value,
        link_type, props_str
    ) INTO result;
    RETURN result;
END;
$$;

-- ============================================================
-- 8. 뷰 생성 (온톨로지 조회용)
-- ============================================================

-- Equipment 목록 뷰
CREATE OR REPLACE VIEW v_equipment AS
SELECT
    (properties->>'equipment_id')::text AS equipment_id,
    (properties->>'name')::text AS name,
    (properties->>'type')::text AS type,
    (properties->>'status')::text AS status,
    (properties->>'location')::text AS location,
    properties
FROM manufacturing."Equipment";

-- Lot 목록 뷰
CREATE OR REPLACE VIEW v_lot AS
SELECT
    (properties->>'lot_id')::text AS lot_id,
    (properties->>'product_code')::text AS product_code,
    (properties->>'quantity')::int AS quantity,
    (properties->>'status')::text AS status,
    (properties->>'current_step')::text AS current_step,
    properties
FROM manufacturing."Lot";

-- Alarm 목록 뷰 (최근 알람)
CREATE OR REPLACE VIEW v_recent_alarms AS
SELECT
    (properties->>'alarm_id')::text AS alarm_id,
    (properties->>'alarm_code')::text AS alarm_code,
    (properties->>'severity')::text AS severity,
    (properties->>'status')::text AS status,
    (properties->>'equipment_id')::text AS equipment_id,
    (properties->>'message')::text AS message,
    (properties->>'occurred_at')::timestamptz AS occurred_at,
    properties
FROM manufacturing."Alarm"
ORDER BY (properties->>'occurred_at')::timestamptz DESC
LIMIT 100;

-- ============================================================
-- Action 감사 기록 (Kinetic Layer)
-- ============================================================
-- 온톨로지의 모든 상태 변경은 Action을 거치고, 여기에 기록된다.
-- 거부된 시도(권한/검증 실패)도 남긴다. "누가 무엇을 하려다 막혔는가"가
-- "누가 무엇을 했는가"만큼 중요하기 때문이다.

CREATE TABLE IF NOT EXISTS action_audit (
    audit_id          TEXT PRIMARY KEY,
    action_type       TEXT        NOT NULL,
    principal_id      TEXT        NOT NULL,
    principal_roles   TEXT[]      NOT NULL DEFAULT '{}',
    parameters        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    reason            TEXT,
    dry_run           BOOLEAN     NOT NULL DEFAULT FALSE,
    succeeded         BOOLEAN     NOT NULL,
    changes           JSONB       NOT NULL DEFAULT '[]'::jsonb,
    error             TEXT,
    occurred_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    on_behalf_of      TEXT,
    is_agent          BOOLEAN     NOT NULL DEFAULT FALSE,
    client_request_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_action_audit_occurred_at
    ON action_audit (occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_audit_action_type
    ON action_audit (action_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_audit_principal
    ON action_audit (principal_id, occurred_at DESC);
-- 실패/거부 이력만 빠르게 보기 위한 부분 인덱스
CREATE INDEX IF NOT EXISTS idx_action_audit_failures
    ON action_audit (occurred_at DESC) WHERE succeeded = FALSE;
-- 특정 객체의 변경 이력 추적용
CREATE INDEX IF NOT EXISTS idx_action_audit_changes
    ON action_audit USING GIN (changes);

COMMENT ON TABLE action_audit IS
    'Action 계층 감사 기록. 상태 변경은 Action을 통해서만 일어난다.';

-- ============================================================
-- 완료 메시지
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE 'Manufacturing Ontology 스키마 초기화 완료';
    RAISE NOTICE '- Vertex Labels: Equipment, Process, Lot, Wafer, Recipe, Measurement, Alarm';
    RAISE NOTICE '- Action 감사 테이블: action_audit';
    RAISE NOTICE '- Edge Labels: PROCESSED_AT, MEASURED_BY, BELONGS_TO, GENERATES_ALARM, USES_RECIPE, FOLLOWS_ROUTE, NEXT_STEP, AFFECTS_LOT, CONTAINS_WAFER';
END $$;
