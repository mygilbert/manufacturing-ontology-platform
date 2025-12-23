-- ============================================================
-- Manufacturing Ontology Platform
-- TimescaleDB 초기화 스크립트
-- ============================================================

-- TimescaleDB 확장 설치
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- 1. FDC 측정 데이터 테이블
-- ============================================================

CREATE TABLE IF NOT EXISTS fdc_measurements (
    time                TIMESTAMPTZ NOT NULL,
    measurement_id      TEXT NOT NULL,
    equipment_id        TEXT NOT NULL,
    chamber_id          TEXT,
    recipe_id           TEXT,
    lot_id              TEXT,
    wafer_id            TEXT,
    slot_no             INTEGER,
    param_id            TEXT NOT NULL,
    param_name          TEXT,
    value               DOUBLE PRECISION NOT NULL,
    unit                TEXT,
    usl                 DOUBLE PRECISION,
    lsl                 DOUBLE PRECISION,
    target              DOUBLE PRECISION,
    status              TEXT DEFAULT 'NORMAL',

    PRIMARY KEY (time, measurement_id)
);

-- 하이퍼테이블 생성 (7일 단위 청크)
SELECT create_hypertable('fdc_measurements', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 압축 정책 (30일 이상 된 데이터)
ALTER TABLE fdc_measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id, param_id'
);

SELECT add_compression_policy('fdc_measurements', INTERVAL '30 days', if_not_exists => TRUE);

-- 보관 정책 (1년 후 삭제)
SELECT add_retention_policy('fdc_measurements', INTERVAL '365 days', if_not_exists => TRUE);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_fdc_equipment_time
ON fdc_measurements (equipment_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_fdc_lot_time
ON fdc_measurements (lot_id, time DESC) WHERE lot_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_fdc_param_time
ON fdc_measurements (param_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_fdc_status_time
ON fdc_measurements (status, time DESC) WHERE status != 'NORMAL';

-- ============================================================
-- 2. SPC 측정 데이터 테이블
-- ============================================================

CREATE TABLE IF NOT EXISTS spc_measurements (
    time                TIMESTAMPTZ NOT NULL,
    measurement_id      TEXT NOT NULL,
    equipment_id        TEXT NOT NULL,
    process_id          TEXT NOT NULL,
    lot_id              TEXT,
    wafer_id            TEXT,
    item_id             TEXT NOT NULL,
    item_name           TEXT,
    value               DOUBLE PRECISION NOT NULL,
    unit                TEXT,
    usl                 DOUBLE PRECISION,
    lsl                 DOUBLE PRECISION,
    ucl                 DOUBLE PRECISION,
    lcl                 DOUBLE PRECISION,
    target              DOUBLE PRECISION,
    subgroup_id         TEXT,
    subgroup_size       INTEGER DEFAULT 1,
    -- SPC 통계
    x_bar               DOUBLE PRECISION,
    range_val           DOUBLE PRECISION,
    std_dev             DOUBLE PRECISION,
    cp                  DOUBLE PRECISION,
    cpk                 DOUBLE PRECISION,
    -- 판정
    status              TEXT DEFAULT 'NORMAL',
    rule_violations     TEXT[],

    PRIMARY KEY (time, measurement_id)
);

-- 하이퍼테이블 생성
SELECT create_hypertable('spc_measurements', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 압축 정책
ALTER TABLE spc_measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'process_id, item_id'
);

SELECT add_compression_policy('spc_measurements', INTERVAL '30 days', if_not_exists => TRUE);

-- 보관 정책
SELECT add_retention_policy('spc_measurements', INTERVAL '365 days', if_not_exists => TRUE);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_spc_process_time
ON spc_measurements (process_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_spc_item_time
ON spc_measurements (item_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_spc_status_time
ON spc_measurements (status, time DESC) WHERE status != 'NORMAL';

-- ============================================================
-- 3. 알람 히스토리 테이블
-- ============================================================

CREATE TABLE IF NOT EXISTS alarm_history (
    time                TIMESTAMPTZ NOT NULL,
    alarm_id            TEXT NOT NULL,
    alarm_code          TEXT NOT NULL,
    alarm_name          TEXT,
    severity            TEXT NOT NULL,
    category            TEXT,
    source_system       TEXT NOT NULL,
    equipment_id        TEXT NOT NULL,
    chamber_id          TEXT,
    process_id          TEXT,
    lot_id              TEXT,
    wafer_id            TEXT,
    param_id            TEXT,
    message             TEXT,
    triggered_value     DOUBLE PRECISION,
    threshold_value     DOUBLE PRECISION,
    status              TEXT DEFAULT 'ACTIVE',
    acknowledged_at     TIMESTAMPTZ,
    acknowledged_by     TEXT,
    resolved_at         TIMESTAMPTZ,
    resolved_by         TEXT,
    action_taken        TEXT,
    root_cause          TEXT,

    PRIMARY KEY (time, alarm_id)
);

-- 하이퍼테이블 생성
SELECT create_hypertable('alarm_history', 'time',
    chunk_time_interval => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_alarm_equipment_time
ON alarm_history (equipment_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_alarm_severity_status
ON alarm_history (severity, status, time DESC);

CREATE INDEX IF NOT EXISTS idx_alarm_lot
ON alarm_history (lot_id, time DESC) WHERE lot_id IS NOT NULL;

-- ============================================================
-- 4. 집계 테이블 (Continuous Aggregates)
-- ============================================================

-- 설비별 시간대 통계 (5분 단위)
CREATE MATERIALIZED VIEW IF NOT EXISTS fdc_stats_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    equipment_id,
    param_id,
    COUNT(*) AS sample_count,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    STDDEV(value) AS std_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) AS median_value
FROM fdc_measurements
GROUP BY bucket, equipment_id, param_id
WITH NO DATA;

-- 자동 갱신 정책
SELECT add_continuous_aggregate_policy('fdc_stats_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- 공정별 SPC 통계 (1시간 단위)
CREATE MATERIALIZED VIEW IF NOT EXISTS spc_stats_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    process_id,
    item_id,
    COUNT(*) AS sample_count,
    AVG(value) AS avg_value,
    STDDEV(value) AS std_value,
    AVG(cp) AS avg_cp,
    AVG(cpk) AS avg_cpk,
    COUNT(*) FILTER (WHERE status != 'NORMAL') AS violation_count
FROM spc_measurements
GROUP BY bucket, process_id, item_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('spc_stats_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- 설비별 알람 통계 (1시간 단위)
CREATE MATERIALIZED VIEW IF NOT EXISTS alarm_stats_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    equipment_id,
    severity,
    source_system,
    COUNT(*) AS alarm_count
FROM alarm_history
GROUP BY bucket, equipment_id, severity, source_system
WITH NO DATA;

SELECT add_continuous_aggregate_policy('alarm_stats_hourly',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- ============================================================
-- 5. 유틸리티 함수
-- ============================================================

-- SPC 관리도 규칙 위반 체크 (Western Electric Rules)
CREATE OR REPLACE FUNCTION check_spc_rules(
    p_values DOUBLE PRECISION[],
    p_ucl DOUBLE PRECISION,
    p_lcl DOUBLE PRECISION,
    p_center DOUBLE PRECISION
)
RETURNS TEXT[]
LANGUAGE plpgsql
AS $$
DECLARE
    violations TEXT[] := '{}';
    sigma DOUBLE PRECISION;
    one_sigma DOUBLE PRECISION;
    two_sigma DOUBLE PRECISION;
    last_value DOUBLE PRECISION;
    arr_len INTEGER;
BEGIN
    arr_len := array_length(p_values, 1);
    IF arr_len IS NULL OR arr_len < 1 THEN
        RETURN violations;
    END IF;

    last_value := p_values[arr_len];
    sigma := (p_ucl - p_center) / 3;
    one_sigma := p_center + sigma;
    two_sigma := p_center + 2 * sigma;

    -- Rule 1: 1점이 관리한계 초과
    IF last_value > p_ucl OR last_value < p_lcl THEN
        violations := array_append(violations, 'RULE1_OOC');
    END IF;

    -- Rule 2: 연속 9점이 중심선 한쪽
    IF arr_len >= 9 THEN
        IF (SELECT COUNT(*) FROM unnest(p_values[arr_len-8:arr_len]) v WHERE v > p_center) = 9 OR
           (SELECT COUNT(*) FROM unnest(p_values[arr_len-8:arr_len]) v WHERE v < p_center) = 9 THEN
            violations := array_append(violations, 'RULE2_RUN9');
        END IF;
    END IF;

    -- Rule 3: 연속 6점 증가 또는 감소
    IF arr_len >= 6 THEN
        -- 구현 생략 (단순화)
        NULL;
    END IF;

    RETURN violations;
END;
$$;

-- ============================================================
-- 완료 메시지
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB 스키마 초기화 완료';
    RAISE NOTICE '- 테이블: fdc_measurements, spc_measurements, alarm_history';
    RAISE NOTICE '- Continuous Aggregates: fdc_stats_5min, spc_stats_hourly, alarm_stats_hourly';
END $$;
