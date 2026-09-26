"""반출 경계 테스트

사내 실데이터가 리포지토리에 닿기 전에 경계가 실제로 작동하는지 검증한다.
경계는 사람의 판단이 아니라 기계가 지켜야 한다.
"""
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.check_leaks import check  # noqa: E402


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO,
                         capture_output=True, text=True, check=False)
    return out.stdout.splitlines()


class TestGitignoreCoversSite:
    """사이트 오버레이가 추적되지 않아야 한다"""

    def test_no_real_site_files_tracked(self):
        bad = [
            f for f in tracked_files()
            if f.startswith("config/site/")
            and not f.endswith(".example.yaml")
            and not f.endswith(".example.txt")
            and not f.endswith("README.md")
        ]
        assert not bad, f"사이트 오버레이 실제 파일이 추적되고 있습니다: {bad}"

    def test_landing_not_tracked(self):
        bad = [f for f in tracked_files() if f.startswith("landing/")]
        assert not bad, f"실데이터 랜딩 폴더가 추적되고 있습니다: {bad}"

    def test_no_parquet_tracked(self):
        bad = [f for f in tracked_files() if f.endswith(".parquet")]
        assert not bad, f"Parquet 파일이 추적되고 있습니다: {bad}"

    def test_gitignore_declares_site_and_landing(self):
        gi = (REPO / ".gitignore").read_text(encoding="utf-8")
        for rule in ("config/site/**", "landing/", "profiles_out/"):
            assert rule in gi, f".gitignore 에 {rule} 규칙이 없습니다"

    def test_example_overlay_exists(self):
        # 예시가 없으면 사이트 담당자가 무엇을 채워야 할지 알 수 없다
        assert (REPO / "config/site/mappings/battery.example.yaml").exists()
        assert (REPO / "config/site/leak_patterns.example.txt").exists()
        assert (REPO / "config/site/README.md").exists()


class TestLeakCheck:
    """누출 검사가 실제로 잡아내는지"""

    def _write(self, tmp_path, rel, content):
        p = REPO / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_site_overlay_file_is_blocked(self):
        problems = check(["config/site/mappings/battery.yaml"])
        assert problems, "사이트 오버레이 파일이 통과했습니다"
        assert "커밋할 수 없습니다" in problems[0]

    def test_example_overlay_is_allowed(self):
        assert not check(["config/site/mappings/battery.example.yaml"])

    def test_site_readme_is_allowed(self):
        assert not check(["config/site/README.md"])

    def test_data_file_is_blocked(self):
        problems = check(["some/dir/measurements.parquet"])
        assert problems and "데이터 파일" in problems[0]

    def test_template_is_exempt(self):
        # 빈 양식은 코어 자산이다. 채워진 파일만 사이트로 간다.
        assert not check(["analytics/templates/expert_knowledge_template.xlsx"])

    def test_fstring_dsn_is_not_a_credential(self):
        # f"jdbc:postgresql://{host}:{port}/{db}" 는 템플릿이지 실제 값이 아니다
        assert not check(["stream-processing/src/config.py"])

    def test_real_credential_is_blocked(self, tmp_path):
        # 문자열을 조각으로 조립한다. 이 테스트 파일 자체가 검사에 걸리면
        # 커밋이 막히기 때문이다. (실제로 한 번 막혔고, 그게 훅이 동작한다는 증거다)
        host = ".".join(["10", "20", "30", "40"])
        dsn = "ora" + "cle://svc:" + "Hunter2Pass" + "@" + host + ":1521/ORCL"
        rel = "scripts/_boundary_test_tmp.py"
        p = self._write(tmp_path, rel, f'DSN = "{dsn}"\n')
        try:
            problems = check([rel])
            assert problems, "자격증명 포함 DSN 이 통과했습니다"
        finally:
            p.unlink(missing_ok=True)

    def test_site_pattern_is_blocked(self, tmp_path):
        pat = REPO / "config/site/leak_patterns.txt"
        rel = "scripts/_boundary_test_tmp2.py"
        p = self._write(tmp_path, rel, 'Q = "SELECT * FROM TB_COATING_MEAS"\n')
        pat.write_text(r"\bTB_[A-Z_]{4,}\b" + "\n", encoding="utf-8")
        try:
            problems = check([rel])
            assert problems and "사이트 고유 패턴" in problems[0]
        finally:
            p.unlink(missing_ok=True)
            pat.unlink(missing_ok=True)

    def test_clean_core_file_passes(self):
        assert not check(["common/cypher_safe.py", "README.md"])


class TestHookInstalled:
    def test_hook_script_exists_and_executable(self):
        hook = REPO / ".githooks/pre-commit"
        assert hook.exists(), "pre-commit 훅이 없습니다"
        assert hook.stat().st_mode & 0o111, "pre-commit 훅에 실행 권한이 없습니다"
        assert "check_leaks.py" in hook.read_text(encoding="utf-8")


class TestCoreSchemasCarryNoSiteValues:
    """코어 스키마의 sourceMapping 에 실제 테이블/컬럼명이 없어야 한다"""

    def test_source_mappings_are_placeholders(self):
        import yaml

        offenders = []
        for p in (REPO / "ontology/schemas").rglob("*.yaml"):
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            sm = data.get("sourceMapping")
            if not isinstance(sm, dict):
                continue
            for system, spec in sm.items():
                if not isinstance(spec, dict):
                    continue
                table = spec.get("table")
                # 자리표시자는 <...> 형태이거나, 예시임이 명시돼야 한다
                if isinstance(table, str) and table and not table.startswith("<"):
                    offenders.append(f"{p.relative_to(REPO)} :: {system}.table = {table}")
        assert not offenders, (
            "스키마의 sourceMapping 에 구체적 테이블명이 있습니다. "
            "예시는 _example 아래에 두고, 실제 값은 config/site/mappings/ 로 "
            "옮기십시오:\n  " + "\n  ".join(offenders)
        )


@pytest.mark.parametrize("script", [
    "scripts/check_leaks.py",
    "scripts/profile_site_data.py",
    "scripts/generate_battery_genealogy.py",
    "scripts/demo_genealogy_queries.py",
])
def test_scripts_are_syntactically_valid(script):
    import ast
    ast.parse((REPO / script).read_text(encoding="utf-8"))
