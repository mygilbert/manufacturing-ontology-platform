"""인증 / 주체 식별 테스트

Action 계층은 "누가"를 요구한다. 인증이 뚫리면 감사 기록이 무의미해진다.
"""
import importlib
import pathlib
import sys

import pytest

fastapi = pytest.importorskip("fastapi", reason="API 의존성이 없는 환경에서는 건너뜀")

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

# python-jose 는 backends 패키지에서 cryptography 를 즉시 import 한다.
# cryptography 가 깨진 환경에서는 ImportError 가 아니라 pyo3 PanicException
# (BaseException 하위)이 올라오므로 importorskip 으로는 잡히지 않는다.
try:
    from jose import jwt as jose_jwt
    JOSE_AVAILABLE = True
except BaseException:  # noqa: BLE001 - PanicException 포함해 모두 잡아야 한다
    jose_jwt = None
    JOSE_AVAILABLE = False

needs_jose = pytest.mark.skipif(
    not JOSE_AVAILABLE, reason="python-jose/cryptography 를 사용할 수 없는 환경"
)


@pytest.fixture(scope="module")
def auth_module(monkeypatch_session=None):
    # api/src 를 경로에 올리고 config 를 먼저 고정한다.
    # (routers/agent.py 가 analytics/src 를 sys.path 앞에 끼워넣어 config 를
    #  가릴 수 있으므로, auth 보다 먼저 import 해 캐시에 올려둔다)
    sys.path.insert(0, str(pathlib.Path("api/src").resolve()))
    importlib.import_module("config")
    return importlib.import_module("auth")


@pytest.fixture
def client(auth_module):
    app = FastAPI()

    @app.get("/whoami")
    async def whoami(principal=Depends(auth_module.get_current_principal)):
        return {"user_id": principal.user_id, "roles": list(principal.roles)}

    return TestClient(app, raise_server_exceptions=False)


class TestTokenAuth:
    def test_no_token_is_rejected(self, client, auth_module, monkeypatch):
        monkeypatch.setattr(auth_module, "AUTH_DEV_MODE", False)
        assert client.get("/whoami").status_code == 401

    @needs_jose
    def test_invalid_token_is_rejected(self, client, auth_module, monkeypatch):
        monkeypatch.setattr(auth_module, "AUTH_DEV_MODE", False)
        r = client.get("/whoami", headers={"Authorization": "Bearer not-a-jwt"})
        assert r.status_code == 401

    @needs_jose
    def test_valid_token_resolves_principal(self, client, auth_module, monkeypatch):
        monkeypatch.setattr(auth_module, "AUTH_DEV_MODE", False)
        token = auth_module.create_access_token(
            "hong.gildong", roles=["process_engineer"], display_name="홍길동"
        )
        r = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json() == {"user_id": "hong.gildong", "roles": ["process_engineer"]}

    @needs_jose
    def test_token_without_sub_is_rejected(self, auth_module, monkeypatch):
        from config import settings

        monkeypatch.setattr(auth_module, "AUTH_DEV_MODE", False)
        bad = jose_jwt.encode({"roles": ["x"]}, settings.jwt_secret, algorithm=settings.jwt_algorithm)
        with pytest.raises(Exception):
            auth_module.principal_from_token(bad)

    @needs_jose
    def test_expired_token_is_rejected(self, auth_module, monkeypatch):
        monkeypatch.setattr(auth_module, "AUTH_DEV_MODE", False)
        token = auth_module.create_access_token("a", expires_minutes=-1)
        with pytest.raises(Exception):
            auth_module.principal_from_token(token)

    def test_dev_mode_bypasses_token(self, client, auth_module, monkeypatch):
        # 개발 편의용이며 운영에서는 반드시 꺼야 한다
        monkeypatch.setattr(auth_module, "AUTH_DEV_MODE", True)
        r = client.get("/whoami")
        assert r.status_code == 200
        assert r.json()["user_id"] == "dev.user"


class TestAgentPrincipal:
    @needs_jose
    def test_agent_claims_are_carried(self, auth_module):
        from config import settings

        token = jose_jwt.encode(
            {"sub": "fdc-agent", "roles": ["process_engineer"],
             "is_agent": True, "on_behalf_of": "hong"},
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
        )
        p = auth_module.principal_from_token(token)
        assert p.is_agent is True
        assert p.on_behalf_of == "hong"
