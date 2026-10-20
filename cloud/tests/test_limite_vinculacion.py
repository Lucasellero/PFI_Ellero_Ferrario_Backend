"""`POST /kits/vincular`: el familiar puede equivocarse con el código hasta `MAX_INTENTOS_FALLIDOS`
veces en la ventana; el siguiente intento da 429, aunque el código sea el correcto."""
from __future__ import annotations

import pytest

from app.routers import kits

from conftest import como


@pytest.fixture(autouse=True)
def contador_limpio():
    kits._fallidos.clear()
    yield
    kits._fallidos.clear()


@pytest.fixture
def familiar(cliente) -> str:
    uid = "uid-vincula"
    assert cliente.post("/me", headers=como(uid), json={"nombre": "Vincula"}).status_code == 200
    return uid


def _vincular(cliente, uid: str, codigo: str):
    return cliente.post("/kits/vincular", headers=como(uid), json={"codigo": codigo})


def test_el_limite_es_de_20_intentos():
    assert kits.MAX_INTENTOS_FALLIDOS == 20


def test_hasta_el_limite_cada_codigo_inexistente_da_404(cliente, familiar):
    for _ in range(kits.MAX_INTENTOS_FALLIDOS):
        assert _vincular(cliente, familiar, "ZZZZZZZZ").status_code == 404


def test_pasado_el_limite_da_429_aunque_el_codigo_sea_bueno(cliente, familiar, crear_hogar):
    for _ in range(kits.MAX_INTENTOS_FALLIDOS):
        _vincular(cliente, familiar, "ZZZZZZZZ")

    codigo_bueno = crear_hogar("limite")["codigo"]  # ya vinculado: sin el límite daría 409
    respuesta = _vincular(cliente, familiar, codigo_bueno)

    assert respuesta.status_code == 429
