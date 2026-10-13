"""Retención local: `EventStore.purge_synced` borra solo eventos ya confirmados por la nube."""
from __future__ import annotations

import sqlite3

import pytest

from schema import EventType, MonitoringEvent
from store import EventStore

DIA = 86400.0
AHORA = 1_800_000_000.0  # epoch fijo: las pruebas no dependen del reloj de la máquina


@pytest.fixture
def store(tmp_path):
    s = EventStore(tmp_path / "events.db")
    yield s
    s.close()


def _agregar(store: EventStore, nombre: str, *, sincronizado_hace_dias: float | None,
             ocurrido_hace_dias: float = 0.0) -> None:
    """Agrega un evento; si `sincronizado_hace_dias` no es None lo marca como enviado entonces."""
    store.add(MonitoringEvent(
        video_id=nombre, event_type=EventType.ACTIVITY, video_timestamp_s=0.0,
        wall_clock_ts=AHORA - ocurrido_hace_dias * DIA,
    ))
    if sincronizado_hace_dias is not None:
        (fila_id, _), = store.unsynced(video_id=nombre)
        store.mark_synced([fila_id], AHORA - sincronizado_hace_dias * DIA)


def _nombres(store: EventStore) -> set[str]:
    return {e.video_id for e in store.all()}


def test_un_evento_sin_sincronizar_nunca_se_borra(store):
    _agregar(store, "pendiente_reciente", sincronizado_hace_dias=None)
    _agregar(store, "pendiente_de_hace_un_anio", sincronizado_hace_dias=None, ocurrido_hace_dias=365)
    _agregar(store, "enviado_viejo", sincronizado_hace_dias=30)

    # Ni con el plazo más corto posible se toca un evento que la nube todavía no confirmó.
    assert store.purge_synced(0.0001, now=AHORA) == 1
    assert store.purge_synced(7, now=AHORA + 3650 * DIA) == 0

    assert _nombres(store) == {"pendiente_reciente", "pendiente_de_hace_un_anio"}
    assert {e.video_id for _, e in store.unsynced()} == {"pendiente_reciente", "pendiente_de_hace_un_anio"}


def test_la_purga_respeta_los_dias_configurados(store):
    _agregar(store, "enviado_hace_6", sincronizado_hace_dias=6)
    _agregar(store, "enviado_hace_8", sincronizado_hace_dias=8)
    _agregar(store, "enviado_hace_40", sincronizado_hace_dias=40)

    assert store.purge_synced(30, now=AHORA) == 1
    assert _nombres(store) == {"enviado_hace_6", "enviado_hace_8"}

    assert store.purge_synced(7, now=AHORA) == 1
    assert _nombres(store) == {"enviado_hace_6"}

    assert store.purge_synced(7, now=AHORA) == 0
    assert _nombres(store) == {"enviado_hace_6"}


def test_el_plazo_corre_desde_la_sincronizacion_no_desde_el_evento(store):
    # Ocurrió hace un año pero la controladora estuvo sin conexión y recién se envió ayer.
    _agregar(store, "viejo_enviado_ayer", sincronizado_hace_dias=1, ocurrido_hace_dias=365)

    assert store.purge_synced(7, now=AHORA) == 0
    assert _nombres(store) == {"viejo_enviado_ayer"}


def test_el_limite_exacto_no_se_borra(store):
    _agregar(store, "justo_7_dias", sincronizado_hace_dias=7)

    assert store.purge_synced(7, now=AHORA) == 0
    assert store.purge_synced(7, now=AHORA + 1) == 1


@pytest.mark.parametrize("dias", [0, -1])
def test_un_plazo_no_positivo_se_rechaza_sin_borrar_nada(store, dias):
    _agregar(store, "enviado_viejo", sincronizado_hace_dias=400)

    with pytest.raises(ValueError):
        store.purge_synced(dias, now=AHORA)
    assert _nombres(store) == {"enviado_viejo"}


def test_no_cambia_el_esquema_de_la_tabla(store, tmp_path):
    _agregar(store, "enviado_viejo", sincronizado_hace_dias=400)
    store.purge_synced(7, now=AHORA)

    conn = sqlite3.connect(tmp_path / "events.db")
    columnas = [fila[1] for fila in conn.execute("PRAGMA table_info(events)")]
    conn.close()
    assert columnas == [
        "id", "video_id", "event_type", "video_timestamp_s", "wall_clock_ts", "cloud_synced_at", "metadata",
        "event_id",  # agregada el 2026-10-13 para la ingestión idempotente, no por la retención
    ]
