"""`evaluate_production_vocab.py`: las piezas de la medición con el vocabulario de producción,
sobre hogares sintéticos (CASAS no viene en el repo). No reproduce los números del README: prueba
que las columnas, los desvíos inyectados y la elección de parámetros hacen lo que dicen."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

DESVIOS_DE_UN_DIA = ("dia_sin_comida", "dia_sin_cocinar", "exceso_descanso", "exceso_ocio_sentado")


def _dias(labels: pd.DataFrame, escenario: str, hogar: str = "hA") -> list:
    return sorted(labels[(labels["escenario"] == escenario) & (labels["hogar"] == hogar)]["fecha"])


def test_el_vocabulario_es_el_de_produccion_menos_lo_que_casas_no_tiene(epv):
    # `--con-puerta`: el vocabulario de producción.
    seis = ("cam_cocina", "cam_comida", "cam_descanso_sofa", "cam_entrar_casa", "cam_ocio_sentado", "cam_salir_casa")
    assert epv.tracked_classes(con_puerta=True) == seis
    assert epv.SIN_EQUIVALENTE_EN_CASAS == ("cam_moverse",)
    assert set(seis) | set(epv.SIN_EQUIVALENTE_EN_CASAS) == set(epv.LIVE_TRACKED_CLASSES)
    assert epv.escenarios(con_puerta=True) == [*DESVIOS_DE_UN_DIA, "semana_sin_salir"]
    # Sin la opción: la variante sin entradas ni salidas, evaluada y no adoptada.
    cuatro = ("cam_cocina", "cam_comida", "cam_descanso_sofa", "cam_ocio_sentado")
    assert epv.tracked_classes() == epv.CLASSIFIER_TRACKED == cuatro
    assert epv.escenarios() == list(DESVIOS_DE_UN_DIA)


def test_las_columnas_siguen_el_camino_de_produccion(epv, hogares):
    cam = epv.add_production_columns(hogares, con_puerta=True)
    dia = cam.iloc[0]
    # comer + beber + medicación son una sola clase; ver televisión es descanso, no ocio sentado.
    assert dia["cam_comida_n"] == 3 + 2 + 1
    assert dia["cam_cocina_n"] == 2 + 1
    assert dia["cam_descanso_sofa_n"] == 2 + 1
    assert dia["cam_ocio_sentado_n"] == 1 + 1 + 1
    assert dia["cam_descanso_sofa_min"] == pytest.approx(dia["descansar_min"] + dia["ver_tv_min"])
    # Entradas y salidas: la cantidad de eventos y ningún minuto, como en producción.
    assert (dia["cam_salir_casa_n"], dia["cam_entrar_casa_n"]) == (1, 1)
    assert (dia["cam_salir_casa_min"], dia["cam_entrar_casa_min"]) == (0.0, 0.0)
    assert not any(c.startswith("cam_dormir") or c.startswith("cam_moverse") for c in cam.columns)

    sin_puerta = epv.add_production_columns(hogares)
    assert not any("salir_casa" in c or "entrar_casa" in c for c in sin_puerta.columns if c.startswith("cam_"))


def test_inyecta_cada_desvio_y_no_evalua_hogares_con_poco_historial(epv, hogares):
    perturbed, labels = epv.inject(hogares, seed=0, con_puerta=True)

    assert sorted(labels["hogar"].unique()) == ["hA", "hB"]
    assert "hCorto" not in perturbed.index.get_level_values("hogar")
    por_hogar = labels[labels["hogar"] == "hA"]["escenario"].value_counts().to_dict()
    assert por_hogar == {
        "real_sin_tocar": 20, "semana_sin_salir": 7, "dia_sin_comida": 1, "dia_sin_cocinar": 1,
        "exceso_descanso": 1, "exceso_ocio_sentado": 1,
    }
    assert not labels.duplicated(["hogar", "fecha"]).any()  # ningún día tiene dos desvíos


def test_cada_desvio_cambia_lo_que_dice_y_nada_mas(epv, hogares):
    perturbed, labels = epv.inject(hogares, seed=0, con_puerta=True)
    antes, despues = epv.add_production_columns(hogares, con_puerta=True), epv.add_production_columns(perturbed, con_puerta=True)

    (f,) = _dias(labels, "dia_sin_comida")
    assert despues.loc[("hA", f), ["cam_comida_n", "cam_comida_min"]].tolist() == [0, 0.0]
    assert despues.loc[("hA", f), "cam_cocina_n"] == antes.loc[("hA", f), "cam_cocina_n"]

    (f,) = _dias(labels, "dia_sin_cocinar")
    assert despues.loc[("hA", f), ["cam_cocina_n", "cam_cocina_min"]].tolist() == [0, 0.0]

    # El exceso sube los minutos al menos 3 horas sobre lo habitual y deja los episodios igual.
    for escenario, clase in (("exceso_descanso", "cam_descanso_sofa"), ("exceso_ocio_sentado", "cam_ocio_sentado")):
        (f,) = _dias(labels, escenario)
        mediana = antes.xs("hA", level="hogar")[f"{clase}_min"].median()
        assert despues.loc[("hA", f), f"{clase}_min"] >= mediana + epv.EXCESO_MIN_EXTRA - 1e-6
        assert despues.loc[("hA", f), f"{clase}_n"] == antes.loc[("hA", f), f"{clase}_n"]

    semana = _dias(labels, "semana_sin_salir")
    todas = sorted(antes.xs("hA", level="hogar").index)
    assert semana == todas[todas.index(semana[0]):][:7]  # 7 días seguidos
    for f in semana:
        assert despues.loc[("hA", f), ["cam_salir_casa_n", "cam_entrar_casa_n"]].tolist() == [0, 0]
        assert despues.loc[("hA", f), "cam_comida_n"] == antes.loc[("hA", f), "cam_comida_n"]

    for f in _dias(labels, "real_sin_tocar"):
        pd.testing.assert_series_equal(despues.loc[("hA", f)], antes.loc[("hA", f)])


def test_la_inyeccion_depende_solo_de_la_semilla(epv, hogares):
    a, b, c = (epv.inject(hogares, seed=s, con_puerta=True) for s in (3, 3, 4))
    pd.testing.assert_frame_equal(a[1], b[1])
    pd.testing.assert_frame_equal(a[0], b[0])
    assert not a[1].equals(c[1])


def test_por_defecto_no_se_inyecta_la_semana_sin_salir(epv, hogares):
    """La medición de producción: sin entradas ni salidas, con los cuatro desvíos de un día."""
    perturbed, sin_puerta = epv.inject(hogares, seed=0)
    assert set(sin_puerta["escenario"]) == {*DESVIOS_DE_UN_DIA, "real_sin_tocar"}
    pd.testing.assert_frame_equal(perturbed[["salir_casa_n", "entrar_casa_n"]], hogares.loc[["hA", "hB"], ["salir_casa_n", "entrar_casa_n"]])


def test_perder_eventos_de_puerta_solo_toca_entradas_y_salidas(epv, hogares):
    perturbed, _ = epv.inject(hogares, seed=0, con_puerta=True)
    igual = epv.drop_door_events(perturbed, loss=0.0, seed=0)
    pd.testing.assert_frame_equal(igual, perturbed)

    todos = epv.drop_door_events(perturbed, loss=1.0, seed=0)
    assert (todos[["salir_casa_n", "entrar_casa_n"]] == 0).all().all()
    otras = [c for c in perturbed.columns if not c.startswith(("salir_casa", "entrar_casa"))]
    pd.testing.assert_frame_equal(todos[otras], perturbed[otras])

    mitad = epv.drop_door_events(perturbed, loss=0.5, seed=0)
    assert 0 < mitad["salir_casa_n"].sum() < perturbed["salir_casa_n"].sum()
    assert (mitad["salir_casa_n"] <= perturbed["salir_casa_n"]).all()  # solo se pierden, no se inventan


def _eventos_de(perturbed: pd.DataFrame, clases: list[str]) -> pd.DataFrame:
    """Un evento por episodio del vector de día, como la tabla de eventos de CASAS."""
    filas = []
    for (hogar, fecha), dia in perturbed.iterrows():
        for clase in clases:
            for i in range(int(dia[f"{clase}_n"])):
                filas.append({"hogar": hogar, "actividad": clase, "fecha": fecha, "inicio": pd.Timestamp(fecha) + pd.Timedelta(hours=8 + i)})
    return pd.DataFrame(filas)


def test_el_error_del_clasificador_mueve_episodios_entre_clases_sin_crear_ni_perder(epv, hogares):
    perturbed, _ = epv.inject(hogares, seed=0, con_puerta=True)
    eventos = _eventos_de(perturbed, list(epv.PRODUCTION_CLASS))
    limpio = epv.add_production_columns(perturbed, con_puerta=True)
    ruido = epv.contaminate(perturbed, eventos, seed=0, con_puerta=True)

    del_clasificador = [f"{c}_n" for c in epv.CLASSIFIER_TRACKED]
    pd.testing.assert_series_equal(
        ruido[del_clasificador + ["cam_moverse_n"]].sum(axis=1), limpio[del_clasificador].sum(axis=1), check_names=False,
    )
    assert ruido["cam_moverse_n"].sum() > 0                      # lo que se confunde con `Moving`
    assert not ruido[del_clasificador].equals(limpio[del_clasificador])
    # Las entradas y salidas no pasan por el clasificador.
    pd.testing.assert_frame_equal(ruido[["cam_salir_casa_n", "cam_entrar_casa_n"]], limpio[["cam_salir_casa_n", "cam_entrar_casa_n"]])
    pd.testing.assert_frame_equal(epv.contaminate(perturbed, eventos, seed=0, con_puerta=True), ruido)
    # Sin puerta: las mismas columnas del clasificador y ninguna de entradas o salidas.
    sin_puerta = epv.contaminate(perturbed, eventos, seed=0)
    pd.testing.assert_frame_equal(sin_puerta[del_clasificador + ["cam_moverse_n"]], ruido[del_clasificador + ["cam_moverse_n"]])
    assert not any("salir_casa" in c or "entrar_casa" in c for c in sin_puerta.columns if c.startswith("cam_"))


def test_el_detector_por_clase_ve_las_ausencias_y_la_semana_sin_salir_pero_no_el_exceso(epv, hogares, baseline_modulo):
    """En un hogar regular: las ausencias y la semana sin salir se detectan por su clase; el
    exceso de descanso (más minutos, mismos episodios) no, porque el detector cuenta episodios."""
    perturbed, labels = epv.inject(hogares, seed=1, con_puerta=True)
    config = baseline_modulo("src.class_deviations").production_config()
    columnas = epv.add_production_columns(perturbed, con_puerta=True)
    todo = epv.evaluate_production(columnas, labels, epv.tracked_classes(con_puerta=True), config)
    r = todo["por_clase"]
    # Dejando de la puerta solo la ausencia sostenida (lo que emite el proceso diario) se pierde,
    # como mucho, el primer día de cada semana sin salir, y nunca se agrega un falso positivo.
    filtrado = todo["por_clase_filtrado"]
    assert filtrado["falsos_positivos"] <= r["falsos_positivos"]
    assert r["detectados_por_escenario"]["semana_sin_salir"] - 2 <= filtrado["detectados_por_escenario"]["semana_sin_salir"] <= r["detectados_por_escenario"]["semana_sin_salir"]
    assert filtrado["detectados_por_escenario"]["dia_sin_comida"] == r["detectados_por_escenario"]["dia_sin_comida"]

    assert r["falsos_positivos"] == 0 and r["n_neg"] == 40
    assert r["n_por_escenario"] == {
        "dia_sin_comida": 2, "dia_sin_cocinar": 2, "exceso_descanso": 2, "exceso_ocio_sentado": 2, "semana_sin_salir": 14,
    }
    assert r["detectados_por_escenario"]["dia_sin_comida"] == 2
    assert r["detectados_por_escenario"]["dia_sin_cocinar"] == 2
    assert r["detectados_por_escenario"]["exceso_descanso"] == 0
    assert r["detectados_por_escenario"]["exceso_ocio_sentado"] == 0
    # De los 7 días sin salir, el primero todavía no es una ausencia sostenida.
    assert r["detectados_por_escenario"]["semana_sin_salir"] >= 2 * (7 - config.dias_consecutivos_ausencia + 1) - 2
    assert r["recall_por_clase_del_desvio"]["semana_sin_salir"] > 0.5


def test_sin_puerta_tambien_se_ven_las_ausencias_y_no_el_exceso(epv, hogares, baseline_modulo):
    """Lo mismo en la variante sin entradas ni salidas, con sus propios parámetros."""
    perturbed, labels = epv.inject(hogares, seed=1)
    config = baseline_modulo("src.class_deviations").production_config(con_puerta=False)
    r = epv.evaluate_production(epv.add_production_columns(perturbed), labels, epv.tracked_classes(), config)

    por_clase = r["por_clase"]
    assert por_clase["falsos_positivos"] == 0 and por_clase["n_neg"] == 40
    assert por_clase["n_por_escenario"] == {"dia_sin_comida": 2, "dia_sin_cocinar": 2, "exceso_descanso": 2, "exceso_ocio_sentado": 2}
    assert por_clase["detectados_por_escenario"] == {"dia_sin_comida": 2, "dia_sin_cocinar": 2, "exceso_descanso": 0, "exceso_ocio_sentado": 0}
    assert set(por_clase["fp_por_clase"]) == set(epv.CLASSIFIER_TRACKED)
    # "Los dos juntos" nunca detecta menos que el detector por clase solo.
    assert r["combinado"]["detectados"] >= por_clase["detectados"]


def test_los_indicadores_de_rutina_ven_el_exceso_y_no_la_falta_de_comida(epv, hogares):
    """Lo que el detector por clase no ve lo cubre el indicador de sedentarismo de la app."""
    perturbed, labels = epv.inject(hogares, seed=0, con_puerta=True)
    r = epv.evaluate_rutina(epv.add_production_columns(perturbed, con_puerta=True), labels)

    assert r["sedentarismo"]["detectados_por_escenario"]["exceso_descanso"] == 2
    assert r["sedentarismo"]["detectados_por_escenario"]["exceso_ocio_sentado"] == 2
    assert r["cocina"]["detectados_por_escenario"]["dia_sin_cocinar"] == 2
    assert r["cualquiera"]["detectados_por_escenario"]["dia_sin_comida"] == 0
    assert r["cualquiera"]["detectados_por_escenario"]["semana_sin_salir"] == 0
    assert r["cualquiera"]["falsos_positivos"] == 0


def _barrido_falso(epv, votos_por_semilla: dict[int, int]) -> dict[int, dict]:
    """Un barrido por semilla donde un solo candidato detecta todo sin falsos positivos."""
    configs = [asdict(c) for c in epv._grid_configs()]
    homes = [f"h{i:02d}" for i in range(10)]
    filas = pd.DataFrame([
        {"hogar": h, "fecha": i, "escenario": "dia_sin_comida" if positivo else "real_sin_tocar", "es_positivo": positivo}
        for h in homes for i, positivo in enumerate([True, True, False, False, False])
    ])
    grids = {}
    for seed, ganador in votos_por_semilla.items():
        hits = np.zeros((len(configs), len(filas)), dtype=bool)
        hits[ganador] = filas["es_positivo"].to_numpy()
        grids[seed] = {"filas": filas, "configs": configs, "hits": hits, "homes": homes, "etiquetas": filas[["hogar", "es_positivo"]]}
    return grids


def test_los_parametros_de_produccion_son_los_mas_votados_por_las_particiones(epv):
    grids = _barrido_falso(epv, {0: 17, 1: 17, 2: 101})
    seleccion = epv.select_production_config(grids, n_folds=5)

    esperado = {k: grids[0]["configs"][17][k] for k in epv.GRID_FIELDS}
    assert seleccion["elegido"] == esperado
    assert (seleccion["votos"], seleccion["particiones"]) == (10, 15)
    assert [v["votos"] for v in seleccion["votacion"]] == [10, 5]
    assert seleccion["cv"][0]["pooled"]["recall_out_of_fold"] == 1.0
    assert seleccion["cv"][0]["pooled"]["fp_rate_out_of_fold"] == 0.0


def test_los_parametros_de_produccion_salen_del_barrido_y_no_tocan_los_defaults(epv, baseline_modulo):
    class_deviations = baseline_modulo("src.class_deviations")
    produccion = class_deviations.production_config()
    candidatos = {tuple(getattr(c, k) for k in epv.GRID_FIELDS) for c in epv._grid_configs()}
    assert tuple(getattr(produccion, k) for k in epv.GRID_FIELDS) in candidatos
    assert produccion.ausencia_metodo == "umbral" and produccion.min_window_days == 7 and produccion.window_days == 30
    # Producción: el juego votado con entradas y salidas en el vocabulario.
    assert tuple(getattr(produccion, k) for k in epv.GRID_FIELDS) == (0.6, 5.0, 2, True, 1, False)
    assert class_deviations.production_config(con_puerta=True) == produccion
    # La variante sin puerta conserva los suyos (los números medidos sin puerta).
    sin_puerta = class_deviations.production_config(con_puerta=False)
    assert tuple(getattr(sin_puerta, k) for k in epv.GRID_FIELDS) == (0.6, 2.5, 2, False, 2, False)
    assert tuple(getattr(sin_puerta, k) for k in epv.GRID_FIELDS) in candidatos
    assert sin_puerta.min_window_days == 7
    # Los defaults del dataclass son los de las calibraciones históricas: no cambian.
    historicos = class_deviations.ClassDeviationConfig()
    assert (historicos.frac_historica_min, historicos.excess_k, historicos.dias_consecutivos_ausencia) == (0.8, 3.0, 2)
    assert (historicos.separar_fin_de_semana, historicos.dias_consecutivos_exceso_deficit, historicos.ausencia_requiere_racha) == (True, 1, False)
