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
    assert set(todo) == {"por_clase", "multivariado", "combinado"}

    assert r["falsos_positivos"] == 0 and r["n_neg"] == 40
    assert r["n_por_escenario"] == {
        "dia_sin_comida": 2, "dia_sin_cocinar": 2, "exceso_descanso": 2, "exceso_ocio_sentado": 2, "semana_sin_salir": 14,
    }
    assert r["detectados_por_escenario"]["dia_sin_comida"] == 2
    assert r["detectados_por_escenario"]["dia_sin_cocinar"] == 2
    assert r["detectados_por_escenario"]["exceso_descanso"] == 0
    assert r["detectados_por_escenario"]["exceso_ocio_sentado"] == 0
    # La puerta dispara desde el día `dias_consecutivos_ausencia_puerta` de la semana sin salir.
    assert r["detectados_por_escenario"]["semana_sin_salir"] == 2 * (7 - config.dias_consecutivos_ausencia_puerta + 1)
    assert r["recall_por_clase_del_desvio"]["semana_sin_salir"] > 0.5
    # Con 2, 3 y 4 días se pierden, en cada hogar, los primeros días de la semana.
    cd = baseline_modulo("src.class_deviations")
    for dias_puerta in (2, 3, 4):
        otra = cd.ClassDeviationConfig(**{**cd.PRODUCTION_PARAMS, "dias_consecutivos_ausencia_puerta": dias_puerta})
        d = epv.evaluate_production(columnas, labels, epv.tracked_classes(con_puerta=True), otra)["por_clase"]
        assert d["detectados_por_escenario"]["semana_sin_salir"] == 2 * (7 - dias_puerta + 1)
        assert d["falsos_positivos"] == 0


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


def test_el_barrido_con_puerta_suma_los_dos_parametros_y_da_lo_mismo_que_evaluar_entero(epv, hogares, monkeypatch):
    """`evaluate_grid` evalúa las clases de puerta aparte y las une; cada candidato tiene que
    marcar los mismos días que el detector corrido entero con esa combinación."""
    dos = [epv.ClassDeviationConfig(frac_historica_min=0.6), epv.ClassDeviationConfig(frac_historica_min=0.9, separar_fin_de_semana=False)]
    monkeypatch.setattr(epv, "_grid_configs", lambda: iter(dos))
    perturbed, labels = epv.inject(hogares, seed=2, con_puerta=True)
    columnas = epv.add_production_columns(perturbed, con_puerta=True)
    tracked = epv.tracked_classes(con_puerta=True)

    grid = epv.evaluate_grid(columnas, labels, "_n", tracked)

    assert grid["campos"] == epv.GRID_FIELDS + epv.DOOR_GRID_FIELDS
    assert len(grid["configs"]) == len(dos) * len(epv.DIAS_PUERTA_GRID) * len(epv.PUERTA_UN_DIA_GRID)
    claves = list(zip(labels["hogar"], labels["fecha"]))
    for config, hits in zip(grid["configs"], grid["hits"]):
        entero = epv._dias(epv.score_class_deviations(
            columnas, tracked_classes=tracked, config=epv.ClassDeviationConfig(**config), clases_de_puerta=epv.DOOR_TRACKED,
        ))
        assert hits.tolist() == [c in entero for c in claves]
    # Las cinco combinaciones que disparan desde el primer día son el mismo detector.
    del_primer_dia = [
        h.tolist() for c, h in zip(grid["configs"], grid["hits"])
        if c["frac_historica_min"] == 0.6 and (c["puerta_dispara_un_dia"] or c["dias_consecutivos_ausencia_puerta"] == 1)
    ]
    assert len(del_primer_dia) == 5 and all(h == del_primer_dia[0] for h in del_primer_dia)
    # Sin clases de puerta el barrido es el de siempre.
    monkeypatch.setattr(epv, "_grid_configs", lambda: iter(dos))
    sin_puerta = epv.evaluate_grid(epv.add_production_columns(perturbed), labels, "_n", epv.tracked_classes())
    assert sin_puerta["campos"] == epv.GRID_FIELDS and len(sin_puerta["configs"]) == len(dos)


def test_la_deteccion_del_sistema_suma_el_indicador_que_corresponde_a_cada_desvio(epv):
    """Una alerta cuenta siempre. El indicador solo cuenta para su desvío; en un día normal,
    cualquier indicador es un falso positivo."""
    dias = ["dia_sin_comida", "dia_sin_cocinar", "exceso_descanso", "exceso_ocio_sentado", "semana_sin_salir", "real_sin_tocar", "real_sin_tocar", "real_sin_tocar"]
    labels = pd.DataFrame({"hogar": "h", "fecha": range(len(dias)), "escenario": dias, "es_positivo": [d != "real_sin_tocar" for d in dias]})

    def _dia(n: int) -> tuple:
        return ("h", n)

    solo_alertas = epv.evaluate_sistema(labels, {_dia(0), _dia(5)}, {"sedentarismo": set(), "cocina": set()})
    assert (solo_alertas["detectados"], solo_alertas["falsos_positivos"]) == (1, 1)

    # sedentarismo marca el día sin comida (no le corresponde), los dos excesos y un día normal;
    # cocina marca el día sin cocinar, la semana sin salir (no le corresponde) y otro día normal.
    rutina = {"sedentarismo": {_dia(0), _dia(2), _dia(3), _dia(6)}, "cocina": {_dia(1), _dia(4), _dia(7)}}
    r = epv.evaluate_sistema(labels, set(), rutina)
    assert r["detectados_por_escenario"] == {
        "dia_sin_comida": 0, "dia_sin_cocinar": 1, "exceso_descanso": 1, "exceso_ocio_sentado": 1, "semana_sin_salir": 0,
    }
    assert (r["falsos_positivos"], r["n_neg"]) == (2, 3)

    filas = epv._fila_subtotales("sistema", [r])
    assert filas["ausencias"].endswith("(1/3)") and filas["excesos"].endswith("(2/2)")


def test_la_matriz_de_confusion_de_un_archivo_se_lee_en_el_orden_del_clasificador(epv, tmp_path):
    categorias = list(reversed(epv.CATEGORIES))
    cuentas = [[10 * (i + 1) if i == j else 1 for j in range(6)] for i in range(6)]
    archivo = tmp_path / "matriz.json"
    archivo.write_text(epv.json.dumps({"categorias": categorias, "matriz": cuentas}), encoding="utf-8")

    matriz = epv.load_confusion(archivo)

    assert matriz.shape == (6, 6) and np.allclose(matriz.sum(axis=1), 1.0)
    # `Cooking` es la última fila del archivo: 60 aciertos de 65.
    assert matriz[0, 0] == pytest.approx(60 / 65)
    assert matriz[5, 5] == pytest.approx(10 / 15)


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

    # Con los parámetros de puerta en el barrido, el elegido los incluye.
    con_puerta = _barrido_falso(epv, {0: 17, 1: 17, 2: 101})
    for grid in con_puerta.values():
        grid["campos"] = epv.GRID_FIELDS + epv.DOOR_GRID_FIELDS
    assert set(epv.select_production_config(con_puerta, n_folds=5)["elegido"]) == set(epv.GRID_FIELDS + epv.DOOR_GRID_FIELDS)
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
    # Producción: el juego votado con entradas y salidas en el vocabulario y los parámetros de
    # puerta en el barrido. La puerta dispara desde el primer día sin eventos.
    assert tuple(getattr(produccion, k) for k in epv.GRID_FIELDS) == (0.6, 2.5, 2, True, 2, False)
    assert tuple(getattr(produccion, k) for k in epv.DOOR_GRID_FIELDS) == (1, False)
    assert (1, False) in {(d, u) for d in epv.DIAS_PUERTA_GRID for u in epv.PUERTA_UN_DIA_GRID}
    assert epv.grid_fields(con_puerta=True) == epv.GRID_FIELDS + epv.DOOR_GRID_FIELDS and epv.grid_fields() == epv.GRID_FIELDS
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
    assert (historicos.dias_consecutivos_ausencia_puerta, historicos.puerta_dispara_un_dia) == (2, False)
