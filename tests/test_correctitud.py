"""Pruebas de correctitud: Beta 0 (secuencial) vs Beta 2 (paralelo)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402  (primero: fija 1 hilo antes de numpy)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

import beta0_sequential as b0  # noqa: E402
import beta2_parallel as b2  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

N = int(os.environ.get("TEST_N", 1500))
M = int(os.environ.get("TEST_M", 32))


@pytest.fixture(scope="module")
def datos():
    return common.generar_datos(N)


@pytest.fixture(scope="module")
def referencia(datos):
    """Resultado del baseline secuencial (Beta 0) para S={0..M-1}."""
    X, y = datos
    _, acc, seed, tiempos = b0.entrenamiento_secuencial(X, y, M)
    return dict(acc=acc, seed=seed,
                acc_por_semilla=[t["accuracy"] for t in tiempos])


@pytest.fixture(scope="module")
def n_iter_p1(datos):
    """n_iter_ por semilla con p=1: referencia para los demas p."""
    X, y = datos
    return b2.entrenamiento_paralelo(X, y, M, 1)["n_iter"]


# --- Reduccion en arbol (rapidas, no entrenan nada) -------------------------
def test_reduccion_desempate_menor_semilla():
    acc, seed, rondas = common.reduce_max_arbol([0.5, 0.9, 0.9, 0.1, 0.9])
    assert (acc, seed, rondas) == (0.9, 1, 3)


def test_reduccion_igual_a_argmax():
    rng = np.random.default_rng(0)
    for m in range(1, 40):
        a = np.round(rng.random(m), 1)  # redondeo -> muchos empates
        _, s, _ = common.reduce_max_arbol(a)
        assert s == int(np.argmax(a))  # argmax devuelve la primera aparicion


# --- Equivalencia secuencial vs paralelo ------------------------------------
@pytest.mark.parametrize("p", [1, 2, 4, 8])
def test_paralelo_igual_a_secuencial(datos, referencia, n_iter_p1, p):
    X, y = datos
    r = b2.entrenamiento_paralelo(X, y, M, p)
    assert r["mejor_seed"] == referencia["seed"]
    assert r["mejor_accuracy"] == referencia["acc"]
    assert list(r["acc"]) == referencia["acc_por_semilla"]
    assert list(r["n_iter"]) == list(n_iter_p1)


def test_modelo_ganador_reconstruido(datos):
    X, y = datos
    r = b2.entrenamiento_paralelo(X, y, 8, 2)
    X_train, X_test, y_train, y_test = common.dividir_datos(X, y)
    modelo = b2.modelo_ganador(r)
    assert accuracy_score(y_test, modelo.predict(X_test)) == r["mejor_accuracy"]


# --- m y p independientes ----------------------------------------------------
def test_p_mayor_que_m(datos):
    X, y = datos
    r = b2.entrenamiento_paralelo(X, y, 3, 8)
    assert len(r["acc"]) == 3


# --- Modo spawn (Windows / macOS) -------------------------------------------
def test_spawn_igual_a_fork(datos):
    X, y = datos
    a = b2.entrenamiento_paralelo(X, y, 4, 2, metodo_inicio="fork")
    b = b2.entrenamiento_paralelo(X, y, 4, 2, metodo_inicio="spawn")
    assert list(a["acc"]) == list(b["acc"])
    assert a["mejor_seed"] == b["mejor_seed"]