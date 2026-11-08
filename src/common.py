"""Funciones compartidas por Beta 0 (secuencial) y Beta 2 (paralelo).

Tener una sola definicion de datos, split, entrenamiento de una semilla y
reduccion garantiza que ambas versiones ejecuten exactamente el mismo workload.
"""
import os

# Paso 8: un hilo por proceso en las bibliotecas numericas. Debe ejecutarse
# ANTES de importar numpy/scikit-learn, por eso va al inicio de este modulo
# (importa `common` antes que numpy en los scripts que lo usen).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

# Hiperparametros congelados de la campana (no mezclar campanas si cambian)
PARAMS_DEFECTO = dict(
    hidden_layer_sizes=(10, 10),
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=200,
)


def generar_datos(n_samples, n_features=20, n_informative=15, random_state=42):
    """Dataset sintetico binario con n_samples exacto."""
    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=2,
        random_state=random_state,
    )


def dividir_datos(X, y, test_size=0.2, split_seed=0):
    """Particion train/test que se hace UNA sola vez por ejecucion."""
    return train_test_split(X, y, test_size=test_size, random_state=split_seed)


def entrenar_una_semilla(seed, X_train, y_train, X_test, y_test, params):
    """Entrena y evalua el MLP de la semilla `seed`.

    Es el cuerpo del `for s in S pardo` del pseudocodigo.
    Devuelve (seed, accuracy, modelo).
    """
    modelo = MLPClassifier(random_state=seed, **params)
    modelo.fit(X_train, y_train)
    acc = accuracy_score(y_test, modelo.predict(X_test))
    return seed, acc, modelo


def reduce_max_arbol(acc):
    """Paso 7: reduccion de maximo por rondas (arbol binario).

    En cada ronda se comparan pares disjuntos de candidatos; un candidato
    sin pareja pasa a la siguiente ronda. En empate gana la MENOR semilla
    (igual que el baseline secuencial, que reemplaza solo con '>').

    Retorna (mejor_acc, mejor_seed, rondas). El numero de rondas es
    ceil(log2 m), la profundidad del arbol PRAM. Aqui la ejecuta un solo
    proceso, por lo que el tiempo real es Theta(m): la profundidad
    Theta(log m) pertenece al modelo, no a la medicion.
    """
    candidatos = list(range(len(acc)))
    rondas = 0
    while len(candidatos) > 1:
        siguiente = []
        for i in range(0, len(candidatos) - 1, 2):
            izq, der = candidatos[i], candidatos[i + 1]
            siguiente.append(izq if acc[izq] >= acc[der] else der)
        if len(candidatos) % 2 == 1:
            siguiente.append(candidatos[-1])
        candidatos = siguiente
        rondas += 1
    s = candidatos[0]
    return float(acc[s]), s, rondas