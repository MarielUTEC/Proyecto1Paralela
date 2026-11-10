"""Beta 2: paralelismo entre semillas con memoria compartida (CREW).

la memoria compartida se maneja con procesos que comparten segmentos de memoria
  - Lectura concurrente (CR): X_train, y_train, X_test, y_test viven en un
    segmento compartido, de SOLO LECTURA para los workers. El maestro los
    copia una sola vez; cada worker se conecta sin copiarlos.
  - Escritura exclusiva (EW): cada semilla s escribe unicamente en su
    posicion Acc[s], n_iter[s], t_seed[s] y en la fila Mod[s, :] de arreglos
    compartidos.
Los workers siguen siendo procesos del SO, por lo que existen costos que
PRAM no modela
"""
import common  # noqa: F401  (debe importarse antes que numpy: fija 1 hilo)

import argparse
import time
import warnings
from multiprocessing import get_context, shared_memory

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from common import (PARAMS_DEFECTO, dividir_datos, entrenar_una_semilla,
                    generar_datos, reduce_max_arbol)

# Vistas de los segmentos compartidos dentro de cada worker
_G = {}


# ---------------------------------------------------------------- utilidades
def _crear_shm(arr=None, shape=None, dtype=None):
    """Crea un segmento compartido (copia `arr` si se entrega)."""
    if arr is not None:
        arr = np.ascontiguousarray(arr)
        shape, dtype = arr.shape, arr.dtype
    nbytes = max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1)
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    vista = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if arr is not None:
        vista[:] = arr
    else:
        vista[:] = 0
    return shm, vista


def _n_params(d, hidden_layer_sizes):
    """Parametros de un MLP binario: pesos + sesgos (salida de 1 neurona)."""
    tam = [d, *hidden_layer_sizes, 1]
    return sum(a * b for a, b in zip(tam[:-1], tam[1:])) + sum(tam[1:])


def _aplanar(modelo):
    return np.concatenate([w.ravel() for w in modelo.coefs_]
                          + [b.ravel() for b in modelo.intercepts_])


def modelo_desde_params(flat, X_ref, y_ref, params, seed):
    """Reconstruye un MLP (solo para inferencia) desde sus parametros."""
    m = MLPClassifier(random_state=seed, **{**params, "max_iter": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.partial_fit(X_ref[:2], y_ref[:2], classes=np.unique(y_ref))
    pos = 0
    for w in m.coefs_:
        w[...] = flat[pos:pos + w.size].reshape(w.shape)
        pos += w.size
    for b in m.intercepts_:
        b[...] = flat[pos:pos + b.size].reshape(b.shape)
        pos += b.size
    return m


# -------------------------------------------------------------------- worker
def _init_worker(specs, params):
    """Se ejecuta una vez por worker: se conecta a los segmentos."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    threadpool_limits(limits=1)
    _G.clear()
    _G["_shms"] = []
    for clave, (nombre, shape, dtype) in specs.items():
        shm = shared_memory.SharedMemory(name=nombre)
        _G["_shms"].append(shm)  # mantener referencia viva
        _G[clave] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    for clave in ("X_train", "y_train", "X_test", "y_test"):
        _G[clave].flags.writeable = False  # datos de solo lectura
    _G["params"] = params


def _tarea(seed):
    """Cuerpo del PARDO: una tarea = una semilla."""
    ini = time.perf_counter()
    _, acc, modelo = entrenar_una_semilla(
        seed, _G["X_train"], _G["y_train"], _G["X_test"], _G["y_test"],
        _G["params"])
    t = time.perf_counter() - ini
    # Escrituras exclusivas: solo la posicion/fila de esta semilla
    _G["acc"][seed] = acc
    _G["n_iter"][seed] = modelo.n_iter_
    _G["t_seed"][seed] = t
    _G["mod"][seed, :] = _aplanar(modelo)
    return seed


# ------------------------------------------------------------------- maestro
def entrenamiento_paralelo(X, y, m, p, params=None, test_size=0.2,
                           split_seed=0, metodo_inicio=None):
    """Entrena S={0,...,m-1} con p procesos worker y memoria compartida.

    m: numero de semillas (tamano del workload). p: numero de workers.
    Retorna un dict con el ganador, los arreglos por semilla y los tiempos
    por fase (paso 9).
    """
    if m < 1 or p < 1:
        raise ValueError("m y p deben ser >= 1")
    params = dict(PARAMS_DEFECTO if params is None else params)
    ctx = get_context(metodo_inicio)
    T = {}
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    X_train, X_test, y_train, y_test = dividir_datos(
        X, y, test_size=test_size, split_seed=split_seed)
    T["split"] = time.perf_counter() - t0

    shms = []
    try:
        # Copia unica del dataset a memoria compartida + arreglos de salida
        t0 = time.perf_counter()
        specs, vistas = {}, {}
        d = X_train.shape[1]
        P = _n_params(d, params["hidden_layer_sizes"])
        entradas = {
            "X_train": dict(arr=X_train), "y_train": dict(arr=y_train),
            "X_test": dict(arr=X_test), "y_test": dict(arr=y_test),
            "acc": dict(shape=(m,), dtype=np.float64),
            "n_iter": dict(shape=(m,), dtype=np.int64),
            "t_seed": dict(shape=(m,), dtype=np.float64),
            "mod": dict(shape=(m, P), dtype=np.float64),
        }
        for clave, kw in entradas.items():
            shm, vista = _crear_shm(**kw)
            shms.append(shm)
            specs[clave] = (shm.name, vista.shape, vista.dtype.str)
            vistas[clave] = vista
        T["shm_setup"] = time.perf_counter() - t0

        # PARDO sobre S: p workers reparten las m semillas (una a una)
        t0 = time.perf_counter()
        pool = ctx.Pool(processes=p, initializer=_init_worker,
                        initargs=(specs, params))
        T["pool_start"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pool.map(_tarea, range(m), chunksize=1)
        T["map"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        pool.close()
        pool.join()
        T["join"] = time.perf_counter() - t0

        # Reduccion final (paso 7)
        t0 = time.perf_counter()
        acc = vistas["acc"].copy()
        mejor_acc, mejor_seed, rondas = reduce_max_arbol(acc)
        T["reduccion"] = time.perf_counter() - t0

        resultado = dict(
            mejor_seed=mejor_seed, mejor_accuracy=mejor_acc, rondas=rondas,
            acc=acc, n_iter=vistas["n_iter"].copy(),
            t_seed=vistas["t_seed"].copy(),
            params_ganador=vistas["mod"][mejor_seed].copy(),
            X_ref=X_train[:2].copy(), y_ref=y_train.copy(),
        )
    finally:
        t0 = time.perf_counter()
        for shm in shms:
            shm.close()
            shm.unlink()
        T["shm_cleanup"] = time.perf_counter() - t0

    T["total"] = time.perf_counter() - t_total0
    resultado["tiempos"] = T
    resultado["params"] = params
    return resultado


def modelo_ganador(res):
    """Reconstruye el modelo ganador (inferencia) desde Mod[s*]."""
    return modelo_desde_params(res["params_ganador"], res["X_ref"],
                               res["y_ref"], res["params"],
                               res["mejor_seed"])


def main():
    parser = argparse.ArgumentParser(
        description="Beta 2: MLP en paralelo con memoria compartida (m fijo, p variable)")
    parser.add_argument("--m", type=int, default=32, help="numero de semillas")
    parser.add_argument("--p", type=int, default=4, help="numero de procesos")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--output", type=str, default="mlp_model_beta2.joblib")
    args = parser.parse_args()

    X, y = generar_datos(args.n_samples)
    res = entrenamiento_paralelo(X, y, args.m, args.p)
    T = res["tiempos"]
    print(f"m={args.m} p={args.p} n_samples={args.n_samples} "
          f"tiempo={T['total']:.4f}s mejor_seed={res['mejor_seed']} "
          f"accuracy={res['mejor_accuracy']:.4f}")
    print("fases: " + " ".join(f"{k}={v:.4f}s" for k, v in T.items()))
    joblib.dump(modelo_ganador(res), args.output)
    print(f"Modelo ganador guardado en {args.output}")


if __name__ == "__main__":
    main()
