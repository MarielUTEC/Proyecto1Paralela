"""Beta 0: baseline secuencial. Entrena las semillas S={0,...,m-1} una tras otra."""
import common  # noqa: F401  (primero: fija 1 hilo antes de numpy)

import argparse
import time

import joblib

from common import (PARAMS_DEFECTO, dividir_datos, entrenar_una_semilla,
                    generar_datos)


def entrenamiento_secuencial(X, y, m, params=None, test_size=0.2,
                             split_seed=0):
    """Retorna: mejor_modelo, mejor_accuracy, mejor_seed, tiempos_por_semilla"""
    if m < 1:
        raise ValueError("m debe ser mayor o igual que 1")
    params = dict(PARAMS_DEFECTO if params is None else params)

    X_train, X_test, y_train, y_test = dividir_datos(
        X, y, test_size=test_size, split_seed=split_seed)

    mejor_seed, mejor_accuracy, mejor_modelo = None, float("-inf"), None
    tiempos = []
    for seed in range(m):
        ini = time.perf_counter()
        s, acc, modelo = entrenar_una_semilla(
            seed, X_train, y_train, X_test, y_test, params)
        tiempos.append({"seed": s, "tiempo": time.perf_counter() - ini,
                        "accuracy": acc, "n_iter": modelo.n_iter_})
        if acc > mejor_accuracy:  # '>' estricto: en empate gana la menor semilla
            mejor_seed, mejor_accuracy, mejor_modelo = s, acc, modelo
    return mejor_modelo, mejor_accuracy, mejor_seed, tiempos


def main():
    parser = argparse.ArgumentParser(description="Beta 0: baseline secuencial")
    parser.add_argument("--m", type=int, default=32, help="numero de semillas")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--output", type=str,
                        default="mlp_model_sequential.joblib")
    args = parser.parse_args()

    X, y = generar_datos(args.n_samples)
    ini = time.perf_counter()
    modelo, acc, seed, tiempos = entrenamiento_secuencial(X, y, args.m)
    total = time.perf_counter() - ini

    print(f"m={args.m} n_samples={args.n_samples} tiempo_secuencial={total:.4f}s "
          f"mejor_seed={seed} accuracy={acc:.4f}")
    for r in tiempos:
        print(f"seed={r['seed']} tiempo={r['tiempo']:.4f}s "
              f"accuracy={r['accuracy']:.4f} n_iter={r['n_iter']}")
    joblib.dump(modelo, args.output)


if __name__ == "__main__":
    main()