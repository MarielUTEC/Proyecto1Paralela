import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

# Desactivar multithreading implícito en librerías numéricas
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

# Ajuste de rutas para importar desde src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beta0_sequential import entrenamiento_secuencial, generar_datos
from beta1_parallel import entrenamiento_paralelo

P_VALUES_DEFAULT = [1, 2, 4, 8, 16, 32]
N_SAMPLES_DEFAULT = [5000, 10000, 20000, 40000]

RAW_FIELDS = [
    "timestamp_utc",
    "n_samples",
    "p",
    "m",
    "repetition",
    "mode",
    "elapsed_seconds",
    "accuracy",
    "best_seed",
]

SUMMARY_FIELDS = [
    "n_samples",
    "p",
    "m",
    "complete_repetitions",
    "sequential_median_seconds",
    "sequential_mean_seconds",
    "sequential_stdev_seconds",
    "parallel_median_seconds",
    "parallel_mean_seconds",
    "parallel_stdev_seconds",
    "speedup_median",
    "speedup_stdev",
    "efficiency_median",
    "accuracy_min",
    "accuracy_max",
]


def _validate_values(p_values, n_values, repetitions):
    if repetitions < 1:
        raise ValueError("Las repeticiones deben ser al menos 1")
    if not p_values or any(p < 1 for p in p_values):
        raise ValueError("Los valores de p deben ser >= 1")
    if not n_values or any(n < 1 for n in n_values):
        raise ValueError("Los valores de n_samples deben ser >= 1")


def _save_metadata(path, p_values, n_values, repetitions):
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "p_values": p_values,
        "n_samples_values": n_values,
        "repetitions": repetitions,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def _append_raw_row(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _read_raw(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_summary(raw_path, summary_path):
    raw_rows = _read_raw(raw_path)
    pairs = {}
    for row in raw_rows:
        key = (int(row["n_samples"]), int(row["p"]), int(row["repetition"]))
        pairs.setdefault(key, {})[row["mode"]] = row

    configurations = {}
    for (n_samples, p, _), modes in pairs.items():
        if "sequential" not in modes or "parallel" not in modes:
            continue
        seq = modes["sequential"]
        par = modes["parallel"]
        
        group = configurations.setdefault(
            (n_samples, p),
            {"sequential": [], "parallel": [], "speedups": [], "accuracies": []},
        )
        
        seq_time = float(seq["elapsed_seconds"])
        par_time = float(par["elapsed_seconds"])
        group["sequential"].append(seq_time)
        group["parallel"].append(par_time)
        group["speedups"].append(seq_time / par_time)
        group["accuracies"].extend([float(seq["accuracy"]), float(par["accuracy"])])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        
        for (n_samples, p), group in sorted(configurations.items()):
            seq = group["sequential"]
            par = group["parallel"]
            speedups = group["speedups"]
            
            writer.writerow(
                {
                    "n_samples": n_samples,
                    "p": p,
                    "m": p,
                    "complete_repetitions": len(speedups),
                    "sequential_median_seconds": statistics.median(seq),
                    "sequential_mean_seconds": statistics.mean(seq),
                    "sequential_stdev_seconds": statistics.stdev(seq) if len(seq) > 1 else 0.0,
                    "parallel_median_seconds": statistics.median(par),
                    "parallel_mean_seconds": statistics.mean(par),
                    "parallel_stdev_seconds": statistics.stdev(par) if len(par) > 1 else 0.0,
                    "speedup_median": statistics.median(speedups),
                    "speedup_stdev": statistics.stdev(speedups) if len(speedups) > 1 else 0.0,
                    "efficiency_median": statistics.median(speedups) / p,
                    "accuracy_min": min(group["accuracies"]),
                    "accuracy_max": max(group["accuracies"]),
                }
            )


def _measure(mode, X, y, p):
    started = time.perf_counter()
    if mode == "sequential":
        _, accuracy, best_seed, _ = entrenamiento_secuencial(X, y, m=p)
    else:
        _, accuracy, best_seed = entrenamiento_paralelo(X, y, p=p)
    elapsed = time.perf_counter() - started
    return elapsed, accuracy, best_seed


def run_experiments(p_values, n_values, repetitions, raw_path, summary_path, overwrite=False):
    _validate_values(p_values, n_values, repetitions)
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)
    metadata_path = raw_path.with_suffix(".metadata.json")

    if overwrite:
        for path in (raw_path, summary_path, metadata_path):
            path.unlink(missing_ok=True)
    elif any(path.exists() for path in (raw_path, summary_path, metadata_path)):
        raise FileExistsError("Los archivos de salida ya existen. Usa --overwrite para sobrescribir.")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    _save_metadata(metadata_path, p_values, n_values, repetitions)

    modes = ["sequential", "parallel"]

    with threadpool_limits(limits=1):
        for n_samples in n_values:
            X, y = generar_datos(n_samples)
            for p in p_values:
                for repetition in range(1, repetitions + 1):
                    for mode in modes:
                        elapsed, accuracy, best_seed = _measure(mode, X, y, p)
                        
                        _append_raw_row(
                            raw_path,
                            {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "n_samples": n_samples,
                                "p": p,
                                "m": p,
                                "repetition": repetition,
                                "mode": mode,
                                "elapsed_seconds": f"{elapsed:.9f}",
                                "accuracy": f"{accuracy:.9f}",
                                "best_seed": best_seed,
                            },
                        )
                        _write_summary(raw_path, summary_path)
                        
                        print(
                            f"n={n_samples:>6} p={p:>2} rep={repetition}/{repetitions} "
                            f"{mode:>10} time={elapsed:.3f}s accuracy={accuracy:.4f} seed={best_seed}"
                        )

    print("\nExperimentos finalizados:")
    print(f"  Raw: {raw_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos comparativos secuencial vs paralelo.")
    parser.add_argument("--p", type=int, nargs="+", default=P_VALUES_DEFAULT)
    parser.add_argument("--n_samples", type=int, nargs="+", default=N_SAMPLES_DEFAULT)
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--raw_out", type=Path, default=ROOT / "results/raw/experimentos.csv")
    parser.add_argument("--summary_out", type=Path, default=ROOT / "results/processed/resumen.csv")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe los resultados previos")
    
    args = parser.parse_args()

    run_experiments(
        args.p,
        args.n_samples,
        args.repeticiones,
        args.raw_out,
        args.summary_out,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()