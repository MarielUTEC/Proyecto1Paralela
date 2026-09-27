"""Run paired sequential/parallel experiments for the same seed workload."""

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

# Set these before importing NumPy/scikit-learn, including in spawned workers.
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

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
        raise ValueError("repeticiones debe ser mayor o igual que 1")
    if not p_values or any(p < 1 for p in p_values):
        raise ValueError("cada valor de p debe ser mayor o igual que 1")
    if not n_values or any(n < 1 for n in n_values):
        raise ValueError("cada valor de n_samples debe ser mayor o igual que 1")
    if len(set(p_values)) != len(p_values) or len(set(n_values)) != len(n_values):
        raise ValueError("las listas de p y n_samples no deben contener duplicados")


def _environment_metadata(p_values, n_values, repetitions):
    import multiprocessing
    import numpy
    import sklearn
    import threadpoolctl

    try:
        from threadpoolctl import threadpool_info

        threadpools = threadpool_info()
    except ImportError:
        threadpools = []

    cpu_model = None
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    cpu_model = cpu_model or platform.processor() or None
    try:
        physical_memory_bytes = (
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, OSError, ValueError):
        physical_memory_bytes = None
    try:
        available_cpu_ids = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpu_ids = None

    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "available_cpu_ids": available_cpu_ids,
        "available_cpu_count": (
            len(available_cpu_ids) if available_cpu_ids is not None else None
        ),
        "physical_memory_bytes": physical_memory_bytes,
        "multiprocessing_start_method": multiprocessing.get_start_method(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "thread_limits": {
            key: os.environ[key]
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "native_threadpools": threadpools,
        "p_values": p_values,
        "n_samples_values": n_values,
        "repetitions": repetitions,
        "workload_rule": "m=p; seeds=range(p)",
        "dataset": {
            "generator": "sklearn.datasets.make_classification",
            "n_features": 20,
            "n_informative": 15,
            "n_redundant": 5,
            "n_classes": 2,
            "random_state": 42,
        },
        "split": {"test_size": 0.2, "random_state": 0},
        "mlp": {
            "hidden_layer_sizes": [10, 10],
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "solver": "scikit-learn default (adam)",
        },
        "timing": (
            "perf_counter around the training function; dataset generation "
            "excluded; split, process-pool lifecycle and winner selection included"
        ),
    }


def _append_raw_row(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


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
        sequential = modes["sequential"]
        parallel = modes["parallel"]
        group = configurations.setdefault(
            (n_samples, p),
            {"sequential": [], "parallel": [], "speedups": [], "accuracies": []},
        )
        sequential_time = float(sequential["elapsed_seconds"])
        parallel_time = float(parallel["elapsed_seconds"])
        group["sequential"].append(sequential_time)
        group["parallel"].append(parallel_time)
        group["speedups"].append(sequential_time / parallel_time)
        group["accuracies"].extend(
            [float(sequential["accuracy"]), float(parallel["accuracy"])]
        )

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
                    "sequential_stdev_seconds": (
                        statistics.stdev(seq) if len(seq) > 1 else 0.0
                    ),
                    "parallel_median_seconds": statistics.median(par),
                    "parallel_mean_seconds": statistics.mean(par),
                    "parallel_stdev_seconds": (
                        statistics.stdev(par) if len(par) > 1 else 0.0
                    ),
                    "speedup_median": statistics.median(speedups),
                    "speedup_stdev": (
                        statistics.stdev(speedups) if len(speedups) > 1 else 0.0
                    ),
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
    return time.perf_counter() - started, accuracy, best_seed


def run_experiments(
    p_values, n_values, repetitions, raw_path, summary_path, overwrite=False
):
    _validate_values(p_values, n_values, repetitions)
    raw_path = Path(raw_path)
    summary_path = Path(summary_path)
    metadata_path = raw_path.with_suffix(".metadata.json")
    output_paths = {path.resolve() for path in (raw_path, summary_path, metadata_path)}
    if len(output_paths) != 3:
        raise ValueError("raw_out, summary_out y metadatos deben ser rutas distintas")

    if overwrite:
        for path in (raw_path, summary_path, metadata_path):
            path.unlink(missing_ok=True)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if any(path.exists() for path in (raw_path, summary_path, metadata_path)):
        raise FileExistsError(
            "Uno de los archivos de salida ya existe; use --overwrite para "
            "iniciar una campaña nueva."
        )

    metadata_path.write_text(
        json.dumps(
            _environment_metadata(p_values, n_values, repetitions),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        for n_samples in n_values:
            X, y = generar_datos(n_samples)
            for p in p_values:
                for repetition in range(1, repetitions + 1):
                    modes = (
                        ("sequential", "parallel")
                        if repetition % 2
                        else ("parallel", "sequential")
                    )
                    for mode in modes:
                        elapsed, accuracy, best_seed = _measure(mode, X, y, p)
                        _append_raw_row(
                            raw_path,
                            {
                                "timestamp_utc": datetime.now(
                                    timezone.utc
                                ).isoformat(),
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
                            f"n={n_samples:>6} p={p:>2} "
                            f"rep={repetition}/{repetitions} {mode:>10} "
                            f"time={elapsed:.3f}s accuracy={accuracy:.4f} "
                            f"seed={best_seed}",
                            flush=True,
                        )

    print(f"Raw: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compara Beta 0 y Beta 1 con las mismas p semillas para cada "
            "combinacion (n_samples, p)."
        )
    )
    parser.add_argument("--p", type=int, nargs="+", default=P_VALUES_DEFAULT)
    parser.add_argument(
        "--n_samples", type=int, nargs="+", default=N_SAMPLES_DEFAULT
    )
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument(
        "--raw_out", type=Path, default=ROOT / "results/raw/experimentos.csv"
    )
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=ROOT / "results/processed/resumen.csv",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reemplaza los archivos de salida existentes",
    )
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
