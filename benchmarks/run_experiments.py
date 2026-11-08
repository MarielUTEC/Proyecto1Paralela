"""Campana final: strong scaling con m FIJO y p variable (Beta 0 vs Beta 2).

Para cada n y cada repeticion se mide:
  - T_seq(n, m): baseline secuencial de las semillas {0,...,m-1} (una vez)
  - T_par(n, m, p): Beta 2 con p workers, para cada p, MISMAS m semillas
Metricas (por repeticion emparejada):
  S(n,m,p) = T_seq / T_par      E(n,m,p) = S / p
  overhead = p * T_par - T_seq
El orden seq/paralelo se alterna entre repeticiones para no sesgar el cache.
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import common  # noqa: E402,F401  (primero: fija 1 hilo antes de numpy)

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

from beta0_sequential import entrenamiento_secuencial  # noqa: E402
from beta2_parallel import entrenamiento_paralelo  # noqa: E402
from common import PARAMS_DEFECTO, generar_datos  # noqa: E402

M_DEFAULT = 32
P_VALUES_DEFAULT = [1, 2, 4, 8, 16, 32]
N_SAMPLES_DEFAULT = [5000, 10000, 20000]
FASES = ["t_split", "t_shm_setup", "t_pool_start", "t_map", "t_join",
         "t_reduccion", "t_shm_cleanup"]

RAW_FIELDS = [
    "timestamp_utc", "n_samples", "m", "p", "repetition", "mode", "order",
    "elapsed_seconds", "accuracy", "best_seed",
    "n_iter_min", "n_iter_max", "n_iter_mean", "n_hit_max_iter",
    "t_seed_sum", "t_seed_max", *FASES,
    "oversubscribed", "same_winner", "same_acc_all",
]

SUMMARY_FIELDS = [
    "n_samples", "m", "p", "complete_repetitions", "oversubscribed",
    "sequential_median_seconds", "sequential_mean_seconds",
    "sequential_stdev_seconds",
    "parallel_median_seconds", "parallel_mean_seconds",
    "parallel_stdev_seconds",
    "speedup_median", "speedup_stdev", "efficiency_median",
    "overhead_median_seconds",
    *[f"{f}_median" for f in FASES],
    "n_iter_min", "n_iter_max", "n_hit_max_iter_median",
    "accuracy_min", "accuracy_max", "all_equivalent",
]


# ------------------------------------------------------------- metadatos
def _cpus_disponibles():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("model name"):
                    return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "desconocido"


def _mem_total_gib():
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for linea in f:
                if linea.startswith("MemTotal"):
                    return round(int(linea.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    return None


def _git_commit():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def _save_metadata(path, m, p_values, n_values, repetitions):
    env_hilos = {v: os.environ.get(v) for v in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS")}
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")
             and k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION",
                       "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
                       "SLURMD_NODENAME", "SLURM_MEM_PER_NODE")}
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "campaign": "strong scaling: m fijo, p variable (Beta 0 vs Beta 2)",
        "m": m, "p_values": p_values, "n_samples_values": n_values,
        "repetitions": repetitions,
        "hyperparameters": {k: list(v) if isinstance(v, tuple) else v
                            for k, v in PARAMS_DEFECTO.items()},
        "dataset": {"n_features": 20, "n_informative": 15, "n_classes": 2,
                    "random_state": 42, "test_size": 0.2, "split_seed": 0},
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpus_available_to_process": _cpus_disponibles(),
        "mem_total_gib": _mem_total_gib(),
        "thread_env": env_hilos,
        "threadpool_info": threadpool_info(),
        "slurm": slurm,
        "multiprocessing_start_method_default": __import__(
            "multiprocessing").get_start_method(allow_none=False),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)


# ------------------------------------------------------------- mediciones
def _stats_iter(n_iter, max_iter):
    vals = [v for v in n_iter if v is not None]
    if not vals:
        return {"n_iter_min": "", "n_iter_max": "", "n_iter_mean": "",
                "n_hit_max_iter": ""}
    return {"n_iter_min": int(min(vals)), "n_iter_max": int(max(vals)),
            "n_iter_mean": f"{statistics.mean(vals):.3f}",
            "n_hit_max_iter": int(sum(v >= max_iter for v in vals))}


def _medir_secuencial(X, y, m):
    ini = time.perf_counter()
    _, acc, seed, tiempos = entrenamiento_secuencial(X, y, m)
    el = time.perf_counter() - ini
    n_iter = [t.get("n_iter") for t in tiempos]
    t_seed = [t["tiempo"] for t in tiempos]
    acc_all = [t["accuracy"] for t in tiempos]
    return dict(el=el, acc=acc, seed=seed, n_iter=n_iter, t_seed=t_seed,
                acc_all=acc_all)


def _medir_paralelo(X, y, m, p):
    ini = time.perf_counter()
    r = entrenamiento_paralelo(X, y, m, p)
    el = time.perf_counter() - ini
    return dict(el=el, acc=r["mejor_accuracy"], seed=r["mejor_seed"],
                n_iter=list(r["n_iter"]), t_seed=list(r["t_seed"]),
                acc_all=list(r["acc"]), fases=r["tiempos"])


def _fila_base(n, m, p, rep, modo, orden):
    f = {k: "" for k in RAW_FIELDS}
    f.update(timestamp_utc=datetime.now(timezone.utc).isoformat(),
             n_samples=n, m=m, p=p, repetition=rep, mode=modo, order=orden)
    return f


def _append_rows(path, rows):
    existe = path.exists()
    with path.open("a", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=RAW_FIELDS)
        if not existe:
            w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------- resumen
def _fl(x):
    return float(x)


def _sd(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


def _write_summary(raw_path, summary_path):
    with raw_path.open(newline="", encoding="utf-8") as s:
        filas = list(csv.DictReader(s))
    seq, par = {}, {}
    for r in filas:
        n, rep = int(r["n_samples"]), int(r["repetition"])
        if r["mode"] == "sequential":
            seq[(n, rep)] = r
        else:
            par[(n, int(r["p"]), rep)] = r

    grupos = {}
    for (n, p, rep), rp in par.items():
        rs = seq.get((n, rep))
        if rs is None:
            continue
        g = grupos.setdefault((n, p), {"ts": [], "tp": [], "sp": [], "ov": [],
                                       "acc": [], "eq": [], "m": rp["m"],
                                       "over": rp["oversubscribed"],
                                       "fases": {f: [] for f in FASES},
                                       "ni": [], "hit": []})
        ts, tp = _fl(rs["elapsed_seconds"]), _fl(rp["elapsed_seconds"])
        g["ts"].append(ts)
        g["tp"].append(tp)
        g["sp"].append(ts / tp)
        g["ov"].append(p * tp - ts)
        g["acc"].append(_fl(rp["accuracy"]))
        g["eq"].append(rp["same_winner"] == "1" and rp["same_acc_all"] == "1")
        for f in FASES:
            if rp[f] != "":
                g["fases"][f].append(_fl(rp[f]))
        if rp["n_iter_min"] != "":
            g["ni"].append((int(rp["n_iter_min"]), int(rp["n_iter_max"])))
            g["hit"].append(int(rp["n_hit_max_iter"]))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for (n, p), g in sorted(grupos.items()):
            fila = {
                "n_samples": n, "m": g["m"], "p": p,
                "complete_repetitions": len(g["sp"]),
                "oversubscribed": g["over"],
                "sequential_median_seconds": statistics.median(g["ts"]),
                "sequential_mean_seconds": statistics.mean(g["ts"]),
                "sequential_stdev_seconds": _sd(g["ts"]),
                "parallel_median_seconds": statistics.median(g["tp"]),
                "parallel_mean_seconds": statistics.mean(g["tp"]),
                "parallel_stdev_seconds": _sd(g["tp"]),
                "speedup_median": statistics.median(g["sp"]),
                "speedup_stdev": _sd(g["sp"]),
                "efficiency_median": statistics.median(g["sp"]) / p,
                "overhead_median_seconds": statistics.median(g["ov"]),
                "accuracy_min": min(g["acc"]), "accuracy_max": max(g["acc"]),
                "all_equivalent": all(g["eq"]),
            }
            for f in FASES:
                v = g["fases"][f]
                fila[f"{f}_median"] = statistics.median(v) if v else ""
            if g["ni"]:
                fila["n_iter_min"] = min(a for a, _ in g["ni"])
                fila["n_iter_max"] = max(b for _, b in g["ni"])
                fila["n_hit_max_iter_median"] = statistics.median(g["hit"])
            w.writerow(fila)


# ------------------------------------------------------------- campana
def _validar(m, p_values, n_values, repetitions):
    if m < 1:
        raise ValueError("m debe ser >= 1")
    if repetitions < 1:
        raise ValueError("Las repeticiones deben ser al menos 1")
    if not p_values or any(p < 1 for p in p_values):
        raise ValueError("Los valores de p deben ser >= 1")
    if not n_values or any(n < 1 for n in n_values):
        raise ValueError("Los valores de n_samples deben ser >= 1")
    no_div = [p for p in p_values if m % p != 0]
    if no_div:
        print(f"AVISO: p={no_div} no divide a m={m}; habra desbalance "
              f"por el techo ceil(m/p).")


def run_experiments(m, p_values, n_values, repetitions, raw_path,
                    summary_path, overwrite=False, warmup=True):
    _validar(m, p_values, n_values, repetitions)
    raw_path, summary_path = Path(raw_path), Path(summary_path)
    meta_path = raw_path.with_suffix(".metadata.json")
    if overwrite:
        for pth in (raw_path, summary_path, meta_path):
            pth.unlink(missing_ok=True)
    elif any(pth.exists() for pth in (raw_path, summary_path, meta_path)):
        raise FileExistsError(
            "Los archivos de salida ya existen. Usa --overwrite.")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    _save_metadata(meta_path, m, p_values, n_values, repetitions)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cpus = _cpus_disponibles()
    max_iter = PARAMS_DEFECTO["max_iter"]
    if max(p_values) > cpus:
        print(f"AVISO: p maximo={max(p_values)} > CPUs disponibles={cpus}: "
              f"esos puntos sobreasignan procesos.")

    with threadpool_limits(limits=1):
        if warmup:  # calienta imports y el camino del Pool
            Xw, yw = generar_datos(500)
            entrenamiento_secuencial(Xw, yw, 1)
            entrenamiento_paralelo(Xw, yw, 2, 2)

        for n in n_values:
            X, y = generar_datos(n)  # fuera de la region cronometrada
            for rep in range(1, repetitions + 1):
                seq_primero = rep % 2 == 1
                orden = "seq_first" if seq_primero else "seq_last"
                rs, rps = None, {}
                if seq_primero:
                    rs = _medir_secuencial(X, y, m)
                for p in p_values:
                    rps[p] = _medir_paralelo(X, y, m, p)
                if not seq_primero:
                    rs = _medir_secuencial(X, y, m)

                filas = []
                f = _fila_base(n, m, 1, rep, "sequential", orden)
                f.update(elapsed_seconds=f"{rs['el']:.9f}",
                         accuracy=f"{rs['acc']:.9f}", best_seed=rs["seed"],
                         t_seed_sum=f"{sum(rs['t_seed']):.6f}",
                         t_seed_max=f"{max(rs['t_seed']):.6f}",
                         **_stats_iter(rs["n_iter"], max_iter))
                filas.append(f)
                for p, rp in rps.items():
                    mismo_ganador = (rp["seed"] == rs["seed"]
                                     and rp["acc"] == rs["acc"])
                    mismas_acc = rp["acc_all"] == rs["acc_all"]
                    if not (mismo_ganador and mismas_acc):
                        print(f"  !! n={n} p={p} rep={rep}: el paralelo NO "
                              f"coincide con el secuencial")
                    f = _fila_base(n, m, p, rep, "parallel", orden)
                    f.update(
                        elapsed_seconds=f"{rp['el']:.9f}",
                        accuracy=f"{rp['acc']:.9f}", best_seed=rp["seed"],
                        t_seed_sum=f"{sum(rp['t_seed']):.6f}",
                        t_seed_max=f"{max(rp['t_seed']):.6f}",
                        oversubscribed=int(p > cpus),
                        same_winner=int(mismo_ganador),
                        same_acc_all=int(mismas_acc),
                        **_stats_iter(rp["n_iter"], max_iter))
                    fases = rp["fases"]
                    for k in FASES:
                        f[k] = f"{fases[k[2:]]:.9f}"
                    filas.append(f)
                _append_rows(raw_path, filas)
                _write_summary(raw_path, summary_path)

                print(f"n={n:>6} rep={rep}/{repetitions} ({orden}) "
                      f"T_seq={rs['el']:.3f}s")
                for p, rp in rps.items():
                    print(f"    p={p:>2} T_par={rp['el']:.3f}s "
                          f"S={rs['el'] / rp['el']:.2f} "
                          f"E={rs['el'] / rp['el'] / p:.2f}")

    print("\nCampana finalizada:")
    print(f"  Raw:      {raw_path}")
    print(f"  Resumen:  {summary_path}")
    print(f"  Metadata: {meta_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Strong scaling: m fijo, p variable (Beta 0 vs Beta 2).")
    ap.add_argument("--m", type=int, default=M_DEFAULT,
                    help="numero de semillas (fijo en toda la campana)")
    ap.add_argument("--p", type=int, nargs="+", default=P_VALUES_DEFAULT)
    ap.add_argument("--n_samples", type=int, nargs="+",
                    default=N_SAMPLES_DEFAULT)
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--raw_out", type=Path,
                    default=ROOT / "results/raw/campana_final_m32.csv")
    ap.add_argument("--summary_out", type=Path,
                    default=ROOT / "results/processed/resumen_campana_final_m32.csv")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sin_warmup", action="store_true")
    a = ap.parse_args()
    run_experiments(a.m, a.p, a.n_samples, a.repeticiones, a.raw_out,
                    a.summary_out, overwrite=a.overwrite,
                    warmup=not a.sin_warmup)


if __name__ == "__main__":
    main()