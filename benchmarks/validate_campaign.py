"""Audita una campaña final a partir de CSV crudo, resumen y metadatos."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def validate(raw_path, summary_path, metadata_path):
    raw = _rows(raw_path)
    summary = _rows(summary_path)
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    n_values = list(meta["n_samples_values"])
    p_values = list(meta["p_values"])
    repetitions = int(meta["repetitions"])
    m = int(meta["m"])
    assert len(raw) == len(n_values) * repetitions * (1 + len(p_values))
    assert len(summary) == len(n_values) * len(p_values)
    assert len({(r["n_samples"], r["repetition"], r["mode"], r["p"])
                for r in raw}) == len(raw), "Filas duplicadas"
    assert all(int(r["m"]) == m for r in raw)
    assert all(int(pool["num_threads"]) == 1
               for pool in meta["threadpool_info"]), "Hilos internos no limitados"

    groups = {}
    for r in raw:
        groups.setdefault((int(r["n_samples"]), int(r["repetition"])),
                          {})[(r["mode"], int(r["p"]))] = r
    for n in n_values:
        for rep in range(1, repetitions + 1):
            group = groups[(n, rep)]
            assert set(group) == {("sequential", 1)} | {
                ("parallel", p) for p in p_values}
            seq = group[("sequential", 1)]
            seq_iter = json.loads(seq["n_iter_by_seed_json"])
            assert len(seq_iter) == m
            for p in p_values:
                par = group[("parallel", p)]
                assert (par["same_winner"], par["same_acc_all"],
                        par["same_n_iter_all"]) == ("1", "1", "1")
                assert json.loads(par["n_iter_by_seed_json"]) == seq_iter
                assert int(par["best_seed"]) == int(seq["best_seed"])
                assert math.isclose(float(par["accuracy"]),
                                    float(seq["accuracy"]), abs_tol=1e-9)

    anomalous = []
    for r in summary:
        n, p = int(r["n_samples"]), int(r["p"])
        assert int(r["complete_repetitions"]) == repetitions
        assert r["all_equivalent"] == "True"
        seq_times = [float(groups[(n, rep)][("sequential", 1)]
                           ["elapsed_seconds"])
                     for rep in range(1, repetitions + 1)]
        par_times = [float(groups[(n, rep)][("parallel", p)]
                           ["elapsed_seconds"])
                     for rep in range(1, repetitions + 1)]
        speedups = [ts / tp for ts, tp in zip(seq_times, par_times)]
        assert math.isclose(float(r["speedup_median"]),
                            statistics.median(speedups), rel_tol=1e-12)
        assert math.isclose(float(r["efficiency_median"]),
                            statistics.median(speedups) / p, rel_tol=1e-12)
        assert math.isclose(float(r["parallel_median_seconds"]),
                            statistics.median(par_times), rel_tol=1e-12)
        median = statistics.median(par_times)
        if max(abs(t - median) / median for t in par_times) > 0.2:
            anomalous.append((n, p, par_times))

    print(f"OK: {len(raw)} filas crudas, {len(summary)} puntos de resumen, "
          f"{repetitions} repeticiones por punto")
    print(f"Commit medido: {meta['git_commit']}")
    print(f"Plataforma: {meta['cpu_model']} / {meta['platform']}")
    print(f"Puntos con desviación temporal >20% de su mediana: {anomalous}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    validate(args.raw, args.summary, args.metadata)


if __name__ == "__main__":
    main()
