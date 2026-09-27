"""Create execution-time, speedup and efficiency plots from experiment summaries."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DEFAULT = ROOT / "results/processed/resumen.csv"
FIGURES_DEFAULT = ROOT / "report/figures"


def load_summary(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No hay resultados completos en {path}")
    return rows


def plot_results(summary_path=SUMMARY_DEFAULT, output_dir=FIGURES_DEFAULT):
    rows = load_summary(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_values = sorted({int(row["n_samples"]) for row in rows})
    p_values = sorted({int(row["p"]) for row in rows})
    colors = plt.get_cmap("viridis")
    color_by_n = {
        n: colors(index / max(1, len(n_values) - 1))
        for index, n in enumerate(n_values)
    }

    fig, axis = plt.subplots(figsize=(8, 5.5))
    for n in n_values:
        group = sorted(
            (row for row in rows if int(row["n_samples"]) == n),
            key=lambda row: int(row["p"]),
        )
        p = [int(row["p"]) for row in group]
        color = color_by_n[n]
        axis.errorbar(
            p,
            [float(row["sequential_median_seconds"]) for row in group],
            yerr=[float(row["sequential_stdev_seconds"]) for row in group],
            color=color,
            marker="o",
            linestyle="--",
            label=f"Secuencial, n={n}",
            capsize=3,
        )
        axis.errorbar(
            p,
            [float(row["parallel_median_seconds"]) for row in group],
            yerr=[float(row["parallel_stdev_seconds"]) for row in group],
            color=color,
            marker="s",
            linestyle="-",
            label=f"Paralelo, n={n}",
            capsize=3,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(p_values, labels=[str(p) for p in p_values])
    axis.set_xlabel("Procesos p (y semillas m=p)")
    axis.set_ylabel("Tiempo mediano por ejecución (s)")
    axis.set_title("Tiempo secuencial y paralelo por tamaño de dataset")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "tiempo.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5.5))
    for n in n_values:
        group = sorted(
            (row for row in rows if int(row["n_samples"]) == n),
            key=lambda row: int(row["p"]),
        )
        p = [int(row["p"]) for row in group]
        speedups = [float(row["speedup_median"]) for row in group]
        errors = [float(row["speedup_stdev"]) for row in group]
        axis.errorbar(
            p,
            speedups,
            yerr=errors,
            marker="o",
            capsize=3,
            color=color_by_n[n],
            label=f"n={n}",
        )
    axis.plot(p_values, p_values, color="black", linestyle=":", label="Ideal S=p")
    axis.set_xscale("log", base=2)
    axis.set_xticks(p_values, labels=[str(p) for p in p_values])
    axis.set_xlabel("Procesos p (y semillas m=p)")
    axis.set_ylabel("Speedup mediano")
    axis.set_title("Speedup respecto al workload secuencial equivalente")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "speedup.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5.5))
    for n in n_values:
        group = sorted(
            (row for row in rows if int(row["n_samples"]) == n),
            key=lambda row: int(row["p"]),
        )
        p = [int(row["p"]) for row in group]
        axis.errorbar(
            p,
            [100 * float(row["efficiency_median"]) for row in group],
            yerr=[
                100 * float(row["speedup_stdev"]) / int(row["p"])
                for row in group
            ],
            marker="o",
            capsize=3,
            color=color_by_n[n],
            label=f"n={n}",
        )
    axis.axhline(100, color="black", linestyle=":", label="Ideal (100%)")
    axis.set_xscale("log", base=2)
    axis.set_xticks(p_values, labels=[str(p) for p in p_values])
    axis.set_xlabel("Procesos p (y semillas m=p)")
    axis.set_ylabel("Eficiencia mediana (%)")
    axis.set_title("Eficiencia E_f = S/p")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eficiencia.png", dpi=180)
    plt.close(fig)

    print(f"Graficas guardadas en {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera las gráficas del benchmark desde resumen.csv."
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--output_dir", type=Path, default=FIGURES_DEFAULT)
    args = parser.parse_args()
    plot_results(args.summary, args.output_dir)


if __name__ == "__main__":
    main()
