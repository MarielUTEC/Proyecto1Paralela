import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Rutas por defecto
ROOT = Path(__file__).resolve().parents[1]
SUMMARY_DEFAULT = ROOT / "results/processed/resumen.csv"
FIGURES_DEFAULT = ROOT / "report/figures"


def load_summary(path):
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"El archivo {path} está vacío.")
    return rows


def plot_results(summary_path=SUMMARY_DEFAULT, output_dir=FIGURES_DEFAULT):
    rows = load_summary(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extraer valores únicos de n y p
    n_values = sorted(list({int(row["n_samples"]) for row in rows}))
    p_values = sorted(list({int(row["p"]) for row in rows}))

    # 1. Gráfica de Tiempos
    plt.figure(figsize=(8, 5))
    for n in n_values:
        group = sorted([r for r in rows if int(r["n_samples"]) == n], key=lambda x: int(x["p"]))
        p = [int(r["p"]) for r in group]
        t_seq = [float(r["sequential_median_seconds"]) for r in group]
        t_par = [float(r["parallel_median_seconds"]) for r in group]

        plt.plot(p, t_seq, marker="o", linestyle="--", label=f"Secuencial (n={n})")
        plt.plot(p, t_par, marker="s", linestyle="-", label=f"Paralelo (n={n})")

    plt.xscale("log", base=2)
    plt.xticks(p_values, [str(p) for p in p_values])
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Tiempo mediano (s)")
    plt.title("Tiempo de Ejecución: Secuencial vs Paralelo")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tiempo.png")
    plt.close()

    # 2. Gráfica de Speedup
    plt.figure(figsize=(8, 5))
    for n in n_values:
        group = sorted([r for r in rows if int(r["n_samples"]) == n], key=lambda x: int(x["p"]))
        p = [int(r["p"]) for r in group]
        speedups = [float(r["speedup_median"]) for r in group]

        plt.plot(p, speedups, marker="o", label=f"n={n}")

    plt.plot(p_values, p_values, color="black", linestyle=":", label="Speedup Ideal (S=p)")
    plt.xscale("log", base=2)
    plt.xticks(p_values, [str(p) for p in p_values])
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Speedup")
    plt.title("Aceleración (Speedup)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "speedup.png")
    plt.close()

    # 3. Gráfica de Eficiencia
    plt.figure(figsize=(8, 5))
    for n in n_values:
        group = sorted([r for r in rows if int(r["n_samples"]) == n], key=lambda x: int(x["p"]))
        p = [int(r["p"]) for r in group]
        efficiency = [100 * float(r["efficiency_median"]) for r in group]

        plt.plot(p, efficiency, marker="o", label=f"n={n}")

    plt.axhline(100, color="black", linestyle=":", label="Ideal (100%)")
    plt.xscale("log", base=2)
    plt.xticks(p_values, [str(p) for p in p_values])
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Eficiencia (%)")
    plt.title("Eficiencia del Paralelismo")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eficiencia.png")
    plt.close()

    print(f"Gráficas generadas exitosamente en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Genera gráficas a partir del resumen de experimentos.")
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT)
    parser.add_argument("--output_dir", type=Path, default=FIGURES_DEFAULT)
    args = parser.parse_args()

    plot_results(args.summary, args.output_dir)


if __name__ == "__main__":
    main()