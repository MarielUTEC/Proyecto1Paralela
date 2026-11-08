# PRAM MLP Training

Proyecto parcial del curso **Computación Paralela y Distribuida** (UTEC, 2026-II).

## Integrantes

- Mariel Tovar Tolentino
- Margiory Alvarado Chavez
- Noemi Huarino Anchillo

## Objetivo

Implementar y comparar el entrenamiento de varios clasificadores MLP inicializados con semillas diferentes. Cada entrenamiento es independiente; se ejecuta una vez por semilla y luego se selecciona el modelo con mayor accuracy.

El modelo de costo del enunciado para un entrenamiento es:

```text
C(n,d,h,E) = Θ(E*n*d*h)
```

La documentación separa:

- `m`: cantidad de semillas/modelos que forman el workload.
- `p`: cantidad de procesos/workers disponibles.

La implementación final (**Beta 2**) paraleliza directamente sobre las semillas (`for s ∈ S pardo`) con `m` fijo y `p` configurable. El diseño CREW-PRAM y su análisis se explican en [docs/decisiones.md](docs/decisiones.md) y [docs/derivacion_pram.md](docs/derivacion_pram.md).

> **Modelo de memoria implementado:** procesos + `multiprocessing.shared_memory`. El dataset vive en un segmento compartido de solo lectura (lectura concurrente) y cada semilla escribe solo en su posición de los arreglos compartidos `Acc[s]`, `n_iter[s]`, `t_seed[s]` y `Mod[s]` (escritura exclusiva). Los workers siguen siendo procesos del sistema operativo, por lo que es una aproximación de CREW y no una PRAM literal. La selección final se hace por rondas (árbol binario) en el proceso maestro.

Las versiones anteriores se conservan como referencia: Beta 0 (baseline secuencial) y Beta 1 (`Pool` con `m=p`, datos enviados a cada tarea, `max` secuencial).

## Estado del código y los resultados

- Código compartido (datos, split, entrenamiento de una semilla, reducción): [src/common.py](src/common.py).
- Baseline secuencial para `m` semillas: [src/beta0_sequential.py](src/beta0_sequential.py).
- **Versión paralela final (memoria compartida, `m` y `p` independientes):** [src/beta2_parallel.py](src/beta2_parallel.py).
- Versión paralela anterior, solo evidencia histórica (`m=p`): [src/beta1_parallel.py](src/beta1_parallel.py).
- Pruebas de correctitud secuencial vs. paralelo: [tests/test_correctitud.py](tests/test_correctitud.py).
- Benchmark comparativo: [benchmarks/run_experiments.py](benchmarks/run_experiments.py) (aún por actualizar a Beta 2 con `m` fijo).
- Generación de gráficas desde el resumen CSV: [plots/plot_results.py](plots/plot_results.py).
- Campaña preliminar (`m=p`, Beta 0 vs. Beta 1) completa para `n_samples=5000` y `10000`.

## Estructura

```text
.
├── benchmarks/
│   ├── run_benchmarkB1.py       # benchmark previo: solo Beta 1
│   └── run_experiments.py       # comparación secuencial vs. paralelo
├── docs/
│   ├── decisiones.md
│   └── derivacion_pram.md
├── plots/
│   └── plot_results.py
├── report/
│   ├── main.tex
│   ├── references.bib
│   └── figures_actualizadas/    # gráficas de la última campaña
├── results/
│   ├── raw/                     # mediciones individuales y metadatos
│   └── processed/               # resúmenes estadísticos
├── src/
│   ├── common.py                # datos, split, entrenamiento, reducción
│   ├── beta0_sequential.py      # baseline secuencial
│   ├── beta1_parallel.py        # versión anterior (m=p)
│   └── beta2_parallel.py        # versión final (memoria compartida)
└── tests/
    └── test_correctitud.py
```

## Requisitos e instalación

Se requiere Python 3 y las dependencias declaradas en [requirements.txt](requirements.txt):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

En Windows, usa `.venv\Scripts\python.exe` en lugar de `.venv/bin/python`.

## Ejecutar las versiones

Desde la raíz del repositorio:

```bash
# Beta 0: entrena m semillas secuencialmente
.venv/bin/python src/beta0_sequential.py --m 32 --n_samples 5000

# Beta 2: entrena m semillas con p procesos y memoria compartida
.venv/bin/python src/beta2_parallel.py --m 32 --p 8 --n_samples 5000
```

`m` (número de semillas) y `p` (número de procesos) son independientes. Beta 2 imprime el tiempo total y el tiempo por fase: `split`, `shm_setup`, `pool_start`, `map`, `join`, `reduccion` y `shm_cleanup`.

Ambos programas generan los datos con `make_classification` (20 features, 15 informativas, clasificación binaria, `random_state=42`), usan una partición test del 20% con `split_seed=0` y asignan `random_state=seed` a cada MLP. La arquitectura e hiperparámetros por defecto son `hidden_layer_sizes=(10,10)`, `alpha=1e-4`, `learning_rate_init=1e-3` y `max_iter=200`. El solver no se especifica explícitamente; se usa el valor predeterminado de scikit-learn. Las bibliotecas numéricas se limitan a un hilo por proceso.

## Pruebas de correctitud

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -v
```

Comparan Beta 0 y Beta 2 con `m=32` y `p ∈ {1,2,4,8}`: misma semilla ganadora, misma accuracy y mismo `n_iter_` por semilla. También cubren la reducción con desempate por menor semilla, `p > m` y el modo `spawn`.

## Experimentos comparativos
> **Nota:** esta sección describe la campaña **preliminar** (`m=p`, Beta 0 vs. Beta 1). La campaña final con `m=32` fijo y Beta 2 se documenta cuando se ejecute.

[run_experiments.py](benchmarks/run_experiments.py) compara Beta 0 y Beta 1 con el mismo dataset, el mismo conjunto de `m=p` semillas y los mismos hiperparámetros. Los valores predeterminados son:

```text
p            = {1, 2, 4, 8, 16, 32}
n_samples    = {5000, 10000, 20000, 40000}
repeticiones = 3
```

Para cada `n`, el dataset se genera fuera de la región cronometrada. La medición incluye la partición train/test y los entrenamientos; en Beta 1 incluye también el ciclo de vida del pool y la selección del ganador. En cada repetición se ejecuta primero la versión secuencial y luego la paralela. `threadpoolctl` limita a un hilo las bibliotecas numéricas durante la medición.

Ejecutar la matriz predeterminada:

```bash
.venv/bin/python benchmarks/run_experiments.py
```

El programa escribe:

- `results/raw/experimentos.csv`: ejecuciones individuales con tiempos, accuracy y semilla ganadora.
- `results/processed/resumen.csv`: medianas, medias, desviaciones estándar muestrales, speedup emparejado y eficiencia.
- `results/raw/experimentos.metadata.json`: timestamp, plataforma, cantidad de CPU lógicas, versiones de Python/NumPy/scikit-learn y parámetros de la campaña.

Si los archivos de salida ya existen, el programa se detiene para evitar sobrescribirlos. Usa `--overwrite` solo si se desea reemplazar esa campaña.

Para ejecutar un subconjunto en rutas propias:

```bash
.venv/bin/python benchmarks/run_experiments.py \
  --p 1 2 4 8 16 32 \
  --n_samples 5000 10000 \
  --repeticiones 3 \
  --raw_out results/raw/experimentos_n5000_10000.csv \
  --summary_out results/processed/resumen_experimentos.csv
```

El archivo JSON de metadatos se genera junto al CSV raw con el mismo nombre base y sufijo `.metadata.json`.

### Métricas y significado de `p`

Cada repetición se compara con el mismo workload:

```text
T_seq(n,p) = tiempo secuencial de las semillas 0,...,p-1
T_par(n,p) = tiempo paralelo de esas mismas p semillas
S(n,p)     = T_seq(n,p) / T_par(n,p)
E_f(n,p)   = S(n,p) / p
```

El resumen reporta la mediana de los speedups calculados por repetición emparejada. Como `m=p`, al aumentar `p` también aumenta el número de modelos. Esta campaña **no es strong scaling de un workload fijo**. `p` es el número de procesos solicitado y puede exceder las CPU disponibles.

## Resultados disponibles

> Resultados de la campaña preliminar (`m=p`), obtenidos en un equipo local de 8 CPU lógicas. No constituyen strong scaling.

La campaña completa disponible cubre `n_samples=5000` y `10000`, todos los valores de `p` y tres repeticiones por configuración:

- Datos sin procesar: [experimentos_5000y10000.csv](results/raw/experimentos_5000y10000.csv).
- Resumen utilizado para las gráficas: [resumen_experimentos.csv](results/processed/resumen_experimentos.csv).
- Metadatos de la campaña: [experimentos_5000y10000.metadata.json](results/raw/experimentos_5000y10000.metadata.json).
- Datos parciales de un intento anterior, archivados y excluidos del resumen: [experimentos_interrumpidos.csv](results/raw/experimentos_interrumpidos.csv).

Los speedups medianos de la campaña vigente son:

| `n_samples` | p=1 | p=2 | p=4 | p=8 | p=16 | p=32 |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 0.99 | 1.88 | **3.38** | 3.34 | 2.99 | 2.84 |
| 10000 | 0.97 | 1.80 | **3.05** | 2.70 | 2.83 | 2.85 |

El máximo mediano se observa en `p=4` para ambos tamaños. La eficiencia en ese punto es aproximadamente 84.5% para `n=5000` y 76.3% para `n=10000`, para `p=32` cae a cerca de 8.9%. El equipo en el que se ejecutó el experimentos tiene 8 CPU, por lo que `p=16` y `p=32` sobreasignan procesos. Estos resultados no permiten sacar conclusiones para `n=20000` ni `n=40000`.

La accuracy usada para elegir la semilla ganadora se mide sobre el conjunto llamado test. Por tanto, es una métrica de selección, no una evaluación imparcial de generalización. Las repeticiones mantienen fijos datos y semillas para medir variación de tiempo de ejecución, no incertidumbre de accuracy.

## Generar las gráficas

`run_experiments.py` genera CSV y metadatos, pero **no genera imágenes**. Para producir las figuras desde un resumen se pone el siguiente comando:

```bash
.venv/bin/python plots/plot_results.py \
  --summary results/processed/resumen_experimentos.csv \
  --output_dir report/figures_actualizadas
```

Se generan `tiempo.png`, `speedup.png` y `eficiencia.png` dentro de la carpeta de salida. Las figuras de la última campaña están en [report/figures_actualizadas](report/figures_actualizadas).

Para usar las rutas predeterminadas, que esperan `results/processed/resumen.csv` y escriben en `report/figures/`:

```bash
.venv/bin/python plots/plot_results.py
```