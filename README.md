# PRAM MLP Training

Proyecto parcial del curso **Computación Paralela y Distribuida** (UTEC, 2026-II).

## Integrantes

- Mariel Tovar Tolentino
- Margiory Alvarado Chavez
- Noemi Huarino Anchillo

## Objetivo

Diseñar y evaluar una paralelización PRAM del entrenamiento de múltiples redes neuronales MLP inicializadas con semillas diferentes. La estrategia principal explota el paralelismo **entre entrenamientos independientes**: cada worker entrena un modelo con una semilla distinta y, al finalizar, se selecciona el modelo con mayor accuracy.

El costo base indicado por el enunciado para entrenar un modelo se representa como:

`C(n,d,h,E) = Θ(E*n*d*h)`

Para evitar mezclar tamaño del problema y recursos, en la documentación se distinguen:

- `m = |S|`: número de semillas/modelos que forman el workload.
- `p`: número de procesadores/workers utilizados para ejecutar el workload.

El enunciado usa como caso particular `S={0,...,p-1}`, es decir, `m=p`. Para el análisis formal se mantienen `m` y `p` separados y luego se especializa al caso `m=p`.

## Modelo PRAM elegido

Se usa **CREW-PRAM (Concurrent Read, Exclusive Write)** como modelo teórico:

- los workers pueden leer concurrentemente el mismo dataset e hiperparámetros;
- cada entrenamiento mantiene estado mutable propio (modelo, pesos, accuracy);
- cada worker escribe su resultado en un slot exclusivo;
- la selección teórica del mejor modelo se realiza mediante una reducción en árbol de profundidad `Θ(log m)`.

No se necesita CRCW porque el diseño evita escrituras simultáneas sobre una misma posición. EREW sería posible con replicación o planificación adicional de lecturas, pero CREW representa de forma más natural el patrón de acceso deseado.

> **Importante:** CREW es el modelo teórico. La Beta 1 usa `multiprocessing.Pool`; en Windows los arrays enviados a workers se serializan/copian, por lo que la implementación actual no constituye memoria compartida CREW literal. Ese costo forma parte del overhead experimental de la Beta 1.

## Estado actual

- ✅ Estructura del repositorio.
- ✅ Decisiones de diseño documentadas.
- ✅ Beta 1 paralela por semillas (`src/beta1_parallel.py`).
- ✅ Automatización inicial de tiempos (`benchmarks/run_benchmarkB1.py`).
- ✅ Derivación PRAM documentada (`docs/derivacion_pram.md`).
- ✅ Informe LaTeX integrado (`report/main.tex`).
- ✅ Beta 0 secuencial completa para m semillas.
- ✅ Campaña reproducible de Beta 0 vs Beta 1 (`benchmarks/run_experiments.py`).
- ✅ Resumen de speedup/eficiencia y gráficas desde los resultados (`plots/plot_results.py`).
- ⚠️ La Beta 1 conserva reducción secuencial en el maestro; no equivale a la reducción PRAM en árbol.
- ✅ Medidas completas para todos los p con n=5000 y n=10000 (3 repeticiones).
- ⏳ n=20000 y n=40000 pendientes; la ejecución parcial se conserva como archivo de auditoría.

## Estructura

```text
pram-mlp-training/
├── README.md
├── requirements.txt
├── benchmarks/
│   ├── run_benchmarkB1.py
│   └── run_experiments.py
├── docs/
│   ├── decisiones.md
│   └── derivacion_pram.md
├── plots/
│   └── plot_results.py
├── report/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── results/
│   ├── raw/       # mediciones individuales y metadatos
│   └── processed/ # resumen estadístico
└── src/
    ├── beta0_sequential.py
    ├── beta1_parallel.py
    ├── beta2_benchmark.py
    └── common.py
```

## Instalación

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

En Windows, use `.venv\Scripts\python.exe` en lugar de `.venv/bin/python`.
## Ejecución de Beta 0

La Beta 0 entrena secuencialmente las mismas `m` semillas que Beta 1
ejecuta en paralelo. En el experimento se utiliza `m=p` para comparar
workloads equivalentes.

Desde la raíz del repositorio:

```bash
.venv/bin/python src/beta0_sequential.py --p 4 --n_samples 5000
```

## Ejecución de Beta 1

Desde la raíz del repositorio:

```bash
.venv/bin/python src/beta1_parallel.py --p 4 --n_samples 5000
```

La Beta 1 actual interpreta `p` simultáneamente como número de workers y, siguiendo el caso particular del enunciado, número de semillas (`m=p`).

## Benchmark inicial Beta 1

```bash
.venv/bin/python benchmarks/run_benchmarkB1.py --p 1 2 4 8 --n_samples 5000 10000 --repeticiones 3
```

El benchmark de Beta 1 **mide tiempos paralelos**, pero todavía no debe usarse por sí solo para calcular speedup final.

### Regla para speedup

Si se usa el caso del enunciado `m=p`, cada punto cambia también el número de modelos. Por eso el tiempo secuencial de referencia para un valor dado de `p` debe entrenar **las mismas `m=p` semillas secuencialmente**:

`S(p) = T_seq(m=p) / T_parallel(m=p, p)`

No es correcto usar el tiempo de `p=1` (un solo modelo) como baseline de una ejecución con `p>1` modelos.

## Campaña experimental comparable

El benchmark comparable mide Beta 0 y Beta 1 para las mismas `p` semillas, datos y configuración. Por defecto recorre todos los valores del proyecto (`p={1,2,4,8,16,32}`, `n_samples={5000,10000,20000,40000}`) con tres repeticiones. Los datos generados se excluyen del cronómetro; división train/test, entrenamiento, creación/cierre del pool y selección del ganador sí se miden. Se alterna el orden secuencial/paralelo entre repeticiones.

```bash
.venv/bin/python benchmarks/run_experiments.py
.venv/bin/python plots/plot_results.py
```

Una campaña nueva completa usa por defecto `p={1,2,4,8,16,32}`, `n_samples={5000,10000,20000,40000}` y tres repeticiones. Guarda cada medición en `results/raw/experimentos.csv`, el resumen en `results/processed/resumen.csv` y el hardware/software e hiperparámetros en `results/raw/experimentos.metadata.json`. El resumen reporta mediana, promedio y desviación estándar muestral de tiempos, además de speedup por repetición emparejada y eficiencia mediana. Las gráficas se escriben en `report/figures/`.

Los resultados completos disponibles en esta revisión cubren todos los valores de p para `n_samples=5000` y `10000`: [experimentos_n5000_10000.csv](results/raw/experimentos_n5000_10000.csv), [resumen_n5000_10000.csv](results/processed/resumen_n5000_10000.csv) y [metadatos](results/raw/experimentos_n5000_10000.metadata.json). La campaña mayor se detuvo por costo de cómputo; sus mediciones parciales de `n=20000` están preservadas separadamente y no se usan en el resumen ni en las gráficas: [experimentos_interrumpidos.csv](results/raw/experimentos_interrumpidos.csv).

Para volver a generar las figuras a partir de las mediciones completas disponibles:

```bash
.venv/bin/python plots/plot_results.py \
  --summary results/processed/resumen_n5000_10000.csv
```

Para probar un subconjunto o reducir el costo inicial:

```bash
.venv/bin/python benchmarks/run_experiments.py --p 1 2 4 --n_samples 5000 10000 --repeticiones 2 \
  --raw_out results/raw/piloto.csv --summary_out results/processed/piloto.csv
.venv/bin/python plots/plot_results.py --summary results/processed/piloto.csv \
  --output_dir results/processed/figuras_piloto
```

Los archivos de salida existentes no se sobrescriben salvo que se indique `--overwrite`. Cada fila raw se vacía a disco inmediatamente, para preservar mediciones si la campaña se interrumpe. El paralelismo interno OpenMP/BLAS se fija a un hilo por proceso y esa configuración queda registrada. `p` es el número de procesos solicitado, no una garantía de disponer de igual cantidad de CPU físicas; se debe discutir el efecto si `p` supera los procesadores lógicos disponibles.

La accuracy registrada es la del modelo ganador, seleccionado usando la partición test según el algoritmo del curso. Por ello debe interpretarse como métrica de selección y no como estimación imparcial de generalización; una evaluación metodológicamente más sólida elegiría la semilla en validación y reservaría test para una única evaluación final. Las repeticiones conservan datos y semillas para medir ruido de ejecución, no incertidumbre estadística de accuracy.
