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
- ✅ Validación inicial contra Beta 1 realizada con p=4 y n_samples=5000.
- ⏳ Reducción experimental en árbol / Beta 2.
- ⏳ Resultados finales, speedup y eficiencia.
- ⏳ Fuentes, reflexión sobre referencias y uso de IA.

## Estructura

```text
pram-mlp-training/
├── README.md
├── requirements.txt
├── benchmarks/
│   └── run_benchmarkB1.py
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
│   ├── raw/
│   └── processed/
└── src/
    ├── beta0_sequential.py
    ├── beta1_parallel.py
    ├── beta2_benchmark.py
    └── common.py
```

## Instalación

```bash
python -m pip install -r requirements.txt
```
## Ejecución de Beta 0

La Beta 0 entrena secuencialmente las mismas `m` semillas que Beta 1
ejecuta en paralelo. En el experimento se utiliza `m=p` para comparar
workloads equivalentes.

Desde la raíz del repositorio:

```bash
python src\beta0_sequential.py --p 4 --n_samples 5000
```

## Ejecución de Beta 1

Desde la raíz del repositorio:

```bash
python src/beta1_parallel.py --p 4 --n_samples 5000
```

La Beta 1 actual interpreta `p` simultáneamente como número de workers y, siguiendo el caso particular del enunciado, número de semillas (`m=p`).

## Benchmark inicial Beta 1

```bash
python benchmarks/run_benchmarkB1.py --p 1 2 4 8 --n_samples 5000 10000 --repeticiones 3
```

El benchmark de Beta 1 **mide tiempos paralelos**, pero todavía no debe usarse por sí solo para calcular speedup final.

### Regla para speedup

Si se usa el caso del enunciado `m=p`, cada punto cambia también el número de modelos. Por eso el tiempo secuencial de referencia para un valor dado de `p` debe entrenar **las mismas `m=p` semillas secuencialmente**:

`S(p) = T_seq(m=p) / T_parallel(m=p, p)`

No es correcto usar el tiempo de `p=1` (un solo modelo) como baseline de una ejecución con `p>1` modelos.

## Consideraciones para los experimentos finales

Antes de obtener las mediciones definitivas se debe:

1. fijar la arquitectura e hiperparámetros y mantenerlos constantes;
2. controlar los hilos internos de BLAS/OpenMP para evitar oversubscription dentro de cada proceso;
3. registrar hardware, versiones de Python/scikit-learn y sistema operativo;
4. ejecutar varias repeticiones por configuración;
5. comparar siempre workload secuencial y paralelo equivalentes.
