# Derivación PRAM - paralelismo entre semillas

## 1. Notación y alcance

- `m = |S|`: número de semillas/modelos del workload.
- `p`: número de procesadores en el modelo PRAM o de procesos worker en Beta 2.
- `C = Θ(E*n*d*h)`: costo de entrenar y evaluar una semilla, según el modelo del
  enunciado.

La unidad paralela principal es el entrenamiento completo de una semilla. No se
paraleliza internamente el forward/backpropagation del MLP. Se supone inicialmente
que los costos de entrenamiento son comparables; en scikit-learn el número
efectivo de iteraciones puede variar por semilla.

El análisis separa el costo ideal del algoritmo del tiempo de la implementación
Python. En Beta 2, `m` y `p` son independientes; la campaña vigente mantiene `m`
fijo y varía `p`.

## 2. Tiempo secuencial del workload

Entrenar las `m` semillas una tras otra tiene costo:

```text
T_seq(m,n,d,h,E) = Θ(m*C)
                  = Θ(m*E*n*d*h)
```

La selección del máximo accuracy añade `Θ(m)`, dominado asintóticamente por el
entrenamiento cuando `C` es grande. Este es el workload de referencia usado por
Beta 0 para comparar las mismas `m` semillas de Beta 2.

## 3. Trabajo W

El DAG de tareas tiene `m` entrenamientos independientes de costo `C` y una
reducción de máximo con `m-1` comparaciones:

```text
W = Θ(m*C + m)
  = Θ(m*E*n*d*h)
```

El trabajo es asintóticamente igual al del baseline secuencial para las mismas
semillas; por ello el diseño es work-optimal respecto de este baseline.

## 4. Profundidad D (span) PRAM

Con suficientes procesadores, los `m` entrenamientos se ejecutan en paralelo. Cada
uno mantiene su costo secuencial `C`. Una reducción binaria paralela requiere
`Θ(log m)` niveles:

```text
D = Θ(C + log m)
  = Θ(E*n*d*h + log m)
```

Esta profundidad describe la reducción paralela en el modelo PRAM. La función
`reduce_max_arbol` de Beta 2 realiza las comparaciones de esas rondas en bucles
secuenciales del proceso maestro; por tanto, el código implementado no alcanza
span `Θ(log m)` en esa fase.

## 5. Paralelismo disponible

```text
Π = W / D
  = Θ(m*C / (C + log m))
```

Cuando `C >> log m`, `Π ≈ Θ(m)`: el paralelismo útil de esta estrategia externa
está limitado esencialmente por las semillas independientes. Más workers que
semillas no reducen el costo secuencial interno de entrenar un MLP.

## 6. Tiempo ideal con p procesadores

Para tareas de costo aproximadamente uniforme, una planificación balanceada de
las `m` semillas da el modelo:

```text
T_p,ideal = Θ(ceil(m/p)*C + log m)
```

La expresión incorpora el costo de entrenamientos asignados al worker más cargado
y la profundidad de una reducción paralela ideal. Es una aproximación teórica: no
incluye granularidad de planificación, creación de workers, comunicación ni
variación del número de iteraciones entre modelos.

La ley de Brent da la cota general:

```text
T_p = O(W/p + D)
    = O(m*C/p + C + log m)
```

En Beta 2, la reducción de máximo se ejecuta en serie. Aun omitiendo los demás
overheads, el término de reducción real es `Θ(m)` en lugar de `Θ(log m)`.

## 7. Tiempo de Beta 2 implementada

Beta 2 (`src/beta2_parallel.py`) divide los datos, copia los arreglos a segmentos
de `multiprocessing.shared_memory`, inicia un `Pool` de `p` procesos y distribuye
las `m` semillas con `chunksize=1`. Los workers leen los datos compartidos y cada
semilla escribe sus resultados en slots exclusivos. El maestro copia las
accuracies y ejecuta la reducción por rondas secuencialmente.

Un resumen útil del tiempo medido es:

```text
T_Beta2 = T_split + T_shm_setup + T_pool_start
        + T_map + T_join + T_reduccion + T_shm_cleanup
```

De forma aproximada, la parte de entrenamiento puede comportarse como
`ceil(m/p)*C` si los trabajos son similares, mientras que la reducción
implementada cuesta `Θ(m)`. Los términos de split, memoria compartida, pool,
planificación y cleanup dependen del sistema y del runtime; no se derivan de la
cota PRAM. El programa reporta estas fases por separado, además del tiempo total.

La memoria compartida reduce la necesidad de serializar/copiar los arreglos de
datos para cada tarea, pero no elimina el costo de crear los segmentos, iniciar
procesos ni gestionar el pool. Tampoco convierte a `multiprocessing` en una PRAM
literal.

## 8. Caso especial m = p

Si el workload se define con `m=p`, hay una semilla por worker en el caso ideal:

```text
T_parallel,ideal(p) = Θ(C + log p)
T_seq(p)            = Θ(p*C)
S_ideal(p)          = T_seq(p) / T_parallel,ideal(p)
                    ≈ p*C / (C + log p)
Ef_ideal(p)         = S_ideal(p) / p
                    ≈ C / (C + log p)
```

Cuando `C >> log p`, el modelo ideal da `S_ideal(p) ≈ p` y `Ef_ideal(p) ≈ 1`.
Beta 2 no implementa una reducción paralela, así que, incluso sin setup ni
comunicación, su reducción serial añade `Θ(p)` en este caso. Las fórmulas ideales
no son predicciones directas de sus tiempos observados.

Este caso `m=p` describe la configuración histórica de Beta 1, no la campaña
actual de Beta 2. En la campaña actual, `m=32` es fijo y se comparan los tiempos
Beta 0/Beta 2 para `p` variable:

```text
S(n,m,p)  = T_seq(n,m) / T_par(n,m,p)
Ef(n,m,p) = S(n,m,p) / p
```

Así se compara el mismo workload en cada punto de strong scaling. No se debe usar
un tiempo secuencial con un número distinto de semillas como baseline.

## 9. Relación entre C y scikit-learn

`C=Θ(E*n*d*h)` es el modelo de costo establecido por el enunciado. Para una MLP
densa de arquitectura fija, el costo de una época es aproximadamente
`Θ(n * sum_l(w_(l-1)*w_l))`, lineal en `n` al mantener fijas las capas. En
`MLPClassifier`, `max_iter=200` es un límite y el entrenamiento puede detenerse
antes por convergencia; además, el número efectivo de iteraciones puede variar
entre semillas.

En consecuencia, `C` es una abstracción uniforme para el análisis. Los tiempos
reales corresponden a los hiperparámetros y al criterio de parada de la versión
de scikit-learn usada en la campaña. La accuracy de selección se calcula sobre el
conjunto test; no debe interpretarse como evaluación independiente de
generalización.
