# Derivación PRAM — paralelismo entre semillas

## 1. Notación

Sea:

- `m = |S|`: número de semillas/modelos independientes.
- `p`: número de procesadores/workers.
- `C = Θ(E*n*d*h)`: costo de entrenar y evaluar un modelo según el baseline del enunciado.

El análisis considera como unidad paralela principal el entrenamiento completo de un modelo. En esta beta no se paraleliza internamente el forward/backpropagation de un MLP.

---

## 2. Tiempo secuencial para el workload completo

Entrenar `m` modelos uno tras otro cuesta:

```text
T_seq(m,n,d,h,E) = Θ(m*C)
                   = Θ(m*E*n*d*h)
```

La comparación final de `m` accuracies añade `Θ(m)`, dominado por el entrenamiento cuando `C` es grande.

---

## 3. Trabajo W

El trabajo total del DAG paralelo contiene:

- `m` entrenamientos, cada uno de costo `C`;
- una reducción de máximo con `m-1` comparaciones.

Por tanto:

```text
W = Θ(m*C + m)
  = Θ(m*E*n*d*h)
```

Esto coincide asintóticamente con el trabajo del algoritmo secuencial para el mismo conjunto de `m` semillas, por lo que el diseño es work-optimal respecto de este baseline.

---

## 4. Profundidad D (span)

Con suficientes procesadores, los `m` entrenamientos independientes pueden ejecutarse al mismo tiempo. Cada entrenamiento sigue siendo una cadena de costo `C` en el nivel de abstracción de esta beta.

La reducción en árbol requiere `Θ(log m)` niveles.

Entonces:

```text
D = Θ(C + log m)
  = Θ(E*n*d*h + log m)
```

---

## 5. Paralelismo disponible

```text
Π = W / D
  = Θ( m*C / (C + log m) )
```

Cuando `C >> log m`:

```text
Π ≈ Θ(m)
```

Interpretación: el máximo paralelismo útil de esta estrategia externa está acotado esencialmente por la cantidad de modelos/semillas independientes. Agregar muchos más procesadores que semillas no reduce el costo secuencial interno de entrenar un modelo.

---

## 6. Tiempo con p procesadores

Asignando las semillas entre `p` workers, cada worker ejecuta como máximo `ceil(m/p)` entrenamientos.

Un modelo de tiempo más informativo que la cota genérica de Brent es:

```text
T_p = Θ( ceil(m/p)*C + log m )
```

para la versión teórica con reducción en árbol.

La Ley de Brent también da:

```text
T_p = O(W/p + D)
    = O(m*C/p + C + log m)
```

La primera expresión representa mejor la planificación de `m` tareas independientes; la segunda es una cota general.

Esta expresión supone una reducción en árbol. Beta 1 implementa `max(resultados, ...)`
en el proceso maestro, de costo `Theta(m)`, y además tiene overhead de creación de
procesos y transferencia de datos. Para interpretar la implementación real debe
considerarse:

```text
T_Beta1(m,p) = T_setup(p,n) + ceil(m/p)*C + Theta(m) + T_comunicacion(p,n)
```

`T_setup` incluye la partición train/test y el ciclo de vida del pool; la transferencia
depende del método de inicio de procesos y del sistema operativo. Son costos
experimentales, no términos del PRAM abstracto.

---

## 7. Caso particular del enunciado: m = p

Si el número de semillas se fija igual al número de workers:

```text
m = p
```

entonces cada worker entrena un único modelo:

```text
T_parallel(p) = Θ(C + log p)
```

El mismo workload ejecutado secuencialmente cuesta:

```text
T_seq(p) = Θ(p*C)
```

Por tanto, el speedup teórico es:

```text
S(p) = T_seq(p) / T_parallel(p)
     ≈ p*C / (C + log p)
```

Y la eficiencia:

```text
Ef(p) = S(p)/p
      ≈ C / (C + log p)
```

Si `C >> log p`, entonces:

```text
S(p) ≈ p
Ef(p) ≈ 1
```

Esto representa el comportamiento ideal del modelo PRAM y no incluye overheads de procesos, serialización, memoria, scheduler ni bibliotecas numéricas.

En la Beta 1 concreta se reemplaza el costo logarítmico de reducción por `Theta(p)`.
La aproximación ideal `S(p) ≈ p*C/(C+log p)` requiere que el entrenamiento domine la
reducción y omite los overheads; no debe presentarse como predicción directa de los
tiempos observados.

### Alcance de C para scikit-learn

`C=Theta(E*n*d*h)` es el modelo de costo proporcionado por el enunciado. Para la
implementación, una época de una MLP densa de arquitectura fija cuesta
aproximadamente `Theta(n * sum_l(w_(l-1)*w_l))`, por lo que es lineal en `n` al mantener
fijas las capas. El `max_iter=200` predeterminado de `MLPClassifier` es un máximo y el
entrenamiento puede detenerse antes por convergencia; por ello el número efectivo de
épocas puede variar por semilla. En las conclusiones, `C` debe entenderse como una
abstracción uniforme del análisis, mientras los tiempos corresponden a la configuración
real y su criterio de parada.

---

## 8. Diferencia con la Beta 1 implementada

La Beta 1 usa `multiprocessing.Pool` para los entrenamientos, pero selecciona el mejor modelo con:

```python
max(resultados, key=lambda r: r[1])
```

Por ello, su reducción real en el maestro es `Θ(m)`, no `Θ(log m)`.

Además, los datos enviados a procesos pueden ser serializados/copied en lugar de residir en una memoria global CREW literal. En consecuencia, el tiempo medido experimentalmente tendrá un overhead adicional que debe discutirse al comparar teoría y experimento.

---

## 9. Condición para medir speedup correctamente

Cuando se usa `m=p`, no debe tomarse `T(p=1)` como tiempo secuencial de todos los demás puntos, porque el workload cambia con `p`.

Para cada `p` se necesita el par:

```text
T_seq(m=p)  : mismas p semillas, ejecutadas una tras otra
T_par(m=p,p): mismas p semillas, ejecutadas con p workers
```

Solo entonces:

```text
S(p) = T_seq(m=p) / T_par(m=p,p)
```

El benchmark comparable está en `benchmarks/run_experiments.py`. El generador de datos
se ejecuta una vez por `n` fuera de la región cronometrada; la división train/test se
incluye tanto en Beta 0 como en Beta 1. La campaña controla los hilos internos BLAS/OpenMP
a uno por proceso y registra mediciones por repetición, un resumen estadístico y
metadatos de hardware/software. Los speedups agregados se derivan de cocientes emparejados
por repetición; eficiencia es `S/p`, donde `p` son procesos configurados y puede superar
los procesadores lógicos disponibles.

La accuracy usada para escoger la semilla se calcula sobre el conjunto test en el
algoritmo vigente. Como el ganador se escoge con esa misma métrica, su accuracy no debe
interpretarse como estimación independiente de generalización; para ello se necesitaría
validación para la selección y un test reservado para la evaluación final.

compara la misma cantidad de trabajo.
