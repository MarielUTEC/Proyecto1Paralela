# Decisiones del proyecto - PRAM aplicado a múltiples MLP

## 1. Baseline y notación

El modelo de costo indicado por el enunciado para entrenar un MLP es:

```text
C(n,d,h,E) = Θ(E * n * d * h)
```

Donde `n` es el número de muestras, `d` el número de features, `h` el ancho
considerado por el modelo de costo y `E` las épocas/iteraciones. Para no confundir
el trabajo con los recursos, se usan:

- `m = |S|`: cantidad de semillas/modelos que forman el workload.
- `p`: cantidad de procesos worker solicitados.

El pseudocódigo puede tomar `m=p`, pero la Beta 2 y la campaña vigente permiten
variarlos independientemente. La complejidad `C` es una abstracción del enunciado,
no una descripción exacta del entrenamiento interno de scikit-learn.

## 2. Estrategia paralela

Se paraleliza el ciclo sobre las semillas `S={0,...,m-1}`. Para cada semilla `s`,
Beta 2:

1. entrena un MLP independiente con el mismo conjunto de entrenamiento;
2. calcula la accuracy sobre el conjunto test;
3. escribe la accuracy, `n_iter_`, el tiempo y los parámetros del modelo en la
   posición correspondiente a `s`.

Los entrenamientos no dependen entre sí. Al finalizar se selecciona la semilla con
mayor accuracy; en caso de empate gana la semilla menor, igual que en el baseline
secuencial.

`p` workers distribuyen las `m` tareas (una semilla por tarea, con `chunksize=1`).
En el caso ideal de costos uniformes, cada worker ejecuta aproximadamente
`ceil(m/p)` entrenamientos. Si `p>m`, algunos workers quedan ociosos.

## 3. Modelo PRAM y memoria de Beta 2

El análisis usa CREW-PRAM:

- **Concurrent Read (CR):** las semillas leen los mismos `X_train`, `y_train`,
  `X_test` y `y_test`.
- **Exclusive Write (EW):** cada semilla `s` escribe en sus propios slots
  `Acc[s]`, `n_iter[s]`, `t_seed[s]` y en la fila `Mod[s,:]`.
- **Reducción:** en el algoritmo PRAM abstracto, cada nivel compara pares
  disjuntos y escribe un único ganador por pareja.

Beta 2 aproxima este patrón con procesos del sistema operativo y
`multiprocessing.shared_memory`: el proceso maestro copia los cuatro arreglos de
datos una vez a segmentos compartidos; cada worker se conecta a esos segmentos y
marca los datos como solo lectura. Los arreglos de salida también son compartidos,
y cada tarea escribe únicamente la posición de su semilla. Los workers limitan a un
hilo los kernels numéricos mediante `threadpoolctl`.

Esto no es una PRAM literal. Creación de procesos, planificación, sincronización,
acceso a memoria compartida y bibliotecas numéricas introducen costos reales que el
modelo no representa. No se usa CRCW porque no hay escrituras concurrentes sobre
una misma posición. CREW describe el patrón de acceso compartido; no implica que la
implementación de Python tenga los tiempos ideales del modelo.

## 4. Reducción del ganador

El algoritmo de `reduce_max_arbol` organiza las comparaciones por rondas binarias.
En un CREW-PRAM, la reducción tiene trabajo `Θ(m)` y profundidad `Θ(log m)`.
Conserva el candidato sin pareja cuando una ronda tiene un número impar de
candidatos y resuelve empates a favor de la menor semilla.

En Beta 2 esa misma estructura de rondas se ejecuta con bucles secuenciales en el
proceso maestro, después de copiar las accuracies desde la memoria compartida.
Por ello, la implementación conserva `ceil(log2 m)` rondas, pero el tiempo de la
reducción ejecutada es `Θ(m)`, no `Θ(log m)`. La profundidad logarítmica corresponde
al diseño PRAM ideal, no a una reducción paralela implementada en Python.

## 5. Implementación vigente: Beta 2

La implementación final está en `src/beta2_parallel.py`; el entrenamiento por
semilla, el split, la generación de datos y la reducción se comparten con el
baseline en `src/common.py`.

En cada ejecución, Beta 2:

1. divide los datos una sola vez;
2. crea segmentos compartidos para los datos y para `Acc`, `n_iter`, `t_seed` y
   `Mod`;
3. inicia un `multiprocessing.Pool` con `p` procesos y conecta los workers a esos
   segmentos;
4. ejecuta las `m` semillas;
5. reduce las accuracies en el proceso maestro y reconstruye el modelo ganador.

El diccionario de tiempos separa `split`, `shm_setup`, `pool_start`, `map`, `join`,
`reduccion`, `shm_cleanup` y `total`. Por tanto, los tiempos observados incluyen
costos del runtime y de memoria compartida que no aparecen en `C`. La matriz de
parámetros compartida tiene una fila por semilla y puede requerir memoria
proporcional a `m` por el número de parámetros del MLP.

Beta 0 (`src/beta0_sequential.py`) es el baseline secuencial para las mismas `m`
semillas. Beta 1 (`src/beta1_parallel.py`) es una versión histórica: usaba `m=p`,
pasaba los datos a las tareas y seleccionaba el máximo con `max` secuencial. No es
la implementación final ni la base de las conclusiones de strong scaling.

## 6. Configuración de datos y entrenamiento

Los valores compartidos por defecto en `src/common.py` son:

```text
n_features = 20
n_informative = 15
clases = 2
random_state de generación = 42
test_size = 0.2
split_seed = 0
hidden_layer_sizes = (10, 10)
solver = adam
alpha = 1e-4
learning_rate_init = 1e-3
max_iter = 200
```

El solver `adam` es el predeterminado de `MLPClassifier`; `max_iter` es un máximo,
no una garantía de que todos los modelos ejecuten exactamente ese número de
iteraciones. El número efectivo (`n_iter_`) puede variar según la semilla y la
convergencia. Cambiar arquitectura o hiperparámetros requiere tratarlo como otra
configuración experimental.

## 7. Comparación experimental

La campaña vigente es de **strong scaling**: mantiene `m=32` semillas y el mismo
dataset, y varía `p`. El runner `benchmarks/run_experiments.py` compara Beta 0 y
Beta 2 para cada `n` y repetición emparejada. El speedup y la eficiencia son:

```text
S(n,m,p)  = T_seq(n,m) / T_par(n,m,p)
Ef(n,m,p) = S(n,m,p) / p
```

El baseline secuencial ejecuta las mismas `m` semillas que Beta 2, por lo que el
workload no cambia al variar `p`. Se excluye la generación del dataset de la región
cronometrada; el split se incluye en ambas implementaciones. En Beta 2 también se
miden el setup y la limpieza de memoria compartida, el ciclo de vida del pool, la
ejecución y la reducción. `p` representa los procesos solicitados; si supera los
CPU disponibles, puede haber contención.

La campaña final documentada en el `README` usa `m=32`, `p={1,2,4,8}`,
`n_samples={5000,10000,20000}` y tres repeticiones. La campaña histórica de Beta 1
con `m=p` es distinta y no debe mezclarse con los resultados de strong scaling.

## 8. Interpretación de resultados

La accuracy que se usa para escoger la semilla se calcula sobre el conjunto llamado
test. Como ese mismo conjunto interviene en la selección, la accuracy del ganador
no es una estimación independiente de generalización. Para medir generalización
sin sesgo de selección se necesitaría seleccionar con validación y reservar otro
test para la evaluación final.

La comparación de correctitud entre Beta 0 y Beta 2 verifica el mismo ganador,
accuracy y `n_iter_` por semilla para el workload de prueba. Esto valida que la
paralelización preserve el cálculo; no elimina el overhead ni implica que los
tiempos reales sigan las cotas PRAM ideales.
