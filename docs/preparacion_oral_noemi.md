# Preparación oral de Noemí

Versión del 10 de noviembre de 2026, completada con el CSV final Windows.

## ¿Qué bucle se paralelizó y por qué `PARDO` va sobre las semillas?

Se paralelizó el ciclo `for seed in S`. Cada semilla inicializa y entrena un MLP independiente sobre el mismo split, evalúa su accuracy y escribe en una posición propia. Dos semillas no dependen entre sí durante el entrenamiento. La única combinación global ocurre al escoger la mayor accuracy. Beta 2 materializa el ciclo mediante `Pool.map(_tarea, range(m), chunksize=1)` con `p` procesos. No paralelizamos las operaciones internas del MLP.

## Deriva trabajo y profundidad desde el pseudocódigo

Si entrenar una semilla cuesta `C(n,d,h,E)=Θ(E n d h)` según el enunciado, las `m` semillas suman `m C` de trabajo. La reducción del máximo requiere `m-1` comparaciones: `W=Θ(m C+m)=Θ(m E n d h)` cuando domina el entrenamiento. Con suficientes procesadores, los entrenamientos independientes corren a la vez y la reducción ideal en árbol tiene `⌈log₂ m⌉` rondas: `D=Θ(C+log m)`. La implementación Beta 2 ejecuta las rondas serialmente en el maestro; por eso su reducción medida cuesta `Θ(m)`, aunque conserve la estructura lógica de árbol.

## ¿Qué cambió respecto de la entrega parcial?

Beta 1 enviaba datos a las tareas, usaba `m=p` en la campaña preliminar y elegía el máximo secuencialmente. Beta 2 mantiene `m` independiente de `p`, comparte el dataset mediante `multiprocessing.shared_memory`, registra fases de tiempo y conserva escrituras separadas por semilla. La campaña final fija `m=32` para medir strong scaling en un solo equipo.

## ¿Qué prueba que los resultados son comparables?

Para cada tamaño `n` se usan las mismas 32 semillas, el mismo generador y split, la misma arquitectura y los mismos hiperparámetros. Se limita a uno el número de hilos internos de bibliotecas numéricas. Cada repetición empareja un baseline secuencial con cada `p` paralelo y el runner verifica ganador, accuracy y `n_iter_` de cada semilla antes de guardar los datos.

## Cifras y límites para la defensa

- Mejor speedup mediano: 3.65 para `n=20000`, `m=32`, `p=8`; eficiencia 45.6%. El paralelo tardó 50.34 s frente a 183.58 s del baseline secuencial, ambos como medianas.
- No se observó saturación hasta `p=8`: el tiempo paralelo disminuyó en cada paso de `p`. La curva sigue por debajo de la referencia ideal `S=p`.
- En `p=8`, las eficiencias fueron 37.3%, 43.4% y 45.6% para `n=5000`, `10000` y `20000`. El overhead estimado `p*T_par-T_seq` aumentó de 80.28 s a 219.12 s entre tamaños extremos; mide exceso de tiempo agregado, no una fase aislada.
- La fase `map` ocupa casi todo el tiempo medido; incluye entrenamiento, planificación y parte del inicio efectivo de workers. No se puede atribuir la brecha con el ideal a una sola causa usando estas fases.
- No hubo fallos de equivalencia ni puntos descartados; las 12 configuraciones tienen tres repeticiones y la auditoría no encontró una medición paralela más de 20% lejos de su mediana.
- La accuracy del conjunto llamado `test` selecciona la semilla ganadora. Una evaluación independiente de generalización requeriría otro conjunto de prueba no usado en la selección.
