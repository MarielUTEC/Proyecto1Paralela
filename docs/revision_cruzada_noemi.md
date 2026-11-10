# Revisión cruzada de Noemí: código, método y evidencia

Fecha: 10 de noviembre de 2026. Revisión sobre el commit `ded6d49`, usado como base de la campaña Windows.

## Correspondencia semántica

- La unidad de trabajo de Beta 2 es una semilla (`_tarea(seed)`); `Pool.map(..., range(m), chunksize=1)` distribuye las `m` semillas entre `p` procesos. No hay paralelismo implementado dentro de un MLP.
- Los workers conectan los arreglos de datos de `multiprocessing.shared_memory` y marcan las vistas de `X_train`, `y_train`, `X_test` y `y_test` como no escribibles. Cada tarea escribe únicamente los índices de su semilla en accuracy, iteraciones, tiempo y parámetros del modelo.
- La reducción se organiza en rondas de pares, pero la ejecuta el proceso maestro de forma secuencial. Su tiempo real es `Θ(m)`; la profundidad `Θ(log m)` corresponde al árbol PRAM abstracto, no a la ejecución del maestro.
- El modelo ganador se reconstruye desde pesos y sesgos para inferencia. La reconstrucción no conserva todo el estado del optimizador del entrenamiento; la prueba verifica su accuracy de inferencia.
- `solver=adam` es explícito en Beta 0 y Beta 2. Se fijan datos, split, semillas e hiperparámetros, y se limitan a un hilo las bibliotecas numéricas en maestro y workers.

## Evidencia de correctitud

- Suite Windows: `8 passed, 1 skipped` con `TEST_N=1500`, `TEST_M=32` (10 de noviembre de 2026). Se omite solo la comparación `fork` vs. `spawn`, porque Windows no ofrece `fork`; las otras pruebas ejercitan `spawn`.
- Piloto de benchmark: `m=8`, `n=500`, `p=1,2,4`, una repetición. El resumen marcó `all_equivalent=True` para todos los puntos. El piloto no sustenta conclusiones de rendimiento final.
- El runner final detiene la campaña si ganador, accuracy por semilla o `n_iter_` por semilla no coinciden entre Beta 0 y Beta 2. Los vectores `n_iter_` se guardan en el CSV crudo como JSON.

## Puntos que debe reflejar el informe integrado

- El informe actualizado aportado el 10 de noviembre ya narra Beta 2 como implementación principal. Para el resultado central deben citarse únicamente los CSV y metadatos de la campaña Beta 2 en Windows. Los históricos están rotulados como preliminares.
- La tabla histórica de Khipu del informe actualizado no coincide con los CSV antiguos presentes en esta copia de GitHub. Esos CSV no representan necesariamente la versión actual. Conservar la tabla como histórica y verificar su procedencia cuando se disponga de los archivos crudos y metadatos correspondientes.
- `docs/decisiones.md` y `docs/derivacion_pram.md` conservan frases sobre Beta 1 como implementación actual. Actualizar la redacción al integrar la teoría final.
- Añadir la cita `pythonsharedmemory` del archivo BibTeX a la explicación de los segmentos compartidos y su costo de inicialización/limpieza.
- Describir la selección de la semilla mediante la accuracy del conjunto llamado `test` como selección, no como estimación independiente de generalización.
- La auditoría `benchmarks/validate_campaign.py` verificó todas las cifras agregadas del resumen contra el CSV crudo, el commit de código y la metadata de la campaña. Las cuatro figuras finales se generaron desde ese resumen.
