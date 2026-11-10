# Fuentes e impacto en la versión final

Esta revisión complementa `report/references.bib`. El informe final debe citar estas fuentes cerca de cada decisión y reemplazar la explicación heredada de Beta 1 por Beta 2. No atribuir al libro de PRAM ni a la documentación de Python mediciones propias del proyecto.

| Fuente | Decisión o afirmación que respalda | Uso en el informe |
|---|---|---|
| Enunciado oficial del curso (`fiestas2026proyecto`) | Problema, costo base `Θ(E n d h)` y exigencias del proyecto | Aclarar que es el modelo dado para una semilla, no una cota exacta de scikit-learn. Verificar los datos bibliográficos contra el PDF original del curso. |
| [Grama et al., segunda edición](https://www.cs.purdue.edu/homes/ayg/book/index.html) (`grama2003`) | Marco de trabajo, profundidad, tiempo paralelo, speedup y eficiencia | Derivar las expresiones del algoritmo final. Autores, año, edición e ISBN cotejados con la página del libro de uno de sus autores. |
| [Python: multiprocessing](https://docs.python.org/3/library/multiprocessing.html) (`pythonmultiprocessing`) | `Pool` ejecuta tareas mediante procesos, también en Windows | Explicar creación y planificación de procesos como costos reales. |
| [Python: shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) (`pythonsharedmemory`) | Segmentos a los que se conectan procesos distintos | Respaldar la decisión de compartir el dataset en Beta 2 y explicar creación, copia inicial y limpieza. |
| [scikit-learn: MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) (`sklearnmlp`) | `solver=adam`, `random_state`, `max_iter` y `n_iter_` | Declarar la configuración real y distinguir máximo de iteraciones de iteraciones efectivas. |
| [scikit-learn: make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) (`sklearnmakeclassification`) | Generador sintético parametrizado | Justificar dataset reproducible y tamaños `n` controlados. |
| [scikit-learn: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) (`sklearnmodelselection`) | Split reproducible con `random_state` entero | Justificar la misma partición para todas las semillas y configuraciones. |
| [Python: perf_counter](https://docs.python.org/3/library/time.html#time.perf_counter) (`pythontime`) | Medición de duraciones | Explicar el reloj empleado para tiempos, sin atribuirle precisión física no demostrada. |
| [threadpoolctl](https://github.com/joblib/threadpoolctl) (`threadpoolctl`) | Inspección y límite de hilos BLAS/OpenMP | Justificar el control del paralelismo interno y registrar `threadpool_info` en metadata. |
| [Goodfellow et al., MIT Press](https://mitpress.mit.edu/9780262035613/deep-learning/) (`goodfellow2016`) | Conceptos de MLP y entrenamiento | Contexto conceptual, sin usarlo para validar los tiempos o el costo particular de scikit-learn. |

## Revisión pendiente del informe

- El informe actualizado aportado el 10 de noviembre presenta Beta 2 como resultado central y rotula Beta 1 como histórico. Sus tablas, figuras y conclusiones cuantitativas se completaron desde el CSV final Windows.
- Citar `pythonsharedmemory` en la sección de implementación y overhead de Beta 2.
- Revisar cada tabla y figura contra el CSV final, el commit de código y la metadata de la misma campaña.
- Mantener la limitación de accuracy: el conjunto llamado `test` se usa para escoger el ganador y no es una evaluación independiente de generalización.
