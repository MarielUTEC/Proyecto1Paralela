# Bitácora de experimentación de Noemí

## 10 de noviembre de 2026: preparación

- Plataforma elegida: PC Windows 11, AMD Ryzen 5 4600H, 12 CPU lógicas, 15.37 GiB de RAM. Se instaló Python 3.12.14 y las dependencias fijadas en `requirements-lock.txt`.
- Se validó Beta 0 frente a Beta 2 con la suite completa: 8 pruebas correctas y 1 omitida por usar `fork`, no disponible en Windows.
- Piloto `m=8`, `n=500`, `p=1,2,4`, una repetición: equivalencia completa. El piloto mostró overhead de procesos elevado en un workload pequeño y se excluye de las conclusiones finales.
- Se fijó el código de medición en el commit `ded6d49` antes de iniciar la campaña principal.

## 10 de noviembre de 2026: campaña principal

- Configuración: `m=32`, `n=5000,10000,20000`, `p=1,2,4,8`, tres repeticiones por configuración. Dataset, split, semillas, solver e hiperparámetros fijos. Hilos internos limitados a uno.
- Archivos: `results/raw/campana_final_windows_m32.csv`, `results/processed/resumen_campana_final_windows_m32.csv` y `results/raw/campana_final_windows_m32.metadata.json`.
- Auditoría con `benchmarks/validate_campaign.py`: 45 filas crudas, 12 puntos de resumen, tres repeticiones por punto, sin duplicados; ganador, accuracy y `n_iter_` por semilla iguales entre secuencial y paralelo en todos los puntos.
- Revisión de anomalías: ninguna medición paralela se apartó más del 20% de la mediana de sus tres repeticiones. No se descartó ni reemplazó ninguna medición.
- `n_iter_`: para `n=5000`, las 32 semillas alcanzaron 200 iteraciones; para `n=10000` y `n=20000`, 31 de 32 alcanzaron 200, con mínimos respectivos de 197 y 178.
- El mejor speedup mediano medido fue 3.65 en `n=20000`, `p=8`; la eficiencia allí fue 45.6%. El tiempo paralelo mediano bajó de 192.38 s (`p=1`) a 50.34 s (`p=8`).

## Límites de interpretación

- La curva principal usa un solo hardware y no mezcla CSV históricos de Linux o Khipu.
- Los tres tamaños y el máximo `p=8` no permiten afirmar dónde se saturaría esta PC con más procesos.
- `map` reúne entrenamiento, planificación y parte del inicio efectivo de workers; las fases no identifican por sí solas la causa del overhead.
- La accuracy sobre el conjunto llamado `test` se usa para seleccionar el ganador, no para estimar su generalización de manera independiente.
- La tabla histórica de Khipu del informe actualizado no se ha cotejado con archivos crudos de esa misma campaña, que no están en esta copia del repositorio. Se conserva separada de la campaña final.
