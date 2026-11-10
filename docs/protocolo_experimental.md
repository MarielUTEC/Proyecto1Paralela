# Protocolo experimental de la campaña final (Noemí)

Estado: preparado para revisión del equipo. La campaña final debe ejecutarse después de validar Beta 0 y Beta 2 con la misma configuración y de fijar el commit de código.

## Decisiones congeladas

- Plataforma principal: esta PC Windows 11, AMD Ryzen 5 4600H, 12 CPU lógicas. Todas las curvas finales se medirán aquí; los CSV históricos de Linux/Khipu se conservarán por separado.
- Workload: `m=32` modelos con semillas `0..31` en todas las configuraciones. `p` designa procesos y toma `1,2,4,8`; se excluyen `16,32` de la campaña principal porque superarían las 12 CPU lógicas disponibles.
- Tamaños: `n_samples=5000,10000,20000`. Para cada `n`, se genera el mismo dataset con 20 features, 15 informativas, dos clases y `random_state=42`. Split `test_size=0.2`, `split_seed=0`.
- MLP: capas ocultas `(10,10)`, `solver=adam`, `alpha=1e-4`, `learning_rate_init=1e-3`, `max_iter=200`, y `random_state=seed`.
- Bibliotecas numéricas: un hilo interno por proceso mediante variables de entorno y `threadpoolctl`. Dependencias exactas en `requirements-lock.txt`.
- Medición: tres repeticiones por combinación `(n,p)`; generación de datos fuera del cronómetro. Beta 0 se mide una vez por `(n,repetición)` y se empareja con cada Beta 2 del mismo `n` y repetición. Se alterna si Beta 0 va antes o después de las ejecuciones paralelas.
- Métricas: `S(n,m,p)=T_seq(n,m)/T_par(n,m,p)`, `E=S/p`, overhead estimado `p*T_par-T_seq`. Se reportan medianas y dispersión de repeticiones emparejadas.

## Gates antes de aceptar resultados

1. Ejecutar `python -m pytest tests -q` en el entorno de campaña. En Windows, la prueba exclusiva de `fork` se omite; las pruebas generales ejercitan `spawn`.
2. Ejecutar un piloto con rutas separadas: `python benchmarks/run_experiments.py --m 8 --p 1 2 4 --n_samples 500 --repeticiones 1 --raw_out results/raw/piloto_windows.csv --summary_out results/processed/piloto_windows.csv`.
3. Comprobar en el CSV piloto que el ganador, las accuracies y `n_iter_` por semilla coinciden exactamente entre Beta 0 y Beta 2; que el resumen indica `all_equivalent=True`; y que el metadata contiene hardware, versiones y límites de hilos.
4. Congelar el código mediante un commit antes de la campaña principal. No modificar implementación, dependencias ni parámetros durante sus repeticiones.

## Ejecución final

Desde la raíz del repositorio, usando el Python del entorno virtual:

```powershell
.\.venv\Scripts\python.exe benchmarks\run_experiments.py --m 32 --p 1 2 4 8 --n_samples 5000 10000 20000 --repeticiones 3 --raw_out results\raw\campana_final_windows_m32.csv --summary_out results\processed\resumen_campana_final_windows_m32.csv
```

Las salidas nuevas no se sobrescriben sin `--overwrite`. El CSV crudo se escribe tras cada repetición completa y el resumen se recalcula a partir de él. Si se interrumpe la campaña, conservar el archivo parcial y ejecutar una campaña nueva con nombre distinto; no combinar mediciones de entornos o commits diferentes.

## Anomalías y descarte

Registrar cualquier fallo, diferencia funcional, interrupción, saturación u observación atípica en una bitácora con fecha, configuración, repetición y motivo. Repetir la configuración afectada con el mismo entorno. Excluir una medición solo por un fallo identificable del procedimiento o de la máquina, nunca por empeorar el speedup. Conservar la medición original y explicar el criterio en el informe.

## Alcance de la accuracy

La accuracy selecciona al ganador sobre el conjunto llamado `test`; no estima de manera independiente la generalización del modelo seleccionado. Las repeticiones mantienen datos y semillas fijos para medir variación temporal, no incertidumbre estadística de accuracy.
