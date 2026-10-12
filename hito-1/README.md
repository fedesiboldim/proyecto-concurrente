# Hito 1: Paralelismo en CPU — Conway's Game of Life

## 1. Entorno de Ejecución y Hardware
* **Lenguaje:** Python 3.11.9 + NumPy
* **Procesador (CPU):** 8 núcleos lógicos
* **Sistema Operativo:** Windows 11
* **Estrategia Concurrente:** Procesamiento paralelo por lotes de franjas mediante `multiprocessing.Pool`

---

## 2. Decisiones de Diseño y Seguridad (Safety)
* **Partición de Datos:** La grilla 2D se divide horizontalmente en franjas contiguas de filas distribuidas de forma balanceada entre los 8 núcleos de CPU. Esto maximiza la localidad espacial de la caché L1/L2 al iterar contiguamente sobre los arreglos de NumPy (*stride-1*).
* **Ausencia de Condiciones de Carrera (Safety):** Cada paso de simulación calcula un nuevo arreglo `next_grid` a partir de un arreglo de lectura `grid`. Ningún proceso modifica celdas que otro proceso está leyendo simultáneamente en la misma generación.
* **Sincronización:** Cada generación actúa como una barrera explícita: todos los subprocesos deben finalizar su franja antes de componer la grilla y pasar a la generación siguiente.

---

## 3. Resultados Experimentales

| Tamaño de Grilla ($N \times N$) | Generaciones | Tiempo Secuencial (s) | Tiempo Paralelo (s) | Speedup ($S = T_s / T_p$) | Eficiencia ($E = S / p$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $512 \times 512$ | 3 | 1.6573 s | 0.5651 s | **2.93x** | **0.37** |
| $1024 \times 1024$ | 3 | 6.0761 s | 2.2868 s | **2.66x** | **0.33** |

---

## 4. Análisis y Preguntas Teóricas del Programa

### ¿Por qué el Speedup no es lineal ($S < 8$ con 8 núcleos)?
1. **Sobrecosto de sincronización y IPC (Inter-Process Communication):** En Python, `multiprocessing` debe serializar y transmitir los fragmentos de matrices entre el proceso padre y los trabajadores (overhead de pickle/IPC) en cada generación.
2. **Cuello de botella de Memoria (Von Neumann bottleneck):** El Juego de la Vida es un algoritmo limitado por ancho de banda de memoria (*memory-bound*). Los 8 núcleos compiten por el mismo bus de memoria RAM al leer las vecindades de las celdas contiguas.
3. **Ley de Amdahl:** La fracción secuencial (creación de procesos, división de tareas y concatenación final de la grilla) impone un techo teórico a la aceleración máxima alcanzable.
