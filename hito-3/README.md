# Hito 3: Optimización con Shared Memory y Escalabilidad Masiva — Conway's Game of Life

## 1. Entorno de Ejecución
* **Hardware:** NVIDIA Tesla T4 (16 GB VRAM, Arquitectura Turing)
* **Entorno:** Google Colab con soporte CUDA
* **Librerías:** Python 3 + Numba CUDA + NumPy

---

## 2. Decisiones de Diseño y Optimización de Memoria

### Técnica de Baldosas (Tiling) con Shared Memory
En el kernel básico (Hito 2), cada hilo leía directamente de la Memoria Global (VRAM) para consultar a sus 8 vecinos, generando redundancia de lecturas en el bus. 

En este hito se implementó la técnica de **Tiling**:
1. Cada bloque de $16 \times 16$ hilos reserva un arreglo en memoria compartida: `cuda.shared.array((18, 18), dtype=np.int32)`.
2. Los hilos cargan cooperativamente su celda central más un halo perimetral de 1 celda (vecinos de borde y esquinas).
3. **Barrera de Sincronización:** Se invoca obligatoriamente `cuda.syncthreads()` para asegurar que la baldosa completa esté en memoria rápida antes de calcular.
4. Las lecturas de los 8 vecinos se realizan sobre la *Shared Memory* (baja latencia y alto ancho de banda, equivalente a caché L1), reduciendo drásticamente la contención en VRAM.

---

## 3. Resultados Experimentales

| Resolución ($N \times N$) | Millones de Celdas | Tiempo Total (10 gen) | Latencia por Gen | Rendimiento (Throughput) |
| :---: | :---: | :---: | :---: | :---: |
| $1024 \times 1024$ | 1.0 M | 0.0018 s | **0.18 ms/gen** | 5,711.0 MCells/s |
| $2048 \times 2048$ | 4.2 M | 0.0069 s | **0.69 ms/gen** | 6,092.1 MCells/s |
| $4096 \times 4096$ | 16.8 M | 0.0271 s | **2.71 ms/gen** | 6,186.1 MCells/s |
| $8192 \times 8192$ | 67.1 M | 0.1089 s | **10.89 ms/gen** | 6,164.3 MCells/s |

### Comparativa de Optimización (Hito 2 vs Hito 3) en $1024 \times 1024$:
* **Kernel Global (Hito 2):** $0.49\text{ ms/gen}$
* **Kernel Shared Memory (Hito 3):** $0.18\text{ ms/gen}$
* **Aceleración por Shared Memory:** **2.72x más rápido**

---

## 4. Análisis Teórico (Unidades 3 y 6)

### Sincronización y Barreras Intra-bloque
`cuda.syncthreads()` garantiza una barrera estricta dentro del bloque de hilos. Sin esta primitiva ocurriría una condición de carrera (*read-after-write hazard*), donde un hilo intentaría evaluar el estado de un vecino antes de que el hilo contiguo termine de cargarlo desde la memoria global.

### Ley de Gustafson y Saturación del Silicio
Al escalar el problema de 1 millón a 67 millones de celdas, el *throughput* se mantiene constante en el techo de saturación de la placa (~6.1 GCells/s). Esto demuestra la Ley de Gustafson: a mayor tamaño del problema, la GPU maximiza su ocupación y eficiencia de cómputo, pulverizando cualquier limitación teórica impuesta por la Ley de Amdahl para problemas de tamaño fijo.