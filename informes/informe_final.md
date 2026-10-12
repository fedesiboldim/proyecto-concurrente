# Informe Final Integrador: Conway's Game of Life en GPU

**Cátedra:** Programación Concurrente y Paralela — UNaB  
**Docente:** Lic. Angel Leonardo Bianco  
**Proyecto:** Simulación de Grilla Masiva con Paralelismo de Datos (GPU)  

---

## 1. Introducción y Arquitectura Experimental
El presente proyecto analiza y optimiza la ejecución del autómata celular bidimensional *Conway's Game of Life* sobre grillas de alta resolución (hasta $8192 \times 8192$, equivalente a 67.1 millones de celdas). Se exploró la evolución desde el modelo de memoria compartida en CPU multicore hasta la aceleración masivamente paralela en GPU mediante CUDA.

### Entorno de Hardware y Software:
* **CPU Host:** Procesador de 8 núcleos lógicos bajo Windows 11.
* **GPU Device:** NVIDIA Tesla T4 (Arquitectura Turing, 2560 CUDA Cores, 16 GB GDDR6, PCIe Gen3 x16).
* **Entorno de Desarrollo:** Python 3.11, NumPy, Numba CUDA, CUDA Driver 580.82 / CUDA 13.0.

---

## 2. Síntesis Evolutiva por Hitos

### Hito 1: Paralelismo en CPU (Memoria Compartida y Amdahl)
* Se implementó el recorrido secuencial y una versión paralela particionada por franjas horizontales de filas (`multiprocessing.Pool`).
* **Resultados ($1024 \times 1024$, 3 gen):**
  * Tiempo Secuencial: 6.0761 s (2.0254 s/gen).
  * Tiempo Paralelo (8 cores): 2.2868 s.
  * **Speedup:** 2.66x | **Eficiencia:** 0.33.
* **Limitación:** El *speedup* se estanca por el cuello de botella del bus de memoria RAM al leer matrices bidimensionales y el overhead de serialización (IPC/pickle) entre procesos.

### Hito 2: Modelo SIMT en GPU (Memoria Global)
* Mapeo de 1 hilo de GPU por cada célula de la grilla mediante `cuda.grid(2)`.
* Bloques 2D de $16 \times 16$ hilos (256 hilos por bloque), alineados con el tamaño de *warp* (32 hilos).
* **Resultados ($1024 \times 1024$):**
  * Tiempo GPU: **0.49 ms/gen**.
  * **Speedup frente a CPU:** **~4,133x**.

### Hito 3: Optimización con Shared Memory y Escalabilidad Masiva
* Reducción de la presión en el bus de VRAM implementando *Tiling* con halos: cada bloque carga cooperativamente una baldosa de $18 \times 18$ en memoria compartida.
* Sincronización estricta intra-bloque mediante barrera `cuda.syncthreads()`.
* **Resultados ($1024 \times 1024$):**
  * Reducción de 0.49 ms/gen a **0.18 ms/gen** (aceleración adicional de 2.72x por uso de Shared Memory).
* **Escala $8192 \times 8192$ (67.1M celdas):**
  * Latencia: **10.89 ms/gen**.
  * *Throughput:* **6,164.3 MCells/s** sostenidos.

---

## 3. Análisis Teórico y Conceptos de Cátedra

### A. Ausencia de Condiciones de Carrera (Safety y Liveness)
El algoritmo requiere evaluar el estado de la vecindad en el tiempo $t$ para determinar el estado en $t+1$. Si se escribiera sobre la misma grilla, se generarían *write-after-read hazards*. Para garantizar *safety*, se implementó **buffer-swapping** (dos arreglos en VRAM, alternando lectura y escritura entre generaciones). La finalización de cada kernel actúa como una barrera global implícita, garantizando que ninguna célula avance a $t+1$ antes de que toda la grilla complete el paso $t$.

### B. Paradigma Manager / Worker
* **Manager (CPU):** Inicializa la grilla pseudoaleatoria en memoria principal y despacha la configuración de ejecución (`grid_dim`, `block_dim`) al acelerador.
* **Worker (GPU):** Ejecuta de forma masiva y autónoma las transiciones de estado en su memoria interna, evitando transferencias intermedias.

### C. Ley de Amdahl vs. Ley de Gustafson
* **Amdahl:** Explica la baja eficiencia observada en CPU ($E = 0.33$), donde la sincronización y la fracción secuencial dominan el tiempo de ejecución.
* **Gustafson:** Al escalar el tamaño del problema de $512 \times 512$ a $8192 \times 8192$, la GPU alcanza un rendimiento constante de más de 6 mil millones de celdas por segundo, demostrando que la arquitectura masivamente paralela maximiza su rendimiento a mayor escala de datos.

---

## 4. Conclusión
El proyecto demostró que la selección de la arquitectura y la gestión de la jerarquía de memoria son determinantes: la combinación de *Tiling* en Shared Memory, buffer-swapping y eliminación de transferencias por el bus PCIe permitió pasar de segundos por generación en CPU a milisegundos en grillas de decenas de millones de celdas en GPU.