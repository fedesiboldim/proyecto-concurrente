# Hito 2: Aceleración Masiva en GPU con CUDA — Conway's Game of Life

## 1. Entorno Experimental
* **Acelerador:** NVIDIA Tesla T4 (Arquitectura Turing)
* **Memoria VRAM:** 16 GB (15,360 MiB) GDDR6
* **Entorno de ejecución:** Google Colab (Linux x86_64) + CUDA 13.0
* **Framework:** Python 3 + Numba CUDA

---

## 2. Decisiones de Diseño y Mapeo SIMT
* **Mapeo Hilo-Célula:** Se utilizó un modelo data-parallel donde cada hilo de la GPU calcula el estado de una única célula $(i, j)$ mediante `cuda.grid(2)`.
* **Organización de Bloques:** Se configuraron bloques 2D de $16 \times 16$ hilos (256 hilos por bloque). Esto asegura la alineación con múltiplos de 32 hilos (tamaño de un *warp* de NVIDIA), favoreciendo la ejecución paralela y el acceso coalescente a memoria global.
* **Técnica de Buffer-Swapping:** Para garantizar la seguridad (*safety*) y evitar condiciones de carrera, se reservan dos matrices en VRAM (`d_grid` y `d_next`). Las lecturas y escrituras alternan sus roles en cada iteración, garantizando que ningún hilo sobreescriba datos en uso.
* **Sincronización:** Cada generación se delimita mediante el fin de ejecución del kernel y la llamada a `cuda.synchronize()`, sirviendo de barrera global entre generaciones.

---

## 3. Resultados Comparativos (CPU vs GPU)

| Resolución ($N \times N$) | Tiempo CPU Secuencial (ms/gen) | Tiempo GPU Tesla T4 (ms/gen) | Speedup Absoluto (GPU vs CPU) |
| :---: | :---: | :---: | :---: |
| $512 \times 512$ | 552.4 ms | **0.14 ms** | **~3,945x** |
| $1024 \times 1024$ | 2,025.4 ms | **0.49 ms** | **~4,133x** |
| $2048 \times 2048$ | *Inviable en CPU* (~8,000 ms est.) | **1.95 ms** | **>4,000x** |

---

## 4. Análisis Teórico (Unidades 1, 3 y 6)

### Ley de Gustafson y Escalabilidad
A medida que el tamaño de la grilla escala de $512$ a $2048$, la carga de trabajo paralela crece cuadráticamente ($O(N^2)$). La GPU satura de forma óptima sus Streaming Multiprocessors (SMs), amortizando el costo fijo de lanzamiento del kernel y sosteniendo un factor de aceleración superior a 4000x.

### Cuello de Botella PCIe
Para evitar el cuello de botella del bus PCIe (Host $\leftrightarrow$ Device), la transferencia de datos se realiza únicamente dos veces:
1. Una copia inicial de la grilla de CPU a VRAM (`cuda.to_device`).
2. La ejecución completa de las $N$ generaciones directamente en la memoria de la placa gráfica.
Esto demuestra la regla fundamental de arquitecturas heterogéneas: mantener los datos en la memoria del dispositivo el mayor tiempo posible.
