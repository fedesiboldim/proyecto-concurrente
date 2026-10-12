import numpy as np
import time
from numba import cuda

# Dimensiones del bloque y tamaño de la baldosa con halo
TPB = 16  # Threads Per Block en cada dimension

# 1. Kernel optimizado con Shared Memory (Tiling)
@cuda.jit
def game_of_life_shared(d_grid, d_next_grid, N):
    # Baldosa de (16 + 2) x (16 + 2) para incluir el halo de 1 celda de vecinos
    s_tile = cuda.shared.array((18, 18), dtype=np.int32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Posicion global
    gx = bx * TPB + tx
    gy = by * TPB + ty

    # Coordenadas locales dentro de la baldosa compartida (1 a 16)
    lx = tx + 1
    ly = ty + 1

    # Cargar celda central
    if gx < N and gy < N:
        s_tile[ly, lx] = d_grid[gy, gx]
    else:
        s_tile[ly, lx] = 0

    # Cargar halos (bordes del tile) cooperativamente
    if tx == 0:
        s_tile[ly, 0] = d_grid[gy, gx - 1] if (gx > 0 and gy < N) else 0
    if tx == TPB - 1:
        s_tile[ly, 17] = d_grid[gy, gx + 1] if (gx + 1 < N and gy < N) else 0
    if ty == 0:
        s_tile[0, lx] = d_grid[gy - 1, gx] if (gy > 0 and gx < N) else 0
    if ty == TPB - 1:
        s_tile[17, lx] = d_grid[gy + 1, gx] if (gy + 1 < N and gx < N) else 0

    # Esquinas del halo
    if tx == 0 and ty == 0:
        s_tile[0, 0] = d_grid[gy - 1, gx - 1] if (gx > 0 and gy > 0) else 0
    if tx == TPB - 1 and ty == 0:
        s_tile[0, 17] = d_grid[gy - 1, gx + 1] if (gx + 1 < N and gy > 0) else 0
    if tx == 0 and ty == TPB - 1:
        s_tile[17, 0] = d_grid[gy + 1, gx - 1] if (gx > 0 and gy + 1 < N) else 0
    if tx == TPB - 1 and ty == TPB - 1:
        s_tile[17, 17] = d_grid[gy + 1, gx + 1] if (gx + 1 < N and gy + 1 < N) else 0

    # BARRERA DE SINCRONIZACIÓN: Asegurar que toda la baldosa esté cargada
    cuda.syncthreads()

    # Calcular estado consultando solo la Shared Memory ultrarrápida
    if 1 <= gx < N - 1 and 1 <= gy < N - 1:
        vecinos = (s_tile[ly - 1, lx - 1] + s_tile[ly - 1, lx] + s_tile[ly - 1, lx + 1] +
                   s_tile[ly,     lx - 1]                      + s_tile[ly,     lx + 1] +
                   s_tile[ly + 1, lx - 1] + s_tile[ly + 1, lx] + s_tile[ly + 1, lx + 1])

        if s_tile[ly, lx] == 1:
            d_next_grid[gy, gx] = 1 if (vecinos == 2 or vecinos == 3) else 0
        else:
            d_next_grid[gy, gx] = 1 if vecinos == 3 else 0

# --- EXPERIMENTO DE ESCALABILIDAD MASIVA (HITO 3) ---
tamanos = [1024, 2048, 4096, 8192]
generaciones = 10
block_dim = (TPB, TPB)

print("=== HITO 3: ESCALABILIDAD Y SHARED MEMORY (NVIDIA Tesla T4) ===")

for N in tamanos:
    grid = np.random.choice([0, 1], size=(N, N), p=[0.8, 0.2]).astype(np.int32)
    grid_dim = ((N + TPB - 1) // TPB, (N + TPB - 1) // TPB)

    d_grid = cuda.to_device(grid)
    d_next = cuda.device_array_like(d_grid)

    # Warmup
    game_of_life_shared[grid_dim, block_dim](d_grid, d_next, N)
    cuda.synchronize()

    t0 = time.time()
    for gen in range(generaciones):
        if gen % 2 == 0:
            game_of_life_shared[grid_dim, block_dim](d_grid, d_next, N)
        else:
            game_of_life_shared[grid_dim, block_dim](d_next, d_grid, N)
    cuda.synchronize()
    t_gpu = time.time() - t0

    ms_gen = (t_gpu / generaciones) * 1000
    celdas_total = N * N
    throughput_mcells = (celdas_total * generaciones / t_gpu) / 1e6

    print(f"Grilla {N:4d}x{N:4d} ({celdas_total/1e6:5.1f}M celdas) | Tiempo: {t_gpu:.4f} s | {ms_gen:6.2f} ms/gen | Throughput: {throughput_mcells:8.1f} MCells/s")