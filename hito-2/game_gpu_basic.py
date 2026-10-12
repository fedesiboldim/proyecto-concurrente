import numpy as np
import time
from numba import cuda

# 1. Kernel CUDA para el Juego de la Vida
@cuda.jit
def game_of_life_gpu(d_grid, d_next_grid, N):
    i, j = cuda.grid(2)
    
    # Manejo de bordes internos (1 a N-2)
    if 1 <= i < N - 1 and 1 <= j < N - 1:
        vecinos = (d_grid[i-1, j-1] + d_grid[i-1, j] + d_grid[i-1, j+1] +
                   d_grid[i,   j-1]                  + d_grid[i,   j+1] +
                   d_grid[i+1, j-1] + d_grid[i+1, j] + d_grid[i+1, j+1])
        
        if d_grid[i, j] == 1:
            d_next_grid[i, j] = 1 if (vecinos == 2 or vecinos == 3) else 0
        else:
            d_next_grid[i, j] = 1 if vecinos == 3 else 0

# 2. Configuración experimental
tamanos = [512, 1024, 2048]
generaciones = 10
block_dim = (16, 16)

print("=== HITO 2: BENCHMARK GPU (NVIDIA Tesla T4) ===")

for N in tamanos:
    # Matriz inicial aleatoria
    grid = np.random.choice([0, 1], size=(N, N), p=[0.8, 0.2]).astype(np.int32)
    grid_dim = ((N + block_dim[0] - 1) // block_dim[0], (N + block_dim[1] - 1) // block_dim[1])
    
    # Transferencia inicial CPU (Host) -> GPU (Device VRAM)
    d_grid = cuda.to_device(grid)
    d_next = cuda.device_array_like(d_grid)
    
    # Warmup (para compilar el kernel en tiempo de ejecución)
    game_of_life_gpu[grid_dim, block_dim](d_grid, d_next, N)
    cuda.synchronize()
    
    # Medición de las 10 generaciones en GPU
    t0 = time.time()
    for gen in range(generaciones):
        # Buffer-swapping para evitar race conditions
        if gen % 2 == 0:
            game_of_life_gpu[grid_dim, block_dim](d_grid, d_next, N)
        else:
            game_of_life_gpu[grid_dim, block_dim](d_next, d_grid, N)
    cuda.synchronize()
    t_gpu = time.time() - t0
    
    ms_por_gen = (t_gpu / generaciones) * 1000
    print(f"Grilla {N:4d}x{N:4d} | Tiempo GPU ({generaciones} gen): {t_gpu:.4f} s | Promedio: {ms_por_gen:.2f} ms/gen")
    