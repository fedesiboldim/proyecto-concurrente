import numpy as np
import time
from multiprocessing import Pool, cpu_count

# 1. Función secuencial (1 solo hilo / núcleo)
def paso_secuencial(grid, N):
    next_grid = np.zeros((N, N), dtype=np.int32)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            vecinos = (grid[i-1, j-1] + grid[i-1, j] + grid[i-1, j+1] +
                       grid[i,   j-1]                + grid[i,   j+1] +
                       grid[i+1, j-1] + grid[i+1, j] + grid[i+1, j+1])
            if grid[i, j] == 1:
                next_grid[i, j] = 1 if (vecinos == 2 or vecinos == 3) else 0
            else:
                next_grid[i, j] = 1 if vecinos == 3 else 0
    return next_grid

# Función auxiliar para repartir franjas de filas entre núcleos
def procesar_franja(args):
    i_start, i_end, grid, N = args
    sub_next = np.zeros((i_end - i_start, N), dtype=np.int32)
    for i in range(i_start, i_end):
        for j in range(1, N - 1):
            vecinos = (grid[i-1, j-1] + grid[i-1, j] + grid[i-1, j+1] +
                       grid[i,   j-1]                + grid[i,   j+1] +
                       grid[i+1, j-1] + grid[i+1, j] + grid[i+1, j+1])
            if grid[i, j] == 1:
                sub_next[i - i_start, j] = 1 if (vecinos == 2 or vecinos == 3) else 0
            else:
                sub_next[i - i_start, j] = 1 if vecinos == 3 else 0
    return sub_next

# 2. Función paralela en CPU (Multiprocessing)
def paso_paralelo(grid, N, pool, num_workers):
    paso = (N - 2) // num_workers
    tareas = []
    for w in range(num_workers):
        ini = 1 + w * paso
        fin = (N - 1) if w == num_workers - 1 else ini + paso
        tareas.append((ini, fin, grid, N))
    
    resultados = pool.map(procesar_franja, tareas)
    
    # Armamos la matriz completa resultante
    next_grid = np.zeros((N, N), dtype=np.int32)
    offset = 1
    for sub in resultados:
        filas = sub.shape[0]
        next_grid[offset:offset+filas, :] = sub
        offset += filas
    return next_grid

if __name__ == '__main__':
    nucleos = cpu_count()
    print("==================================================")
    print(f"HITO 1: Juego de la Vida en CPU")
    print(f"Nucleos logicos detectados en tu procesador: {nucleos}")
    print("==================================================")

    # Probamos con 512x512 y 1024x1024 para no colgar la máquina
    tamanos = [512, 1024]
    generaciones = 3

    with Pool(processes=nucleos) as pool:
        for N in tamanos:
            # Matriz inicial aleatoria (20% vivas)
            grid = np.random.choice([0, 1], size=(N, N), p=[0.8, 0.2]).astype(np.int32)
            
            # --- Test Secuencial ---
            t0 = time.time()
            g_sec = grid.copy()
            for _ in range(generaciones):
                g_sec = paso_secuencial(g_sec, N)
            t_sec = time.time() - t0

            # --- Test Paralelo en CPU ---
            t0 = time.time()
            g_par = grid.copy()
            for _ in range(generaciones):
                g_par = paso_paralelo(g_par, N, pool, nucleos)
            t_par = time.time() - t0

            # Métricas
            speedup = t_sec / t_par
            eficiencia = speedup / nucleos

            print(f"\n[Grilla {N}x{N} | {generaciones} Generaciones]")
            print(f"  Tiempo Secuencial: {t_sec:.4f} s")
            print(f"  Tiempo Paralelo:   {t_par:.4f} s")
            print(f"  Speedup:           {speedup:.2f}x")
            print(f"  Eficiencia:        {eficiencia:.2f}")
    
    print("\n¡Prueba completada con exito!")
    