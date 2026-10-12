import numpy as np
import time
from numba import cuda

TPB = 16

@cuda.jit
def game_of_life_kernel(d_grid, d_next_grid, N):
    s_tile = cuda.shared.array((18, 18), dtype=np.int32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by = cuda.blockIdx.x, cuda.blockIdx.y
    gx, gy = bx * TPB + tx, by * TPB + ty
    lx, ly = tx + 1, ty + 1

    # Cargar celda central
    s_tile[ly, lx] = d_grid[gy, gx] if (gx < N and gy < N) else 0

    # Cargar halos
    if tx == 0: s_tile[ly, 0] = d_grid[gy, gx - 1] if (gx > 0 and gy < N) else 0
    if tx == TPB - 1: s_tile[ly, 17] = d_grid[gy, gx + 1] if (gx + 1 < N and gy < N) else 0
    if ty == 0: s_tile[0, lx] = d_grid[gy - 1, gx] if (gy > 0 and gx < N) else 0
    if ty == TPB - 1: s_tile[17, lx] = d_grid[gy + 1, gx] if (gy + 1 < N and gx < N) else 0

    # Esquinas
    if tx == 0 and ty == 0: s_tile[0, 0] = d_grid[gy - 1, gx - 1] if (gx > 0 and gy > 0) else 0
    if tx == TPB - 1 and ty == 0: s_tile[0, 17] = d_grid[gy - 1, gx + 1] if (gx + 1 < N and gy > 0) else 0
    if tx == 0 and ty == TPB - 1: s_tile[17, 0] = d_grid[gy + 1, gx - 1] if (gx > 0 and gy + 1 < N) else 0
    if tx == TPB - 1 and ty == TPB - 1: s_tile[17, 17] = d_grid[gy + 1, gx + 1] if (gx + 1 < N and gy + 1 < N) else 0

    cuda.syncthreads()

    if 1 <= gx < N - 1 and 1 <= gy < N - 1:
        v = (s_tile[ly - 1, lx - 1] + s_tile[ly - 1, lx] + s_tile[ly - 1, lx + 1] +
             s_tile[ly,     lx - 1]                      + s_tile[ly,     lx + 1] +
             s_tile[ly + 1, lx - 1] + s_tile[ly + 1, lx] + s_tile[ly + 1, lx + 1])
        d_next_grid[gy, gx] = 1 if ((s_tile[ly, lx] == 1 and (v == 2 or v == 3)) or (s_tile[ly, lx] == 0 and v == 3)) else 0

class GameOfLifeSimulator:
    def __init__(self, N):
        self.N = N
        self.grid_dim = ((N + TPB - 1) // TPB, (N + TPB - 1) // TPB)
        self.block_dim = (TPB, TPB)

    def run(self, generaciones=50):
        # 1. CPU Manager: genera configuración inicial
        h_grid = np.random.choice([0, 1], size=(self.N, self.N), p=[0.8, 0.2]).astype(np.int32)

        # 2. Copia inicial Host to Device
        t_trans_in = time.time()
        d_grid = cuda.to_device(h_grid)
        d_next = cuda.device_array_like(d_grid)
        t_trans_in = time.time() - t_trans_in

        # 3. GPU Worker: simulación masiva sin copias intermedias
        t_sim = time.time()
        for gen in range(generaciones):
            if gen % 2 == 0:
                game_of_life_kernel[self.grid_dim, self.block_dim](d_grid, d_next, self.N)
            else:
                game_of_life_kernel[self.grid_dim, self.block_dim](d_next, d_grid, self.N)
        cuda.synchronize()
        t_sim = time.time() - t_sim

        # 4. Copia final Device to Host
        t_trans_out = time.time()
        h_res = (d_next if generaciones % 2 == 1 else d_grid).copy_to_host()
        t_trans_out = time.time() - t_trans_out

        return t_trans_in, t_sim, t_trans_out

if __name__ == "__main__":
    N = 4096
    gens = 50
    sim = GameOfLifeSimulator(N)
    tin, tcalc, tout = sim.run(gens)
    print(f"Pipeline N={N} ({gens} gen):")
    print(f"  - Transferencia HtoD: {tin*1000:.2f} ms")
    print(f"  - Cómputo puro en GPU: {tcalc*1000:.2f} ms ({tcalc/gens*1000:.2f} ms/gen)")
    print(f"  - Transferencia DtoH: {tout*1000:.2f} ms")
    