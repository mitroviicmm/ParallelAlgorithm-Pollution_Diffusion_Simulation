from pathlib import Path
import numpy as np
import pycuda.autoinit as autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

BASE_DIR = Path(__file__).resolve().parent

def load_kernel(filename: str) -> str:
    return (BASE_DIR / filename).read_text(encoding="utf-8")

# KERNEL_OPTIONS = ["-O3"]
#
# mod_diff = SourceModule(load_kernel("diffusion_step_kernel.cu"), options=KERNEL_OPTIONS)
# mod_apply = SourceModule(load_kernel("apply_sources_kernel.cu"), options=KERNEL_OPTIONS)
# mod_danger = SourceModule(load_kernel("danger_kernel.cu"), options=KERNEL_OPTIONS)
# mod_region = SourceModule(load_kernel("region_hotspot_stats_kernel.cu"), options=KERNEL_OPTIONS)

mod_diff = SourceModule(load_kernel("diffusion_step_kernel.cu"))
mod_apply = SourceModule(load_kernel("apply_sources_kernel.cu"))
mod_danger = SourceModule(load_kernel("danger_kernel.cu"))
mod_region = SourceModule(load_kernel("region_hotspot_stats_kernel.cu"))

K_DIFF = mod_diff.get_function("diffusion_step_kernel")
K_APPLY = mod_apply.get_function("apply_sources_kernel")
K_COUNT_DANGER = mod_danger.get_function("count_danger_kernel")
K_COUNT_EVER = mod_danger.get_function("count_ever_dangerous_kernel")
K_REGION_STATS = mod_region.get_function("region_hotspot_stats_kernel")

BLOCK_HW = (16, 16, 1)

def make_cell_type(H: int, W: int) -> np.ndarray:
    cell_type = np.zeros((H, W), dtype=np.uint8)
    cell_type[H // 2, W // 2] = 1
    cell_type[1, 1] = 1
    cell_type[H - 2, W - 2] = 1
    if W >= 4:
        cell_type[:, W // 3] = 2
    return cell_type

def step(conc_in, conc_out, cell_gpu, H, W, decay, source_value, absorb_value, grid):
    K_DIFF(
        conc_in.gpudata, conc_out.gpudata,
        np.int32(H), np.int32(W),
        np.float32(decay),
        block=BLOCK_HW, grid=grid
    )
    K_APPLY(
        conc_out.gpudata, cell_gpu.gpudata,
        np.int32(H), np.int32(W),
        np.float32(source_value), np.float32(absorb_value),
        block=BLOCK_HW, grid=grid
    )
    return conc_out, conc_in

def run_case(
    H, W,
    T=1000,
    decay=0.01,
    source_value=1.0,
    absorb_value=0.04,
    danger_threshold=0.01,
    R=16, C=16,
):
    _ = autoinit.context

    if (H % R) != 0 or (W % C) != 0:
        raise ValueError(f"H and W must be divisible by R and C. Got H={H},W={W},R={R},C={C}")
    if R * C > 1024:
        raise ValueError(f"R*C must be <= 1024. Got R*C={R*C}")

    conc0 = np.zeros((H, W), dtype=np.float32)
    cell_type = make_cell_type(H, W)

    conc_a = gpuarray.to_gpu(conc0.ravel())
    conc_b = gpuarray.empty_like(conc_a)
    cell_gpu = gpuarray.to_gpu(cell_type.ravel())

    danger_count = gpuarray.zeros((H * W,), dtype=np.int32)
    danger_counter = gpuarray.zeros((1,), dtype=np.int32)

    grid = ((W + BLOCK_HW[0] - 1) // BLOCK_HW[0], (H + BLOCK_HW[1] - 1) // BLOCK_HW[1], 1)

    start = cuda.Event()
    stop = cuda.Event()

    start.record()
    for _ in range(T):
        conc_a, conc_b = step(
            conc_a, conc_b, cell_gpu,
            H, W, decay, source_value, absorb_value,
            grid
        )
        K_COUNT_DANGER(
            conc_a.gpudata, danger_count.gpudata,
            np.int32(H), np.int32(W),
            np.float32(danger_threshold),
            block=BLOCK_HW, grid=grid
        )
    stop.record()
    stop.synchronize()

    total_ms = float(start.time_till(stop))
    per_step_ms = total_ms / float(T)

    danger_counter.fill(0)
    K_COUNT_EVER(
        danger_count.gpudata,
        np.int32(H), np.int32(W),
        danger_counter.gpudata,
        block=BLOCK_HW, grid=grid
    )
    cuda.Context.synchronize()
    ever_dangerous = int(danger_counter.get()[0])

    conc_final = conc_a.get().reshape(H, W)
    mean_conc = float(conc_final.mean())

    alpha = 1.0

    regions_x = W // C
    regions_y = H // R
    n_regions = regions_x * regions_y

    region_means = gpuarray.zeros((n_regions,), dtype=np.float32)
    region_stds  = gpuarray.zeros((n_regions,), dtype=np.float32)
    is_hotspot   = gpuarray.zeros((H * W,), dtype=np.uint8)

    block_region = (int(C), int(R), 1)
    grid_region = (int(regions_x), int(regions_y), 1)

    N = R * C
    # int[N] + float[N] + 2 float (mean,sigma) + padding
    shared_bytes = N * (4 + 4) + 2 * 4 + 16

    K_REGION_STATS(
        danger_count.gpudata,
        np.int32(H), np.int32(W),
        np.int32(R), np.int32(C),
        region_means.gpudata,
        region_stds.gpudata,
        is_hotspot.gpudata,
        np.float32(alpha),
        block=block_region, grid=grid_region, shared=int(shared_bytes)
    )
    cuda.Context.synchronize()

    region_means_host = region_means.get()
    region_stds_host = region_stds.get()

    hotspot_count = int(is_hotspot.get().sum())

    return {
        "total_ms": total_ms,
        "per_step_ms": per_step_ms,
        "mean_conc": mean_conc,
        "ever_dangerous": ever_dangerous,
        "region_means_min": float(region_means_host.min()),
        "region_means_max": float(region_means_host.max()),
        "region_stds_min": float(region_stds_host.min()),
        "region_stds_max": float(region_stds_host.max()),
        "hotspot_count": hotspot_count
    }

def main():
    cases = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    # cases = [
    #     (10, 10),
    #     (100, 100),
    #     (1000, 1000),
    # ]

    # cases = [(16, 16), (96, 96), (1008, 1008)]

    T = 100
    decay = 0.01
    source_value = 1.0
    absorb_value = 0.04
    danger_threshold = 0.01
    R, C = 16, 16

    print(f"T={T}, decay={decay}, source_value={source_value}, absorb_value={absorb_value}, danger_threshold={danger_threshold}, R={R},C={C}")
    for (H, W) in cases:
        out = run_case(
            H, W,
            T=T, decay=decay,
            source_value=source_value, absorb_value=absorb_value,
            danger_threshold=danger_threshold,
            R=R, C=C
        )
        print(
            f"{H}x{W}: total {out['total_ms']:.3f} ms, per-step {out['per_step_ms']:.6f} ms, "
            f"mean(conc)={out['mean_conc']:.6f}, "
            f"ever-dangerous={out['ever_dangerous']}, "
            f"region_mean[min,max]=({out['region_means_min']:.6f},{out['region_means_max']:.6f}), "
            f"region_std[min,max]=({out['region_stds_min']:.6f},{out['region_stds_max']:.6f}), "
            f"hotspots={out['hotspot_count']}"
        )

if __name__ == "__main__":
    main()
