__global__ void count_danger_kernel(
    const float* conc,
    int* danger_count,
    int H,
    int W,
    float danger_threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    int idx = y * W + x;

    if (conc[idx] >= danger_threshold) {
        danger_count[idx] += 1;
    }
}


__global__ void count_ever_dangerous_kernel(
    const int* danger_count,
    int H,
    int W,
    int* danger_counter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H)
        return;

    int idx = y * W + x;

    if (danger_count[idx] > 0) {
        atomicAdd(danger_counter, 1);
    }
}
