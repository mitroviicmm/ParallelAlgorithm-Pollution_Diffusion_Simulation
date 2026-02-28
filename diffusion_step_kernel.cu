__global__ void diffusion_step_kernel(const float* conc_in, float* conc_out, int H, int W, float decay)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int idx = y * W + x;

    float center = conc_in[idx];
    float up    = (y > 0)     ? conc_in[(y - 1) * W + x] : 0.0f;
    float down  = (y + 1 < H) ? conc_in[(y + 1) * W + x] : 0.0f;
    float left  = (x > 0)     ? conc_in[y * W + (x - 1)] : 0.0f;
    float right = (x + 1 < W) ? conc_in[y * W + (x + 1)] : 0.0f;

    float avg = (center + up + down + left + right) * 0.2f;
    conc_out[idx] = avg * (1.0f - decay);
}
