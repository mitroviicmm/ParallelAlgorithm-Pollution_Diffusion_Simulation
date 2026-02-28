__global__ void apply_sources_kernel(float* conc, const unsigned char* cell_type, int H, int W, float source_value, float absorb_value)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    unsigned char t = cell_type[idx];

    if (t == 1)
    {
        conc[idx] = source_value;
    }
    else if (t == 2)
    {
        float v = conc[idx] - absorb_value;
        conc[idx] = (v > 0.0f) ? v : 0.0f;
    }
}
