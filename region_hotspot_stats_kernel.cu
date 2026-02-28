__global__ void region_hotspot_stats_kernel(
    const int* danger_steps,
    int H, int W,
    int R, int C,
    float* region_means,
    float* region_stds,
    unsigned char* is_hotspot,
    float alpha
)
{
    int region_x = (int)blockIdx.x;
    int region_y = (int)blockIdx.y;

    int lx = (int)threadIdx.x;
    int ly = (int)threadIdx.y;

    if (lx >= C || ly >= R) return;

    int N = R * C;
    int t = ly * C + lx;

    int gx = region_x * C + lx;
    int gy = region_y * R + ly;
    int idx = gy * W + gx;

    extern __shared__ unsigned char shared_memory[];
    int* s_vals  = (int*)shared_memory;
    float* s_temp  = (float*)(s_vals + N);
    float* s_stats = (float*)(s_temp + N);

    int v = danger_steps[idx];
    s_vals[t] = v;
    s_temp[t] = (float)v;

    __syncthreads();

    int active = N;
    while (active > 1) {
        int half = (active + 1) >> 1;
        if (t < half) {
            int j = t + half;
            if (j < active) s_temp[t] += s_temp[j];
        }
        __syncthreads();
        active = half;
    }


    if (t == 0)
        s_stats[0] = s_temp[0] / (float)N;

    __syncthreads();
    float mean = s_stats[0];

    float diff = ((float)s_vals[t]) - mean;
    s_temp[t] = diff * diff;

    __syncthreads();

    active = N;
    while (active > 1) {
        int half = (active + 1) >> 1;
        if (t < half) {
            int j = t + half;
            if (j < active) s_temp[t] += s_temp[j];
        }
        __syncthreads();
        active = half;
    }


    if (t == 0)
    {
        float var = s_temp[0] / (float)N;
        float sigma = __fsqrt_rn(var);

        int regions_x = W / C;
        int region_id = region_y * regions_x + region_x;

        region_means[region_id] = mean;
        region_stds[region_id]  = sigma;

        s_stats[1] = sigma;
    }

    __syncthreads();
    float sigma = s_stats[1];

    float thr = mean + alpha * sigma;
    is_hotspot[idx] = (((float)s_vals[t]) > thr) ? (unsigned char)1 : (unsigned char)0;
}
