Pollution Diffusion Simulation and Risk Zone Analysis (CUDA / PyCUDA Project)
Parallel Computing Project

Implemented a GPU-accelerated simulation of pollution diffusion on a 2D grid using CUDA and PyCUDA.

The model simulates pollution diffusion between neighboring cells, natural decay of concentration, and the effects of pollution sources and cleaning zones.

Developed multiple CUDA kernels for:

pollution diffusion simulation

applying pollution sources and absorption zones

tracking the number of time steps each cell spends in a dangerous pollution state

detecting cells that exceeded the danger threshold at least once.

Implemented regional analysis where the grid is divided into regions and for each region the mean and standard deviation of time spent in the danger zone are computed.

Used these statistics to detect local pollution hotspots based on a threshold defined by mean + standard deviation.

The project includes parallel processing of large matrices (up to 2048×2048) and evaluation of GPU execution performance.

Technologies: CUDA, PyCUDA, Python, GPU Computing, Parallel Algorithms
