#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


__global__ void gpu_shared_mem(float* d_a, int size);
void testGpuShrMem();
