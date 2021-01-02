#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000000

__global__ void gpuAdd(int *d_a, int *d_b, int *d_c){
    //总线程id = 当前块线程id.x + 块id*块维度x
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		d_c[tid] = d_a[tid] + d_b[tid];// 加法
		tid += blockDim.x * gridDim.x; // 一次执行一个格子 块维度x*格子维度x
	}
}

int main(){
    int *h_a, *h_b, *h_c;
    int *d_a0, *d_b0, *d_c0;
    int *d_a1, *d_b1, *d_c1;
    int sizeByte = N*sizeof(int)*2;

    //create two streams
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);

    //Use cudaHostMalloc to allocate page-locked memory
    cudaHostAlloc(&h_a, sizeByte, cudaHostAllocDefault);
    cudaHostAlloc(&h_b, sizeByte, cudaHostAllocDefault);
    cudaHostAlloc(&h_c, sizeByte, cudaHostAllocDefault);
    //same as single stream, just add stream parameter when launching kernel
    for(int i=0;i<N*2;i++){
        h_a[i] = i;
        h_b[i] = i;
    }

    cudaMalloc(&d_a0,sizeByte/2);
    cudaMalloc(&d_b0,sizeByte/2);
    cudaMalloc(&d_c0,sizeByte/2);
    cudaMalloc(&d_a1,sizeByte/2);
    cudaMalloc(&d_b1,sizeByte/2);
    cudaMalloc(&d_c1,sizeByte/2);

    cudaMemcpyAsync(d_a0, h_a, sizeByte/2, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_a1, h_a + N, sizeByte/2, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_b0, h_b, sizeByte/2, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_b1, h_b + N, sizeByte/2, cudaMemcpyHostToDevice, stream1);

    gpuAdd<<<512,512,0,stream0>>>(d_a0,d_b0,d_c0);
    gpuAdd<<<512,512,0,stream1>>>(d_a1,d_b1,d_c1);

    cudaMemcpyAsync(h_c, d_c0, sizeByte/2, cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(h_c+N, d_c1, sizeByte/2, cudaMemcpyDeviceToHost, stream1);

    //only synchronize on cpu/host
    cudaDeviceSynchronize();
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time consumption: %lf\n", time);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    int Correct = 1;
    int wrongIndex = -1;
	printf("Vector addition on GPU \n");
	//Printing result on console
	for (int i = 0; i < 2*N; i++) 
	{
		if ((h_a[i] + h_b[i] != h_c[i]))
		{
            Correct = 0;
            wrongIndex = i;
            break;
		}

	}
	if (Correct == 1)
	{
		printf("GPU has computed Sum Correctly\n");
	}
	else
	{
        printf("There is an Error in GPU Computation, at index %d, CPU(%d)!=GPU(%d)\n", wrongIndex, (h_a[wrongIndex] + h_b[wrongIndex]), h_c[wrongIndex]);
	}

    // 清空GPU内存
	cudaFree(d_a0);
	cudaFree(d_b0);
	cudaFree(d_c0);
	cudaFree(d_a0);
	cudaFree(d_b0);
	cudaFree(d_c0);
	
	// 清空cuda分配的cpu内存
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	return 0;
}