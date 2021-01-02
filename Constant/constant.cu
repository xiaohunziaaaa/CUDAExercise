#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define NUM 100

__constant__ int constant_f;
__constant__ int constant_g;


__global__ void gpuAtomicAdd(float* d_a){    
    int threadId = blockDim.x*blockIdx.x + threadIdx.x;
    extern __shared__ float sh_arr[];                   //just for __shared__ memory practice
    if(threadId<NUM){
        sh_arr[threadId] = d_a[threadId];
        //it seems no necessary for __syncthreads()
        atomicAdd(&sh_arr[threadId], 10);
        d_a[threadId] = sh_arr[threadId];
        
    }
    return;
}

__global__ void gpuMemConst(float* d_a){
    int threadId = blockDim.x*blockIdx.x + threadIdx.x;
    if(threadId<NUM){
        d_a[threadId] = constant_f*d_a[threadId] + constant_g;
    }
}


int main(){
    float* h_a = (float*) malloc(NUM*sizeof(float));
    for(int i=0;i<NUM;i++){
        h_a[i] = i;
    }

    float* res_a = (float*) malloc(NUM*sizeof(float));

    //device mem
    float* d_a;
    cudaMalloc(&d_a, NUM*sizeof(float));

    //mem copy: host -> device
    cudaMemcpy(d_a, h_a, NUM*sizeof(float), cudaMemcpyHostToDevice);

    int f = 3;
    int g = 2;
    //mem copy: host -> device; constant memory
    cudaMemcpyToSymbol(constant_f, &f, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_g, &g, sizeof(int));

    //call kernel
    gpuMemConst<<<2,64>>>(d_a);

    //rew copy: device -> host
    cudaMemcpy(res_a, d_a, NUM*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0;i<NUM;i++){
        if((res_a[i] - h_a[i])!=10){
            printf("res_a - h_a = %lf\n", (res_a[i]-h_a[i]));
        }
    }

    cudaFree(d_a);

    printf("Finish!\n");
}