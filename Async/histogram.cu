#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define N 512
#define CLUSTER 10

__global__ void histrogram_atomic(int* d_a, int* d_b){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < N)
	{
        int val = d_a[threadId];
        //atomicAdd(&(d_b[val]), 1);// 原子操作+1
        d_b[val]++;
        __syncthreads();
	}
}

__global__ void histogram_shared(int* d_a, int* d_b){
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int shr[CLUSTER];
    shr[threadIdx.x] = 0;
    __syncthreads();

    if(threadId < N){
        atomicAdd(&(shr[d_a[threadId]]), 1);// 原子操作+1
    }

    __syncthreads();
    atomicAdd(&(d_b[threadIdx.x]), shr[threadIdx.x]);
        
}

int main(){
    int sizeByte = N*sizeof(int);
    int* h_a = (int*) malloc(sizeByte);
    int* h_b = (int*) malloc(CLUSTER*sizeof(int));
    int* h_b_shared = (int*) malloc(CLUSTER*sizeof(int));
    int* ref = (int *) malloc(CLUSTER*sizeof(int));
    for(int i=0;i<N;i++){
        h_a[i] = i%CLUSTER;
    }

    for(int i=0;i<CLUSTER;i++){
        h_b[i] = 0;
        h_b_shared[i] = 0;
        ref[i] = 0;
    }
    


    //allocate device memory
    int *d_a, *d_b, *d_b_shared;
    cudaMalloc(&d_a, sizeByte);
    cudaMalloc(&d_b, CLUSTER*sizeof(int));
    cudaMalloc(&d_b_shared, CLUSTER*sizeof(int));

    //Memory copy: host ->> device
    cudaMemcpy(d_a, h_a, sizeByte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, CLUSTER*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_shared, h_b_shared, CLUSTER*sizeof(int), cudaMemcpyHostToDevice);
    

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    histrogram_atomic<<<1,512>>>(d_a,d_b);
    cudaEventRecord(stop, 0);

    //copy result: device ->> host
    cudaMemcpy(h_b, d_b, CLUSTER*sizeof(int), cudaMemcpyDeviceToHost);
    float time_atomic = 0;
    cudaEventElapsedTime(&time_atomic, start, stop);
    printf("Time consumption(Atomic):%lf\n", time_atomic);
    

    //using shared memory 
    cudaEventRecord(start, 0);
    histogram_shared<<<60,10>>>(d_a,d_b_shared);
    cudaEventRecord(stop, 0);

    cudaMemcpy(h_b_shared, d_b_shared, CLUSTER*sizeof(int), cudaMemcpyDeviceToHost);
    float time_shared = 0;
    cudaEventElapsedTime(&time_shared, start, stop);
    printf("Time consumption(Shared):%lf\n", time_shared);


    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    //check
    
    for(int i=0;i<N;i++){
        ref[h_a[i]]++;
    }
    for(int i=0;i<CLUSTER;i++){
        //if(h_b[i]!=ref[i]){
            printf("Ref(%d) Atomic(%d) Shared(%d)\n", ref[i],h_b[i],h_b_shared[i]);
            //return -1;
        //}
    }
    

    cudaFree(d_a);
    cudaFree(d_b);

    free(h_a);
    free(h_b);
    return 0;
}