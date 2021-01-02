#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#define N 1024


__global__ void gpu_sort(int *d_a, int *d_b){

    __shared__ int tmp[512];
    
    int tid = threadIdx.x;
    int ttid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = d_a[ttid];
    int count =0;
    for(int i=tid;i<N;i+=512){
        tmp[tid] = d_a[i];
        __syncthreads();
        for(int j=0;j<512;j++){
            if(val>tmp[j]){
                count++;
            }
        }
        __syncthreads();
    }

    d_b[count] = val;

}


int main(){
    int sizeByte = sizeof(int)*N;
    int *h_a = (int*) malloc(sizeByte);
    int *h_b = (int*) malloc(sizeByte);
    int *h_a_cpu = (int*) malloc(sizeByte);
    int *h_b_cpu = (int*) malloc(sizeByte);
 

    int *d_a, *d_b;
    cudaMalloc(&d_a, sizeByte);
    cudaMalloc(&d_b, sizeByte);

    for(int i=0;i<N;i++){
        h_a[i] = rand();
        h_a_cpu[i] = h_a[i];
    }

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    
    cudaMemcpy(d_a, h_a, sizeByte, cudaMemcpyHostToDevice);

    gpu_sort<<<2, 512>>>(d_a,d_b);


    cudaMemcpy(h_b, d_b, sizeByte, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf("Time consumption on GPU: %lf\n", time);

    for(int i=0;i<N-1;i++){
        if(h_b[i]>h_b[i+1]){
            printf("Error at index %d\n GPU[%d] = %d\n", i,i,h_b[i]);
            break;
        }
    }

    cudaEvent_t start_cpu,stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu,0);
    //sort on cpu
    for(int i=N;i>0;i--){
        for(int j=0;j<i-1;j++){
            if(h_a_cpu[j]>h_a_cpu[j+1]){
                int tmp = h_a_cpu[j];
                h_a_cpu[j] = h_a_cpu[j + 1];
                h_a_cpu[j+1] = tmp; 
            }
        }
    }
    cudaEventRecord(stop_cpu,0);
    cudaEventSynchronize(stop_cpu);
    float time_cpu = 0;
    cudaEventElapsedTime(&time_cpu, start_cpu, stop_cpu);
    printf("Time consumption on CPU: %lf\n", time_cpu);

    return 0;

}
