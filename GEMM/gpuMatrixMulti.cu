#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE 400
// two dimension
#define Blocks 20        
#define threadPerBlock 20
//For practicing, only consider 16*16 matrix

__global__ void gpuMM_noshared(float* d_a, float* d_b, float* d_res){
    //each thread is responsible for one element of res matrix

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.x;

    for(int k=0;k<SIZE;k++){
        d_res[row * SIZE + col] += d_a[row * SIZE + k] * d_b[k * SIZE + col];
    }
}


__global__ void gpuMM_shared(float* d_a, float* d_b, float* d_res){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.x;

	__shared__ float tmp_a[20][20];
    __shared__ float tmp_b[20][20];
    
    for(int k=0;k<SIZE/threadPerBlock;k++){
        tmp_a[threadIdx.x][threadIdx.y] = d_a[row*SIZE + k * threadPerBlock + threadIdx.y];
        tmp_b[threadIdx.x][threadIdx.y] = d_b[(k*threadPerBlock + threadIdx.x)*SIZE + col];

        __syncthreads();

        for(int j=0;j<SIZE/Blocks;j++){
            d_res[row*SIZE + col] += tmp_a[threadIdx.x][j] * tmp_b[j][threadIdx.y];
        }
        __syncthreads();
    }
}

void cpuMM(float* a, float* b, float* c){
    for(int i=0;i<SIZE;i++){
        for(int j=0;j<SIZE;j++){
            for(int k=0;k<SIZE;k++){
                c[i*SIZE + j] += a[i* SIZE + k] * b[k * SIZE + j];
            }
        }
    }
}


int main(){
    int sizeByte = SIZE * SIZE * sizeof(float);
    float* h_a = (float* ) malloc(sizeByte);
    float* h_b = (float* ) malloc(sizeByte);
    float* h_c_cpu = (float* ) malloc(sizeByte);
    float* h_c_gpu = (float* ) malloc(sizeByte);

    float time = 0;

    for(int i=0;i<SIZE;i++){
        for(int j=0;j<SIZE;j++){
            h_a[i*SIZE+j] = (i+j)/2+200;
            h_b[i*SIZE+j] = (i+j)/2+200;
            h_c_cpu[i*SIZE+j] = 0;
            h_c_gpu[i*SIZE+j] = 0;
        }
    }

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    cpuMM(h_a,h_b,h_c_cpu);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Running on CPU, time elapsed...%lf\n",time);
    
    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, sizeByte);
    cudaMalloc(&d_b, sizeByte);
    cudaMalloc(&d_c, sizeByte);

    cudaMemcpy(d_a, h_a, sizeByte, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeByte, cudaMemcpyHostToDevice);

    dim3 Grid(20,20,1);
    dim3 Block(20,20,1);

    cudaEventRecord(start,0);

    gpuMM_shared<<<Grid,Block>>>(d_a,d_b,d_c);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Running on GPU(noshared), time elapsed...%lf\n",time);

    cudaMemcpy(h_c_gpu, d_c, sizeByte, cudaMemcpyDeviceToHost);

    for(int i=0;i<SIZE*SIZE;i++){
        if(h_c_cpu[i] != h_c_gpu[i]){
            printf("Error, CPU(%lf) != GPU(%lf) at index %d\n", h_c_cpu[i], h_c_gpu[i], i);
            return -1;
        }
    }

    return 0;

}
