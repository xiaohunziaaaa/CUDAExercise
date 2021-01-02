#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "sharedMem.h"
#define VECTORSIZE 1024
#define threadPerBlock 128
//do fp16 dot

//function called by gpuDot
__forceinline__ __device__ void dot(half2* shrres){
    int stride = threadPerBlock/2;
    for(int i=threadPerBlock/2; i>0; i/=2){
        if(threadIdx.x < i){
            shrres[threadIdx.x] = __hadd2(shrres[threadIdx.x], shrres[threadIdx.x + stride]);
            //shrres[threadIdx.x] = shrres[threadIdx.x];
            __syncthreads();
            stride/=2;
        }
    }
}

/*
one block was responsible for threadPerBlock*2 of dot operation
part dot result was store into device memory
and accumlate results in device memory in host not in device
another world, pass device memory to host

shared memory was used for store elements of a/b temporary

@half2* a: input vector A
@half2* b: input vector B
@half2* c: output result
*/
__global__ void gpuDot(half2* a, half2*b , half2* c){
    int threadId = threadIdx.x;
    int gId = threadIdx.x + blockIdx.x * blockDim.x;

    //dynamic shared memory
    //extern __shared__ half2 tmp[];

    //static shared memory
    //__shared__ half2 tmp[threadPerBlock];

    //using template 
    SharedArr<half2> shr;
    half2 *tmp = shr.getPointer();

    tmp[threadId] = __hmul2(a[gId], b[gId]);

    __syncthreads();
    // for debug
    //if(blockIdx.x == 0){
    //    printf("%lf * %lf = %lf \n",__low2float(a[gId]),__low2float(b[gId]), __low2float(tmp[threadId]));
    //}
    if(blockIdx.x == 0 && threadIdx.x==0){
        printf("before %lf\n", __low2float(tmp[threadId]));
    }
    
    dot(tmp);

    // for debug
    if(blockIdx.x == 0 && threadIdx.x==0){
        printf("after %lf\n", __low2float(tmp[threadId]));
    }

    if(threadId==0 ){
        c[blockIdx.x] = tmp[threadId];
    }
}

void init(half2* arr, int size){
    for(int i=0;i<size;i++){
        half2 temp;
        temp.x = static_cast<float>(rand() % 4);
        temp.y = static_cast<float>(rand() % 4);
        arr[i] = temp;
    }
}
int main(){
    const int SIZEVECTOR_FP16 = VECTORSIZE/2;                        //one half2 struct store two FP16 number
    const int SIZEBYTE = SIZEVECTOR_FP16 * sizeof(half2);
    const int BLOCKS = VECTORSIZE/threadPerBlock/2;

    half2 *h_a = (half2*) malloc(SIZEBYTE);
    half2 *h_b = (half2*) malloc(SIZEBYTE);
    half2 *h_c = (half2*) malloc(BLOCKS*sizeof(half2));
    half2 *d_a, *d_b, *d_c;

    
    init(h_a, SIZEVECTOR_FP16);
    init(h_b, SIZEVECTOR_FP16);


    cudaMalloc((void**) &d_a, SIZEBYTE);
    cudaMalloc((void**) &d_b, SIZEBYTE);
    cudaMalloc((void**) &d_c, BLOCKS*sizeof(half2));

    cudaMemcpy(d_a, h_a, SIZEBYTE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZEBYTE, cudaMemcpyHostToDevice);
    //dynamic shared memory
    //gpuDot<<<BLOCKS, threadPerBlock, threadPerBlock>>>(d_a, d_b, d_c);

    //static shared memory
    //gpuDot<<<BLOCKS, threadPerBlock>>>(d_a, d_b, d_c);

    //using template
    gpuDot<<<BLOCKS, threadPerBlock, threadPerBlock * sizeof(half2)>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, BLOCKS*sizeof(half2), cudaMemcpyDeviceToHost);

    float res = 0.0;

    for(int i=0;i<BLOCKS;i++){
        float tmp = __high2float(h_c[i]) + __low2float(h_c[i]);
        res += tmp;
    }

    
    float res_ref = 0.0;
    for(int i=0;i<SIZEVECTOR_FP16;i++){
        float tmp = __high2float(h_a[i]) * __high2float(h_b[i]) + __low2float(h_a[i])*__low2float(h_b[i]);
        res_ref += tmp;
    }

    printf("GPU res(%lf) vs. CPU res(%lf)\n", res, res_ref);

    return 0;
}