#include "gpuSharedMem.h"

__global__ void gpu_shared_mem(float* d_a, int size){

	int threadId = threadIdx.x;
	float avr,sum = 0.0;

	//using dynamic shared memory
	extern __shared__ int sh_arr[];

	sh_arr[threadId] = d_a[threadId];

	__syncthreads();

	for(int i=0;i<=threadId;i++){
		sum+=sh_arr[i];
	}
	avr = sum/(threadId + 1);
	d_a[threadId] = avr;
	//__syncthreads();
	sh_arr[threadId] = avr;

	return;
}


void testGpuShrMem(){
	int size = 1025;
	int sizeByte = size*sizeof(float);
	printf("%d", sizeByte);
	float* h_a =(float*) malloc(sizeByte);
	float* h_res = (float*) malloc(sizeByte);
	for(int i=0;i<size;i++){
		h_a[i] = i;
	}

	//allocate gpu mem
	float* d_a;
	cudaMalloc(&d_a, sizeByte);

	//mem copy
	cudaMemcpy(d_a,h_a,sizeByte,cudaMemcpyHostToDevice);

	//issue kernel
	gpu_shared_mem<<<1,size, sizeByte>>>(d_a,size);

	//copy result
	cudaMemcpy(h_res,d_a,sizeByte,cudaMemcpyDeviceToHost);

	printf("Results");
	for(int i=0;i<size;i++){
		printf("%lf ",h_res[i]);
	}

	printf("\n");
}

