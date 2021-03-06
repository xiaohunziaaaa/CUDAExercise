#include "add.h"

__global__ void gpuAdd_scaler(int* d_a, int* d_b, int* d_c){
	*d_c = *d_a + *d_b;
}

__global__ void gpuAdd_vector(int* d_a, int* d_b, int* d_c, int size){
	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	int id = blockId*blockDim.x + threadId;
	if(id < size){
		d_c[id] = d_a[id] + d_c[id];
	}
}


int Add_scaler(int h_a, int h_b){
	printf("Start!");
	cudaInfo();


	int h_c;

	int *d_a, *d_b, *d_c;



	//allocate memory for variables on gpu
	cudaMalloc((void**)&d_a, sizeof(int));
	cudaMalloc((void**)&d_b, sizeof(int));
	cudaMalloc((void**)&d_c, sizeof(int));

	//memory copy: from cpu to gpu
	cudaMemcpy(d_a,&h_a,sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,&h_b,sizeof(int),cudaMemcpyHostToDevice);

	//call kernel
	gpuAdd_scaler<<<1,1>>>(d_a,d_b,d_c);

	//memory copy: from gpu to cpu
	cudaMemcpy(&h_c,d_c,sizeof(int),cudaMemcpyDeviceToHost);

	printf("Passing paramerter by reference output:%d + %d =%d", h_a,h_b,h_c);

	//free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return h_c;

}

int* Add_vector(int* _h_a, int* _h_b, int size){
	printf("Start!");
	cudaInfo();

	int* h_c = (int*) malloc(size*sizeof(int));

	int* d_a, *d_b, *d_c;

	//allcoate memory on gpu
	cudaMalloc((void**)&d_a, size*sizeof(int));
	cudaMalloc((void**)&d_b, size*sizeof(int));
	cudaMalloc((void**)&d_c, size*sizeof(int));

	//memory copy: from cpu to gpu
	cudaMemcpy(d_a, _h_a, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, _h_b, size*sizeof(int), cudaMemcpyHostToDevice);

	gpuAdd_vector<<<2,8>>>(d_a, d_b, d_c, size);

	//copy result: from gpu to cpu
	cudaMemcpy(h_c, d_c, size*sizeof(int), cudaMemcpyDeviceToHost);

	printf("Passing paramerter by reference output:%d + %d =%d", _h_a[0],_h_b[0],h_c[0]);

	return h_c;
}
