#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <memory.h>
int main(void){
	cudaDeviceProp prop;
	int devCount,devID;
	cudaGetDeviceCount(&devCount);
	printf("CUDA devices:\n\t%d\n",devCount);

	cudaGetDevice(&devID);
	printf("ID of current CUDA device:\n\t%d\n",devID);

	memset(&prop, 0, sizeof(cudaDeviceProp));

	cudaGetDeviceProperties(&prop, devID);

	printf("CC of current device is\n\t%d\n",prop.major);

	return 0;
}


