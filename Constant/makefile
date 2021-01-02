TARGET = simpleTexture
SRC = simpleTexture.cu

all:simpleTexture

simpleTexture:$(SRC)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/usr/local/cuda-10.1/bin/nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_75,code=compute_75 -gencode arch=compute_75,code=sm_75 -link -o "$@" "$<"
	@echo 'Finished building target: $@'
	@echo ' '
