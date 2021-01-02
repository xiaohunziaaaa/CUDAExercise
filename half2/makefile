TARGET = half2
SRC = half2.cu 
HOST_COMPILER = g++ 
all:$(TARGET)


$(TARGET):$(SRC)
	/usr/local/cuda-10.1/bin/nvcc --cudart static  --relocatable-device-code=false -gencode arch=compute_75,code=sm_75 -o $@ $<

clean:
	rm -f $(TARGET)