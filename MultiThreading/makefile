TARGET = simplecallback
SRC = simplecallback.cu


all:$(TARGET)

$(TARGET):$(SRC)
	/usr/local/cuda-10.1/bin/nvcc --cudart static  --std=c++11 --relocatable-device-code=false -gencode arch=compute_75,code=sm_75 -o $@ $< -lpthread

clean:
	rm -f $(TARGET)