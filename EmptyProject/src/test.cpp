#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void gpu_shared_memory(float *d_a)
{
	// 当前线程id    局部数据，线程直接不共享
	int i, index = threadIdx.x;
	float average, sum = 0.0f;

	// 共享数据 ，多线程之间 贡献
	// 可以被同一block中的所有线程读写
	__shared__ float sh_arr[10];

	// 共享数据，记录数组数据
	sh_arr[index] = d_a[index];

	__syncthreads();    // 确保共享数据写入完成

	for (i = 0; i<= index; i++)
	{
		sum += sh_arr[i]; // 数组累加
	}
	average = sum / (index + 1.0f);//求平均值

	d_a[index] = average; // 输出值 记录平均值

	sh_arr[index] = average;// 共享数据也被替换
}

int main(int argc, char **argv)
{
	// CPU数据
	float h_a[10];
	// 指向GPU内存数据
	float *d_a;
        // 赋值
	for (int i = 0; i < 10; i++)
	{
		h_a[i] = i;
	}
	// 分配gpu内存
	cudaMalloc((void **)&d_a, sizeof(float) * 10);
	// cpu数据到gpu数据
	cudaMemcpy((void *)d_a, (void *)h_a, sizeof(float) * 10, cudaMemcpyHostToDevice);

        // 一个线程块，10个线程 执行
	gpu_shared_memory <<<1, 10 >>>(d_a);
	// 拷贝结果 gpu  到 cpu
	cudaMemcpy((void *)h_a, (void *)d_a, sizeof(float) * 10, cudaMemcpyDeviceToHost);
	printf("Use of Shared Memory on GPU:  \n");
	//Printing result on console
	for (int i = 0; i < 10; i++) {
		printf("The running average after %d element is %f \n", i, h_a[i]);
	}
	// gpu内存清理
	return 0;
}
