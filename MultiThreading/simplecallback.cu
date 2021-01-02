/*
This example demonstrate how to use multi thread(on host) in cooperated with cuda stream
It's strange here
Using std::thread and pthread_mutex_t/p_thread_cond_t 
*/
#include "Condition.h"
#include "MutexLockGuard.h"
#include "MutexLock.h"
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>


MutexLock mutex;
Condition cond(mutex);
int finished_threads = 0;
int total_threads = 8;

//GPU tasks
__global__ void gpuTask(){

    //do task
    if(threadIdx.x == 0){
        printf("I'm kernel\n");
    }
}

void postprocess(int id){
    {
        MutexLockGuard lock(mutex);
        finished_threads++;
        if(finished_threads == total_threads){
            cond.notify();
        }
    } 
    printf("#ThreadID %d, I have notied\n",id);
}

void myStreamCallback(cudaStream_t event, cudaError_t status, void *data){
    long id =(long ) data;
    printf("#KernelID %d I'm back from gpu, continue postprocess\n", id);
    
    //start a thread for every kernel finishing
    std::thread tmp(postprocess,id);
    tmp.detach();
    
}

void kernelLanuch(int id){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    gpuTask<<<1, 128,0,stream>>>();

    //set callback
    cudaStreamAddCallback(stream, myStreamCallback, (void*) id, 0);

    //pass task to gpu and return
    return;
}

void testThreadFunc(int id){
    printf("#ThreaID %d, I'm working\n", id);
}


int main(){

    //creat 8 thread and run the same function
    for(int i=0;i<8;i++){
        std::thread tmp(kernelLanuch,i);
        tmp.detach();
    }

    //wait for all thread finish
    MutexLockGuard lock(mutex);
    while(finished_threads < total_threads){
        cond.wait();
    }
    
    printf("\n");
    printf("#Main threads, I have finished!\n");
}





