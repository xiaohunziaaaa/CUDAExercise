#ifndef __SHAREDMEM_H__
#define __SHAREDMEM_H__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
 
template <class T>
struct SharedArr
{
    // Ensure that we won't compile any un-specialized types
    __device__ T *getPointer()
    {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

//this is called template specilizaiton
template<>
struct  SharedArr<float>
{
    __device__ float *getPointer(){
        extern __shared__ float arr_float[];
        return arr_float;
    }
};

template<>
struct  SharedArr<int>
{
    __device__ int *getPointer(){
        extern __shared__ int arr_int[];
        return arr_int;
    }
};

template<>
struct  SharedArr<half2>
{
    __device__ half2 *getPointer(){
        extern __shared__ half2 arr_half2[];
        return arr_half2;
    }
};


template<>
struct  SharedArr<half>
{
    __device__ half *getPointer(){
        extern __shared__ half arr_half[];
        return arr_half;
    }
};

#endif