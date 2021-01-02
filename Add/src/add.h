#ifndef _ADD_H
#define _ADD_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cudaInit.h"



int Add_scaler(int _h_a, int _h_b);

int* Add_vector(int* _h_a,int *_h_b, int size);

#endif
