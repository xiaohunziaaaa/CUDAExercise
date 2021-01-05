// allocate pitch memory and cudaArray

#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NX 1003
#define NY 1003


int main(){

    size_t sizeByte = NX*NY*sizeof(float);
    //host data declaration and initialization
    float* hdata = (float* )malloc(sizeByte);
    for(int i=0;i<NX*NY; i++){
        hdata[i] = i;
    }


    //using pitch linear memory
    float *ddata_pl;
    float *ddata_pl_res;
    size_t sizePL;

    cudaMallocPitch((void**)&ddata_pl, &sizePL, NX*sizeof(float), NY);

    printf("Pitch of ddata_pl is %d \n", sizePL);
    printf("While Pitch of hdata is %d \n", NX*sizeof(float));

    return 0;

}
