// allocate pitch memory and cudaArray

#include <stdio.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NX 1003
#define NY 1024


__global__ void PLShift(float* odata, int pitch, int width, int height, int shiftx, int shifty, cudaTextureObject_t texRefPL ){
    int global_x_ = threadIdx.x + blockIdx.x*blockDim.x;
    int global_y_ = threadIdx.y + blockIdx.y*blockDim.y;

    /*
    if(threadIdx.x == 0 && blockIdx.x == 0){
        for(int i=0; i<NX; i++){
            for(int j=0;j<NY;j++){
                printf("%d ", tex2D<float>(texRefPL,
                    (i) / (float) width,
                    (j) / (float) height))
            }
            printf("\n");
        }
    }
    */


    //odata[global_y_*pitch + global_x_] = tex2D<float>(texRefPL,
    //                                 (global_x_) / (float) width,
    //                                 (global_y_) / (float) height);        //because using normalizedCoords and cudaAddressModeWrap(automaticaly loop), Line 78/80/81
    odata[global_y_*pitch + global_x_] = tex2D<float>(texRefPL,
                                    (global_x_ + shiftx)%width,
                                     (global_y_ + shifty)%height);
}


int main(){

    size_t sizeByte = NX*NY*sizeof(float);
    //host data declaration and initialization
    float* hdata = (float* )malloc(sizeByte);
    for(int i=0;i<NX*NY; i++){
        hdata[i] = i;
    }
    float* hdata_res = (float* )malloc(sizeByte);
    memset(hdata_res, 0, sizeByte);
    //gloden result
    float *gold = (float*) malloc(sizeByte);
    for (int j = 0; j < NY; ++j)
    {
        int jshift = (j + 2) % NY;

        for (int i = 0; i < NX; ++i)
        {
            int ishift = (i + 1) % NX;
            gold[j * NX + i] = hdata[jshift * NX + ishift];
        }
    }

    //cudaChannelDesc, similar to data type
     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    //using pitch linear memory
    float *ddata_pl;
    float *ddata_pl_res;
    size_t sizePL;

    cudaMallocPitch((void**)&ddata_pl, &sizePL, NX*sizeof(float), NY);
    cudaMallocPitch((void**)&ddata_pl_res, &sizePL, NX*sizeof(float), NY);

    printf("Pitch of ddata_pl is %d , while NX =%d\n", sizePL, NX);

    //memory copy: host -> device
    cudaMemcpy2D(ddata_pl, sizePL, hdata, NX*sizeof(float),NX*sizeof(float), NY, cudaMemcpyHostToDevice);      //no padding in host memory, so spitch = width

    //allocate texture memory
    //cudaResourceDesc  -->  cudaTextureDesc --> cudaTextureObject_t
    
    //cudaResourceDesc
    cudaResourceDesc texRes;
    memset(&texRes,0,sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = ddata_pl;
    texRes.res.pitch2D.desc = channelDesc;
    texRes.res.pitch2D.width = NX;
    texRes.res.pitch2D.height = NY;
    texRes.res.pitch2D.pitchInBytes = sizePL;       //why not sizePL

    //cudaTextureDesc
    cudaTextureDesc         texDescr;
    memset(&texDescr,0,sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    //create cudaTextureObject
    cudaTextureObject_t         texRefPL;
    cudaCreateTextureObject(&texRefPL, &texRes, &texDescr, NULL);


    //start running kernel
    cudaMemset2D(ddata_pl_res, sizePL, 0, NX*sizeof(float), NY);
    dim3 mygrid(32, 32, 1);
    dim3 myblock(32, 32, 1);
    PLShift<<<mygrid, myblock>>>(ddata_pl_res, sizePL/sizeof(float), NX, NY, 1, 2, texRefPL);

    cudaMemcpy2D(hdata_res, NX*sizeof(float), ddata_pl_res, sizePL, NX*sizeof(float), NY, cudaMemcpyDeviceToHost);

    for(int i=0;i<NX*NY;i++){
        if(hdata_res[i] != gold[i]){
            printf("Error at %d, GPU(%lf) != CPU(%lf)\n",i, hdata_res[i], gold[i]);
            //return 0;
        }
    }
}
