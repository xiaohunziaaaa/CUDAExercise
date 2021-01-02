#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void transformKernel(float *outputData, int width, int height, float theta, cudaTextureObject_t tex){
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float u = (float)x - (float)width/2; 
    float v = (float)y - (float)height/2; 
    float tu = u*cosf(theta) - v*sinf(theta); 
    float tv = v*cosf(theta) + u*sinf(theta); 

    tu /= (float)width; 
    tv /= (float)height; 

    // read from texture and write to global memory
    outputData[y*width + x] = tex2D<float>(tex, tu+0.5f, tv+0.5f);
}

void loadPGM(float* _imageData, const char* _imagePath, int* width, int* height){
    for(int i=0;i<256;i++){
        _imageData[i] = i+0.0; 
    }
    *width = 16;
    *height = 16;
    return;
}

int main(){
    /**
        1. load image and reference image
        2. allocate device memory for result
        3. [most important] allocate array and copy image data, then bind it to texture memory (as well as copy host data to device)
        4. issure kernel 
        5. copy result of device to host
        6. do post process
    */

    int width = 0;
    int height = 0;
    int size = 256*sizeof(float);
    const float angle = 0.5f; 
    // load image
    float* h_imageData = (float*) malloc(256*sizeof(float));
    const char* imagePath = "./test.pgm";
    loadPGM(h_imageData, imagePath, &width, &height);

    

    // load reference image
    //float* h_refData;
    //const char* refPath = "./testRef.pgm";
    //loadPGM(h_refData, refPath, );


    float* h_resData = (float*) malloc(256*sizeof(float));

    // allocate device memory for res, inorder to copy result from gpu to cpu.
    float* d_resData;
    cudaMalloc((void **) &d_resData, size);
    // allocate arry and copy image data, bind to texture memory 

    //first cudaChannelDesc
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); //low level, C style
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();                                         //high level, C++ style, using template
    
    //second, create cudaArray used for texture memory and fill it with h_imageData
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, h_imageData, size, cudaMemcpyHostToDevice);

    //third, create cudaResourceDesc (with cudaArray) and cudaTextureDesc
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = cuArray;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeElementType;

    // fourth, create cudaTextureObject
    cudaTextureObject_t tex;            //aka cudaTextureObject*
    cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);                //Bind cudaArray(imageData) to texture memory HERE!!!!

    //Done! Issue kernel       
    //Do not need warm up, just for testing
    dim3 threadsPerBlock(8, 8, 1);                                                 //
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);
    transformKernel<<<numBlocks,threadsPerBlock>>>(d_resData, width, height, angle, tex);

    //copy result from gpu to host
    cudaMemcpy(h_resData, d_resData, size, cudaMemcpyDeviceToHost);


    //Do post processing
    //sdkSavePGM(outputFilename, hOutputData, width, height);

    //check result
    for(int i=0;i<256;i++){
        printf("%lf ", h_resData[i]);
    }
}