#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#define threadPerBlock 64

using namespace cooperative_groups;

__device__ int sumReduction(thread_group g, int* x, int val){
    int lane = g.thread_rank();

    for(int i=g.size()/2;i>0;i/=2){
        x[lane] = val;

        //g.sync();

        if(lane<i){
            val += x[lane+i];
        }

        g.sync();
    }

    if(g.thread_rank() == 0){
        return val;
    }
    else{
        return -1;
    }
}

__global__ void cgkernel(){

    extern __shared__ int workspace[];

    thread_block wholeBlock = this_thread_block();

    int input = wholeBlock.thread_rank();
 
    int output = sumReduction(wholeBlock, workspace, input);
    int expectedout = wholeBlock.size()*(wholeBlock.size()-1)/2;

    if(wholeBlock.thread_rank()==0){
        printf(" Sum of all ranks 0..%d in wholeBlock is %d (expected %d)\n\n",
                wholeBlock.size()-1,output,
                expectedout);
    
        printf(" Now creating %d groups, each of size 16 threads:\n\n",
                wholeBlock.size()/16);
    }

    thread_block_tile<16> tile16 = tiled_partition<16>(this_thread_block());

    input = tile16.thread_rank();
    output = sumReduction(tile16, workspace, input);
    expectedout = 120;

    if(tile16.thread_rank()==0)
         printf("   Sum of all ranks 0..15 in this tiledPartition16 group is %d (expected %d)\n",output,expectedout);

}


int main(){
    
    cgkernel<<<1,threadPerBlock,threadPerBlock*sizeof(int)>>>();
    cudaDeviceSynchronize();

    return 0;
}