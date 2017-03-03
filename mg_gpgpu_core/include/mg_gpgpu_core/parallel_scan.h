#pragma once
#include <memory>
#include <mg_gpgpu_core/utils.h>

namespace mg_gpgpu
{

/**This is a gpu kernel that implements a naive hillis and steel scan kernel, which is not 
 * work efficient.
 * This kernel only reduce for N hops away for each thread, meaning this kernel needs to be 
 * called several times in order to yield the right result
 * @param in: device pointer to the input data
 * @param out: device pointer to  the output memory 
 * @param hop: how many element to jump left to find the element to add
 * @param count: number of element to be processed
 */
template<typename T>
__global__ void parallel_scan_hillis_steel_kernel(const T* in, T* out, uint32_t hop, uint32_t count)
{
    uint32_t tid =threadIdx.x + (blockDim.x * blockIdx.x);
    //firs we check if we are inside the data boundary 
    if( tid < count)
    {
		if(tid >= hop)
		{ out[tid] = in[tid] + in[tid-hop]; }
		else
		{ out[tid] = in[tid]; }
    }    
}
template<typename T>
inline T* parallel_scan_hillis_steel(T* d_in, T* d_out, uint32_t count)
{

    //since we perform swap at beginning of the loop, we need to start 
    //with in and out inverted
    std::swap(d_in, d_out);
    
    ////computing the wanted blocks
    uint32_t threads = 512;
    uint32_t blocks = ((count/threads) != 0)?(count/threads) +1 : (count/threads);
    if (blocks ==0)
    { blocks =1; }

    for (int hop =1; hop<count;hop<<=1)
    {
        std::swap(d_in, d_out);
        parallel_scan_hillis_steel_kernel<T><<<blocks, threads>>>(d_in,d_out, hop, count);
        gpuErrchkDebug( cudaPeekAtLastError() );
    }

    return d_out;
}

///////////////////////////////////////////////////////////////////////////////////////
//  ALLOCS VARIANTS
//////////////////////////////////////////////////////////////////////////////////////


/** Alloc variant of hillis_steel
 * @param data: host data point to copy and process on gpu
 * @param count: element count of the buffer to process
 * @returns: unique pointer with result memory from the kernel
 */
template<typename T>
std::unique_ptr<T[]> parallel_scan_hillis_steel_alloc(T* data, uint32_t count)
{
    T* in;
    T* out;
    gpuErrchkDebug(cudaMalloc( (void**)&in,  count*sizeof(T)));
    gpuErrchkDebug(cudaMalloc( (void**)&out, count*sizeof(T)));
    gpuErrchkDebug(cudaMemcpy( in, data, count*sizeof(T), cudaMemcpyHostToDevice ));


    auto final_ptr = parallel_scan_hillis_steel<T>(in,out,count);
    //copying result back from gpu
    auto ptr =std::unique_ptr<T[]>(new T[count]);
    gpuErrchkDebug(cudaMemcpy( ptr.get(), final_ptr, count*sizeof(T), cudaMemcpyDeviceToHost));
    //freeing memory
    cudaFree(in);
    cudaFree(out);

    return ptr;

}
}//end mg_gpgpu namespace
