#pragma once
#include <memory>
#include <mg_gpgpu_core/utils.h>
#include <mg_gpgpu_core/parallel_reduce.h>
#include <limits>

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


template<typename T, T SENTINEL_VALUE>
__inline__ __device__ void intra_block_barrier(int tid, 
                                               int gid, 
                                               volatile T* d_intermediate, 
                                               T curr_value, 
                                               T* out)
{
    T p =0;
    if (tid ==0)
    {
        if (gid==0)
        {
            d_intermediate[0] = curr_value; 
        } 
        else
        {
            //spin lock on global memory value
            p = d_intermediate[gid -1];
            while(p == SENTINEL_VALUE)
            {
                p = d_intermediate[gid -1];
            }
            d_intermediate[gid] = p+curr_value;
        }
    }
    __syncthreads();
    *out = p;
}


template<typename T>
__device__ inline uint32_t get_dynamic_block_id(T* mem_ptr)
{
	__shared__ uint32_t gId;
    if(threadIdx.x == 0)
    {
        gId= atomicAdd(mem_ptr,1);
    }
    __syncthreads();
    return gId;
}


template<typename T, T SENTINEL_VALUE>
__global__ void parallel_stream_scan_kernel(T* d_in, volatile T* d_intermediate, T* atom,uint32_t count,uint32_t blocks)
{

	//getting block id dynamiccally, not using the one cuda provides
   	uint32_t _gId = get_dynamic_block_id(atom); 

    //reducing block wise
    int tId = blockDim.x * _gId + threadIdx.x;
    T sum =  d_in[tId];
    
    ////TODO WARP SIZE 
    T res = block_reduce_masked<T,32>(sum, tId,count);    
    T prev_result =0;
    intra_block_barrier<T,SENTINEL_VALUE>(threadIdx.x, _gId, d_intermediate, res,&prev_result);

    //perform scan in the block
    for(int i =1; i<blockDim.x; i<<=1)
    {
		if(threadIdx.x>= i)
		{ d_in[tId] = d_in[tId] + d_in[tId-i]; }
        __syncthreads(); 
    }

	//TODO remove the branch
	//here we add the result from the previous block
    if (_gId !=0)
    { d_in[tId] += d_intermediate[_gId-1]; } 
}


template<typename T, T SENTINEL_VALUE>
inline void parallel_stream_scan(T* d_in, T* d_intermediate, uint32_t count)
{

    //kicking the kernels, first we reduce 
    const uint32_t WARP_SIZE = 32;
    

    //
    uint32_t threads = 128;
    uint32_t blocks = ((count%threads) != 0)?(count/threads) +1 : (count/threads);
    //here we have an extra one which will be our atomic value for blocks
    if (blocks == 0)
    {blocks =1;}

    uint32_t threadsSet = 32;
    uint32_t blocksSet = ((blocks/threadsSet) != 0)?(blocks/threadsSet) +1 : (blocks/threadsSet);
    if (blocksSet==0)
    {blocksSet=1;}

    //setting memory  
    mg_gpgpu::utils::set_value_kernel<T,SENTINEL_VALUE><<<blocksSet,threadsSet>>>(d_intermediate ,blocks);

    //zeroing out last value , this might be optimized a little by having a bespoke kernel
    mg_gpgpu::utils::zero_out_kernel<<<1,1>>>(d_intermediate + blocks, 1);

    uint32_t sharedMemorySize = (threads/WARP_SIZE) *sizeof(T);

    //kicking the kernel
    parallel_stream_scan_kernel<T,SENTINEL_VALUE><<<blocks,threads, sharedMemorySize>>>(d_in,d_intermediate, d_intermediate+blocks ,count,blocks);

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

template<typename T>
std::unique_ptr<T[]> parallel_stream_scan_alloc(T* data, uint32_t count)
{
    T* d_in;
    T* d_intermediate;
    gpuErrchkDebug(cudaMalloc( (void**)&d_in,  count*sizeof(T)));
    gpuErrchkDebug(cudaMemcpy( d_in, data, count*sizeof(T), cudaMemcpyHostToDevice ));
    //
    ////computing the wanted blocks
    uint32_t threads = 128;
    uint32_t blocks = ((count%threads) != 0)?(count/threads) +1 : (count/threads);
    //here we have an extra one which will be our atomic value for blocks
    if (blocks == 0)
    {blocks =1;}

    //compute_blocks(threads, blocks,count);
    gpuErrchkDebug(cudaMalloc( (void**)&d_intermediate,  (blocks + 1)*sizeof(T)));

    constexpr T SENTINEL = std::numeric_limits<T>::max();
    parallel_stream_scan<T, SENTINEL>(d_in, d_intermediate, count);

    auto ptr =std::unique_ptr<T[]>(new T[count]);
    gpuErrchkDebug(cudaMemcpy( ptr.get(), d_in, count*sizeof(T), cudaMemcpyDeviceToHost));

    return ptr;
}

}//end mg_gpgpu namespace
