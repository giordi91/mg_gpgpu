#pragma once
#include <memory>
#include <mg_gpgpu_core/utils.h>
#include <mg_gpgpu_core/reduce.h>
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

    template<typename T>
        __device__ inline void  parallel_scan_blelloch_kernel(T* d_in, int lg, uint32_t count, int blockId)
        {

            //uint32_t globalOffset = blockId * blockDim.x;

            //int myId = (threadIdx.x + blockDim.x * blockIdx.x);
            for(int l =lg; l>=0; l--)
            {
                uint32_t hop = l;
                bool cnd = (hop ==0);
                int shift = 2 * cnd + (2<<hop) * !cnd;
                int myId =  threadIdx.x*shift;

                int exponent = 2<<(hop) ;
                int exponent2 =1*cnd +  ((2<<(hop-1)) * !cnd);
                int array_id = myId + exponent -1;


                if(array_id < count)
                {
                    T temp = d_in[myId + exponent2 -1];
                    d_in[myId + exponent2 -1] = d_in[array_id];
                    d_in[array_id ] += temp  ;

                }
                __syncthreads();
            }
        }


    /**
     * @brief wrapper function for the blellock_array_kernel, to kick one for each block
     */
    template <typename T>
        __global__ void parallel_scan_blelloch_wrap_kernel(T* d_in,  uint32_t lg, uint32_t count )
        {
            parallel_scan_blelloch_kernel<T>(d_in, lg, count,blockIdx.x );
        }

    template<typename T, T SENTINEL_VALUE>
        __device__ void intra_block_barrier(int tid,
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
                    *out = p;
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
                    *out = p;
                }
            }
            __syncthreads();
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


    template<typename T, T SENTINEL_VALUE, bool EXCLUSIVE=0>
        __global__ void parallel_stream_scan_kernel(T* d_in, volatile T* d_intermediate, T* atom,uint32_t count,uint32_t blocks)
        {
            __shared__ T temp[2048];
            //getting block id dynamically, not using the one Cuda provides
            uint32_t _gId= get_dynamic_block_id(atom);

            //reducing block wise
            int tId = blockDim.x * _gId + threadIdx.x;
            if ( tId< count)
            { temp[ threadIdx.x] =  d_in[tId] ;  }
            else
            { temp[ threadIdx.x] =  0 ;  }

            //load value into registers
            T sum = temp[threadIdx.x];
            T res = block_reduce<T,32>(sum);

            //spin locking barrier for intra block sync
            __shared__ T prev;
            intra_block_barrier<T,SENTINEL_VALUE>(threadIdx.x, _gId, d_intermediate, res,&prev);


            int thid = threadIdx.x;
            int n =1024;
            int pout = 0, pin = 1;

#pragma unroll
            for (int offset = 1; offset < n; offset *= 2)
            {
                pout = 1 - pout; // swap double buffer indices
                pin = 1 - pout;

                //bool cnd =(thid >= offset);
                //T A = temp[pin*n+thid]+ temp[pin*n+thid - (offset*cnd)];
                //T B = temp[pin*n+thid];
                //temp[pout*n+thid] = (A * cnd) + (B * (!cnd));

                if (thid >= offset)
                    temp[pout*n+thid] = temp[pin*n+thid]+ temp[pin*n+thid - offset];
                else
                    temp[pout*n+thid] = temp[pin*n+thid];
                __syncthreads();
            }

            //here we add the result from the previous block

            if ( tId< count )
            {
                d_in[tId+ EXCLUSIVE] = temp[pout*n + thid]  +prev;
            }
            if(EXCLUSIVE && tId ==0)
            {
                d_in[0] = 0;
            }

        }


    /**
     * @brief performs a scan inclusive, using dynamic blocks allocation and spin locking
     *
     * This kernel is based on the paper:
     * http://dl.acm.org/citation.cfm?id=2442539
     * The main idea behind it, is you move the syncronization point in the algorithm as early as
     * possible, in this case, we use a reduce in the blocks to feed the result in the next block.
     * To make sure we process the blocks increasing order atomic id are generated and spin lock
     * in global memory is used to stall the blocks, waiting for the previous reduce
     *
     * @tparam T data type to use in the algorithm
     * @tparam SENTINEL_VALUE T value we want to use for the spin lock, as long this value is in memory
     *                          the block won't proceed any further
     * @tparam WARP_SIZE the size of a warp usually 32
     * @param d_in device memory, to note this memory is not constant, and scan is in place
     * @param d_intermediate intermediate device memory, this array has the same size as number of blocks
     * @param count size of the d_in array
     */
    template<typename T, T SENTINEL_VALUE , bool EXCLUSIVE=0>
        inline void parallel_stream_scan(T* d_in, T* d_intermediate, uint32_t count)
        {

            const uint32_t WARP_SIZE = 32;

            uint32_t threads = 1024;
            uint32_t blocks = ((count%threads) != 0)?(count/threads) +1 : (count/threads);
            if (blocks == 0)
            {blocks =1;}

            uint32_t threadsSet = 128;
            uint32_t blocksSet = ((blocks/threadsSet) != 0)?(blocks/threadsSet) +1 : (blocks/threadsSet);
            if (blocksSet==0)
            {blocksSet=1;}

            //setting memory
            mg_gpgpu::utils::set_value_kernel<T,SENTINEL_VALUE><<<blocksSet,threadsSet>>>(d_intermediate ,blocks);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

            //zeroing out last value , this might be optimized a little by having a bespoke kernel
            mg_gpgpu::utils::zero_out_kernel<T><<<1,1>>>(d_intermediate + blocks, 1);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));

            //here we use the shuffle block reduce, where each block reduces a warp in shared memory and then performs
            //a final shuffle, intermediate result stored in shared memory
            uint32_t sharedMemorySize = (threads/WARP_SIZE) *sizeof(T);

            //kicking the kernel
            parallel_stream_scan_kernel<T,SENTINEL_VALUE,EXCLUSIVE><<<blocks,threads, sharedMemorySize>>>(d_in,d_intermediate, d_intermediate+blocks ,count,blocks);
            err = cudaGetLastError();
            if (err != cudaSuccess)
                printf("Error: %s\n", cudaGetErrorString(err));


        }



    ///////////////////////////////////////////////////////////////////////////////////////
    //  ALLOCS VARIANTS
    //////////////////////////////////////////////////////////////////////////////////////


    /**
     * @brief Alloc variant of hillis_steel
     *
     * @tparam T basic datatype of the objects to be processed
     * @param data host data point to copy and process on gpu
     * @param count element count of the buffer to process
     *
     * @return unique pointer with result memory from the kernel
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

    /**
     * @brief allocation variant of stream scan function
     *
     * @param data: Host pointer of the data to process
     * @param count: size of the host pointer array
     *
     * @return unique pointer of the result memory, since is an smart pointer
     *         ownership is transfered
     */
    template<typename T, bool EXCLUSIVE=1>
        std::unique_ptr<T[]> parallel_stream_scan_alloc(T* data, uint32_t count)
        {
            T* d_in;
            T* d_intermediate;
            gpuErrchkDebug(cudaMalloc( (void**)&d_in,  count*sizeof(T)));
            gpuErrchkDebug(cudaMemcpy( d_in, data, count*sizeof(T), cudaMemcpyHostToDevice ));
            //
            ////computing the wanted blocks
            uint32_t threads = 1024;
            uint32_t blocks = ((count%threads) != 0)?(count/threads) +1 : (count/threads);
            //here we have an extra one which will be our atomic value for blocks
            if (blocks == 0)
            {blocks =1;}

            //compute_blocks(threads, blocks,count);
            gpuErrchkDebug(cudaMalloc( (void**)&d_intermediate,  (blocks + 1)*sizeof(T)));

            //using maximum possible value as a sentinel
            constexpr T SENTINEL = std::numeric_limits<T>::max();
            std::cout<<"exclusive "<<EXCLUSIVE<<std::endl;
            parallel_stream_scan<T, SENTINEL, EXCLUSIVE>(d_in, d_intermediate, count);
            cudaError_t err = cudaGetLastError();
            //if (err != cudaSuccess)
            //    printf("Error: %s\n", cudaGetErrorString(err));

            auto ptr =std::unique_ptr<T[]>(new T[count]);
            gpuErrchkDebug(cudaMemcpy( ptr.get(), d_in, count*sizeof(T), cudaMemcpyDeviceToHost));

            return ptr;
        }

    template<typename T>
        std::unique_ptr<T[]> parallel_scan_blelloch_alloc(T* data, uint32_t count)
        {
            T* d_in;
            gpuErrchkDebug(cudaMalloc( (void**)&d_in,  count*sizeof(T)));
            gpuErrchkDebug(cudaMemcpy( d_in, data, count*sizeof(T), cudaMemcpyHostToDevice ));

            //one block with given size of element, this kernel is supposed to work on one block only
            uint32_t threads = count;
            uint32_t blocks = 1;

            //computing the log needed
            //TODO not really an expensive operation but would be nice to try compute it myself,
            //leveraging the fact (maybe) i know is a power of two
            int lg  =  static_cast<int>(std::log2(count));
            parallel_reduce_full_array_wrap_kernel<T><<<blocks,threads>>>(d_in,count);

            mg_gpgpu::utils::zero_out_kernel<T><<<1,1>>>(d_in+ (count -1),1);
            parallel_scan_blelloch_wrap_kernel<T><<<blocks, threads>>>(d_in, lg, count);

            auto ptr =std::unique_ptr<T[]>(new T[count]);
            gpuErrchkDebug(cudaMemcpy( ptr.get(), d_in, count*sizeof(T), cudaMemcpyDeviceToHost));

            cudaFree(d_in);
            return ptr;

        }

}//end mg_gpgpu namespace
