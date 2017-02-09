#pragma once

#include <iostream>
#include <mg_gpgpu_core/utils.h>

/**
 * This version of parallel reduce utilized the shared memory to do the kernel accumulation
 * per block, once the whole reduction is done, only thread 0 in the block is going to write 
 * to global memory, this method is good because reduce memory pressure on global memory
 * @parm d_out, the buffer we are going to write the result to
 * @param d_in: the buffer with the source data
 * @param count: how many elements we need to process
 */
template <typename T>
__global__ void parallel_reduce_shared_kernel(T * d_out, const T * d_in, unsigned int count)
{
    //this is a pretty off the shelf cuda impelemntation of a reduce kernel,
    //here we use a unsigned char shared_data and then cast the value, due to the 
    //fact that cuda won't let me redefined the shared memory for some reason.
	extern __shared__ __align__(sizeof(T)) unsigned char shared_data[];
    T *sdata= reinterpret_cast<T *>(shared_data);

    //computing global index thread and index inside the block
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    //here we load the memory from global to shared, we take extra care
    // in handling the boundary condition since the following for loop
    // will loop over the whole block, no matter if used or not
    if (myId >= count)
    { sdata[tid] = static_cast<T>(0);}
    else
    { sdata[tid]= __ldg(&d_in[myId]); }

    //here we syncronize the thread in a block wise fashion making sure 
    //the data has been loaded for the whole block
    __syncthreads();        
    
    //here we trigger the reduction, we start with a block wide dimension 
    //and we proceed in a log(blockDim.x) fashion, halving the size at 
    //each iteration
    for (unsigned int block_width = blockDim.x / 2;block_width > 0; block_width >>= 1)
    {
        //here we need to mask out the threads that actually need
        //to perform the operation, since we sum the higher part of the 
        //block to the lower part of the block, half of the threads won't be 
        //working
        if (tid < block_width)
        {
            sdata[tid] += sdata[tid +block_width ];
        }
        //since we are working with global memory we always need to performa
        //a block wise syncronization to make sure not incur in race conditions
        __syncthreads();        
    }

    //at this point the block has been reduced in the threadIdx.x == 0,
    //so we let the first thread write to memory, doesnt necessarly have 
    //to be thread zero, the key thing is that only one thread per block
    //writes to shared memory
    if (tid == 0)
    {
        //here we use the block index to write to global memory, this mean
        //we eneded up with blockCount data written, where block count is the 
        //total number of blocks in the kernel
        d_out[blockIdx.x] = sdata[tid];
    }
}

/**This is the host version for the reduce using shared memory,
*to note, it is a host function but it expects already device memory pointers,
*if you want to pass in host memory pointers please check the alloc variant of this 
*method. To note, we are going to ping pong between the input and output buffer, so 
*        make a copy of the data if you need the original, the memory for the output buffer
*        needs to have the size of the number of blocks, if you run 512 threads per block
*        you need n/512 + 1 elements. (+1 might be redundant if n is divisibile by 512
*@param d_in : input device buffer
*@param d_out :output device buffer
*/
template<typename T>
inline T parallel_reduce_shared( T* d_in ,T*d_out, unsigned int count)
{
    unsigned int block_size = 512;

    //we perform the swap before the run, so that the result is in d_out compared than d_in
    //so we perform an additional swap before running to compensate for that
    std::swap(d_in,d_out);
    for(int i =count; i>1; )
    {

        //we keep deviding the grid_size by two, so we expect log_2 of n of the original grid 
        //size iterations
        std::swap(d_in,d_out);
        unsigned int ratio =i/ block_size;
        unsigned int grid_size = ((i%block_size) != 0)?(ratio) +1: (ratio);
        dim3 dimBlock( block_size, 1 );
        dim3 dimGrid( grid_size, 1 );
        parallel_reduce_shared_kernel<T><<<grid_size, block_size, sizeof(T)*block_size>>>( d_out, d_in, i);
        i = grid_size;
    }
    //brining the result down
    T result;
    cudaMemcpy(&result, d_out,sizeof(T), cudaMemcpyDeviceToHost);
    return result;
}


//Device method for reducing a warp, it is based on the shuffle method where
//an intrisic instruction is used to reduce data withing a warp in a single instruction 
template<typename T, unsigned int WARP_SIZE>
__inline__ __device__ T warp_reduce(T value)
{
    for (unsigned int offset = warpSize/2; offset>0; offset/=2)
    {
        //here we pass down the value accumulating from the warp
        //in a thread. What is happening here is that the since the SIMD
        //register is 32 floats wide, it will shift the value accross thread
        //like a shift inside a register, so __shfl_down down return the value 
        //that the threadIdx.x + 1 holds, and we add it to the value of
        //threadIdx.x, this happen syncronous across threads
        value += __shfl_down(value, offset); 
    }
    return value;
}

//This mehotd wil perform a  block reduce, a block is composed of multiple warps
//so we call warp reduce and then force a sync , wonce that is done we know all the 
//warps reduced and we  can get the value
template<typename T, unsigned int WARP_SIZE>
__inline__ __device__ T block_reduce(T value)
{
    //allocating shared memory per block
    //here since we are allocating unsigned char (8 bit, 1 byte) we multipleply the size
    //of the type T (which is in byte) times the warp size.
    //Now we only need to be able to store one element per warp, since each warp will 
    //reduce inthe registers and not in shared memory
	static __shared__ __align__(sizeof(T)) unsigned char shared_data[WARP_SIZE * sizeof(T)];
    T *shared= reinterpret_cast<T *>(shared_data);

    //the lane is the id in the warp size register, meaning between 0-31
    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    
    //performing the reduce
    value = warp_reduce<T, WARP_SIZE>(value);
    //now if this is the first thread in the warp we are going to write the value 
    //in shared memory, otherwise we do nothing
    if (lane ==0) shared[warp_id] = value;

    //making sure all the values from the warps have been written
    __syncthreads();

    //now only the first warp is going to load all the values from the shared memory
    //and we perform a final warp reduce
    value = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (warp_id == 0) value = warp_reduce<T, WARP_SIZE>(value);

    //value now holds the value of the whole reduced block
    return value;
}



/**
 * This procedure perform a reduction using the shuffle instruction which is available from
 * kerpler architecture and above, to note this reduce each block and store the block result
 * in d_out, does not compute the final value if you have multiple blocks, for more information
 * check the alloc variant
 * @param d_in: input buffer for the operation, aka the array to be reduced
 * @param d_out: this buffer has the same size of the number of blocks, since each block
 *               perform a reduction
 * @param count: how many element we have to process
 */
template <typename T, unsigned int WARP_SIZE>
__global__ void parallel_reduce_shuffle_kernel(const T * d_in, T * d_out, unsigned int count)
{
    //this is a grid-stride loop, this allows to handle arbitrary
    //size keeping maximum coaleshed access of memory,as showed in 
    //parallel for all nvidia blog post.
    //In short allow to tune for specific hardware by running specific amount of 
    //blocks and threads to fully maximise occupancy
    T sum = static_cast<T>(0);
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
         i < count;
         i+= (blockDim.x*gridDim.x))
    {
        sum += d_in[i];
    }
    
    //now sum contains the reduction of the grid size element now we know that we are
    //fitting the block width and we can proceed accordingly
    sum = block_reduce<T,WARP_SIZE>(sum);
    if (threadIdx.x==0)
        d_out[blockIdx.x]=sum;
}

/**
 * This is a variation of the shuffle procedure, the regular method performs a reduction per block
 * then each block stores in memory the needed value , instead, here we perform an atomic
 * add, leveranging the constant performance improvment that there was since kepler architecture
 * and was improved even more in maxwell.
 * @param d_in: input buffer for the operation, aka the array to be reduced
 * @param d_out: this buffer has the same size of the number of blocks, since each block
 *               perform a reduction
 * @param count: how many element we have to process
 */
template <typename T, unsigned int WARP_SIZE>
__global__ void parallel_reduce_shuffle_atomic_kernel(const T * d_in, T * d_out, unsigned int count)
{
    //this is a grid-stride loop, this allows to handle arbitrary
    //size keeping maximum coaleshed access of memory,as showed in 
    //parallel for all nvidia blog post.
    //In short allow to tune for specific hardware by running specific amount of 
    //blocks and threads to fully maximise occupancy
    T sum = static_cast<T>(0);
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
         i < count;
         i+= (blockDim.x*gridDim.x))
    {
        //__ldg instruction should work better if you load from const in this 
        //case did not yield better results
        sum += __ldg(&d_in[i]);
    }
    
    //now sum contains the reduction of the grid size element now we know that we are
    //fitting the block width and we can proceed accordingly
    sum = block_reduce<T,WARP_SIZE>(sum);
    if (threadIdx.x==0)
        atomicAdd(d_out,sum);
}

/**
 * Host method to kick a parallel reduce using the shuffle instruction
 * @param d_in: input buffer for the operation, aka the array to be reduced
 * @param d_out: this buffer has the same size of the number of blocks, since each block
 *               perform a reduction
 * @param count: how many element we have to process
 *
 */
template<typename T>
void  parallel_reduce_shuffle( const T* in,T* out, unsigned int count)
{
    //computing the wanted blocks
    unsigned int threads = 512;
    unsigned int blocks = min((count + threads - 1) / threads, 1024);

    //kicking the kernels, first we reduce 
    const unsigned int WARP_SIZE = 32;

    //might wont an hardcoded kernel? I will bench and see if it helps
    parallel_reduce_shuffle_kernel<T,WARP_SIZE><<<blocks, threads>>>(in, out, count);
    //here we have up to max 1024 output elements that need to be further reduced
    parallel_reduce_shuffle_kernel<T,WARP_SIZE><<<1, 1024>>>(out, out, blocks);
}


/**
 * Host method to kick a parallel reduce using the shuffle instruction and the final atomic
 * accumulation
 * @param d_in: input buffer for the operation, aka the array to be reduced
 * @param d_out: must hold just a single T value , since final accumulation is done with atomics
 *               
 * @param count: how many element we have to process
 *
 */
template<typename T>
void parallel_reduce_shuffle_atomic( const T* in,T* out, unsigned int count)
{

    //computing the wanted blocks
    unsigned int threads = 512;
    unsigned int blocks = min((count + threads - 1) / threads, 1024);

    //kicking the kernels, first we reduce 
    const unsigned int WARP_SIZE = 32;

    //here we need to zero out the outptu value since we are going to acculuate into it
    //the fastest way is to have a kernel to write the value to it, memset is slower!!
    mg_gpgpu::utils::zero_out_kernel<T><<<1,1>>>(out,1);
    parallel_reduce_shuffle_atomic_kernel<T,WARP_SIZE><<<blocks, threads>>>(in, out, count);
}
///////////////////////////////////////////////////////////////////////////////////////
//  ALLOCS VARIANTS
//////////////////////////////////////////////////////////////////////////////////////

/**
 * Variant version of the shared reduce where memory is allocated on the gpu automatically for 
 * the user.
 * @parm host_data: pointer to host memory to load
 * @parm count: how many element we need to process
 */
template<typename T>
T parallel_reduce_shared_alloc( T* host_data, unsigned int count)
{
    unsigned int block_size = 512;
    //declaring gpu pointers
    T* device_data;
    T* out_device_data;

    //computing the block 
    unsigned int blocks = ((count%block_size) != 0)?(count/block_size) +1:
                                                            (count/block_size);
    unsigned int size =count*sizeof(T); 

    //allocating memory
    cudaMalloc( (void**)&device_data, size );
    //we know already how many blocks we are gonna kick in the first iteration
    //so we don't have to allocate the full array for the output data,
    //we can get away with block count elements
    cudaMalloc( (void**)&out_device_data, blocks*sizeof(T));
    cudaMemcpy( device_data , host_data, size, cudaMemcpyHostToDevice );

    T result = parallel_reduce_shared<T>(device_data, out_device_data, count);
    return result;

}

/**
 * Variant version of the shuffle atomic reduce where memory is 
 * allocated on the gpu automatically for the user.
 * @parm host_data: pointer to host memory to load
 * @parm count: how many element we need to process
 */
template<typename T>
T parallel_reduce_shuffle_atomic_alloc( T* host_data, unsigned int count)
{
    //copying/allocating memory to device
    T* in;
    T* out;

    //computing the wanted blocks
    unsigned int threads = 512;
    unsigned int blocks = min((count + threads - 1) / threads, 1024);

    cudaMalloc( (void**)&in, count*sizeof(T));
    cudaMalloc( (void**)&out, blocks*sizeof(T));
    cudaMemcpy( in, host_data, count*sizeof(T), cudaMemcpyHostToDevice );


    parallel_reduce_shuffle_atomic<T>(in, out, count);
    //computing result and freeing memory
    T result;
    cudaMemcpy(&result, out,sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);
    return result;

}

/**
 * Variant version of the shuffle reduce where memory is 
 * allocated on the gpu automatically for the user.
 * @parm host_data: pointer to host memory to load
 * @parm count: how many element we need to process
 */
template<typename T>
T parallel_reduce_shuffle_alloc( T* host_data, unsigned int count)
{
    //copying/allocating memory to device
    T* in;
    T* out;

    //computing the wanted blocks
    unsigned int threads = 512;
    unsigned int blocks = min((count + threads - 1) / threads, 1024);

    cudaMalloc( (void**)&in, count*sizeof(T));
    cudaMalloc( (void**)&out, blocks*sizeof(T));
    cudaMemcpy( in, host_data, count*sizeof(T), cudaMemcpyHostToDevice );


    parallel_reduce_shuffle<T>(in, out, count);
    T result;
    cudaMemcpy(&result, out,sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);
    return result;

}

