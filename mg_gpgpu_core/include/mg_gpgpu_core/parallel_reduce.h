#pragma once

#include <iostream>

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

template<typename T>
inline T parallel_reduce_shared( T* in ,T*out, unsigned int element_count)
{
    unsigned int block_size = 1024;

    for(int i =element_count; i>1; )
    {

        std::swap(in,out);
        unsigned int ratio =i/ block_size;
        unsigned int grid_size = ((i%block_size) != 0)?(ratio) +1: (ratio);
        dim3 dimBlock( block_size, 1 );
        dim3 dimGrid( grid_size, 1 );
        parallel_reduce_shared_kernel<T><<<grid_size, block_size, sizeof(T)*block_size>>>( out, in, i);
        i = grid_size;
    }
    T result;
    cudaMemcpy(&result, out,sizeof(T), cudaMemcpyDeviceToHost);
    return result;
}

template<typename T>
T parallel_reduce_shared_alloc( T* host_data, unsigned int element_count)
{
    unsigned int block_size = 1024;
    T* device_data;
    T* out_device_data;

    unsigned int blocks = ((element_count%block_size) != 0)?(element_count/block_size) +1:
                                                            (element_count/block_size);
    unsigned int size = blocks * block_size* sizeof(T);

    cudaMalloc( (void**)&device_data, size );
    //we know already how many blocks we are gonna kick in the first iteration
    //so we don't have to allocate the full array for the output data,
    //we can get away with block count elements
    cudaMalloc( (void**)&out_device_data, blocks*sizeof(T));
    cudaMemcpy( device_data , host_data, size, cudaMemcpyHostToDevice );

    T * in = out_device_data;
    T * out = device_data;
    T result = parallel_reduce_shared<T>(in, out, element_count);
    return result;

}

template<typename T, unsigned int WARP_SIZE>
__inline__ __device__ T warp_reduce(T value)
{
    for (unsigned int offset = warpSize/2; offset>0; offset/=2)
    {
        value += __shfl_down(value, offset); 
    }
    return value;
}

template<typename T, unsigned int WARP_SIZE>
__inline__ __device__ T block_reduce(T value)
{
	static __shared__ __align__(sizeof(T)) unsigned char shared_data[WARP_SIZE * sizeof(T)];
    T *shared= reinterpret_cast<T *>(shared_data);

    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    
    value = warp_reduce<T, WARP_SIZE>(value);
    if (lane ==0) shared[warp_id] = value;

    __syncthreads();

    value = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (warp_id == 0) value = warp_reduce<T, WARP_SIZE>(value);

    return value;
}



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
        sum += __ldg(&d_in[i]);
    }
    
    //now sum contains the reduction of the grid size element now we know that we are
    //fitting the block width and we can proceed accordingly
    sum = block_reduce<T,WARP_SIZE>(sum);
    if (threadIdx.x==0)
        d_out[blockIdx.x]=sum;
}

template<typename T>
inline T parallel_reduce_shuffle( const T* in,T* out, unsigned int element_count)
{

    //computing the wanted blocks
    unsigned int threads = 512;
    unsigned int blocks = min((element_count + threads - 1) / threads, 1024);

    //kicking the kernels, first we reduce 
    const unsigned int WARP_SIZE = 32;
    parallel_reduce_shuffle_kernel<T,WARP_SIZE><<<blocks, threads>>>(in, out, element_count);
    parallel_reduce_shuffle_kernel<T,WARP_SIZE><<<1, 1024>>>(out, out, blocks);

    //computing result and freeing memory
    T result;
    cudaMemcpy(&result, out,sizeof(T), cudaMemcpyDeviceToHost);

    return result;

}

template<typename T>
T parallel_reduce_shuffle_alloc( T* host_data, unsigned int element_count)
{
    //copying/allocating memory to device
    T* in;
    T* out;

    //computing the wanted blocks
    unsigned int threads = 512;
    unsigned int blocks = min((element_count + threads - 1) / threads, 1024);

    cudaMalloc( (void**)&in, element_count*sizeof(T));
    cudaMalloc( (void**)&out, blocks*sizeof(T));
    cudaMemcpy( in, host_data, element_count*sizeof(T), cudaMemcpyHostToDevice );


    T result = parallel_reduce_shuffle<T>(in, out, element_count);

    cudaFree(in);
    cudaFree(out);
    return result;

}

