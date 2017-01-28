#pragma once

#include <iostream>


    
template <typename T>
__global__ void global_reduce_kernel(T * d_out, T * d_in, unsigned int count)
{
	extern __shared__ __align__(sizeof(T)) unsigned char shared_data[];
    T *sdata= reinterpret_cast<T *>(shared_data);

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    if (myId >= count)
    { sdata[tid] = 0.0;}
    else
    {
        sdata[tid]= d_in[myId];
    }
    __syncthreads();        // make sure all adds at one stage are done!
    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[tid];
    }
}


template<typename T>
T parallel_reduce( T* host_data, unsigned int element_count)
{

    unsigned int block_size = 1024;
    T* device_data;
    T* out_device_data;

    unsigned int blocks = ((element_count%block_size) != 0)?(element_count/block_size) +1:
                                                            (element_count/block_size);
    unsigned int size = blocks * block_size* sizeof(T);

   

    cudaMalloc( (void**)&device_data, size );
    cudaMalloc( (void**)&out_device_data, size );
    cudaMemcpy( device_data , host_data, size, cudaMemcpyHostToDevice );

    T * in = out_device_data;
    T * out = device_data;
    for(int i =element_count; i>1; )
    {

        std::swap(in,out);
        unsigned int ratio =i/ block_size;
        unsigned int grid_size = ((i%block_size) != 0)?(ratio) +1: (ratio);
        dim3 dimBlock( block_size, 1 );
        dim3 dimGrid( grid_size, 1 );
        global_reduce_kernel<T><<<grid_size, block_size, sizeof(T)*block_size>>>( out, in, i);
        i = grid_size;
       
       
    }
    T result;
    cudaMemcpy(&result, out,sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(device_data);
    cudaFree(out_device_data);
    return result;

}

