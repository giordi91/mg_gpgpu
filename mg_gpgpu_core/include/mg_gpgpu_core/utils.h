#pragma once
#include <memory>


namespace mg_gpgpu
{
namespace utils
{
    template<typename T> 
    __global__ void zero_out_kernel( T*data, unsigned int size)
    {
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
             i < size;
             i+= (blockDim.x*gridDim.x))
        {
            data[i] = static_cast<T>(0);
        }
    }

    template<typename T>
    std::unique_ptr<T[]> zero_out_alloc(T*data, unsigned int size)
    {
         
        T* d_data;
        cudaMalloc( (void**)&d_data, size*sizeof(T));
        cudaMemcpy( d_data, data, size*sizeof(T), cudaMemcpyHostToDevice );
        
        //computing the wanted blocks
        unsigned int threads = 512;
        unsigned int blocks = min((size+ threads - 1) / threads, 1024);
        zero_out_kernel<T><<<blocks,threads>>>(d_data,size);

        //alloc return memory
        auto ptr = std::unique_ptr<T[]>(new T[size]);
        cudaMemcpy( ptr.get(), d_data, size*sizeof(T), cudaMemcpyDeviceToHost);
        
        return ptr;
    
    }


}//namespace utils
}//namespace mg_gpu
