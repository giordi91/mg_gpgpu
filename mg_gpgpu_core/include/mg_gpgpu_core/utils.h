#pragma once
#include <memory>


namespace mg_gpgpu
{
namespace utils
{
    //simple kernel to zero out a block of memory
    ///param data: pointer to the device data to zero out
    ///param size: size of the buffer
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


    //simple kernel to set out a block of memory to the specific value
    ///param data: pointer to the device data to work on 
    ///param size: size of the buffer
    template<typename T, T VALUE> 
    __global__ void set_value_kernel( T*data, unsigned int size)
    {
        for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
             i < size;
             i+= (blockDim.x*gridDim.x))
        {
            data[i] = static_cast<T>(VALUE);
        }
    }
 


    //utility method to check if the cuda operation was successiful
    #define gpuErrchk(ans) { mg_gpgpu::utils::gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
       if (code != cudaSuccess) 
       {
          fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort) exit(code);
       }
    }
    //same as the abouve but should evaluate to no OP in release mode
    #define gpuErrchkDebug(ans) { mg_gpgpu::utils::gpuAssertDebug((ans), __FILE__, __LINE__); }
    inline void gpuAssertDebug(cudaError_t code, const char *file, int line, bool abort=true)
    {
       #ifndef NDEBUG
       if (code != cudaSuccess) 
       {
          fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort) exit(code);
       }
       #endif
    }


    //////////////////////////////////////////////////////////////////////////////////////
    //ALLOC version of the kernels
    //////////////////////////////////////////////////////////////////////////////////////
    //alloc version for the zero out kernel, for documentation check the kernel
    ///param data: pointer to the device data to zero out
    ///param size: size of the buffer
    ///param threads: how many threads per block
    template<typename T>
    std::unique_ptr<T[]> zero_out_alloc(T*data, unsigned int size, unsigned int threads=512)
    {
         
        T* d_data;
        cudaMalloc( (void**)&d_data, size*sizeof(T));
        cudaMemcpy( d_data, data, size*sizeof(T), cudaMemcpyHostToDevice );
        
        //computing the wanted blocks
        unsigned int blocks = min((size+ threads - 1) / threads, 1024);
        zero_out_kernel<T><<<blocks,threads>>>(d_data,size);

        //alloc return memory
        auto ptr = std::unique_ptr<T[]>(new T[size]);
        cudaMemcpy( ptr.get(), d_data, size*sizeof(T), cudaMemcpyDeviceToHost);
        
        return ptr;
    }

    //alloc version for the set value kernel, for documentation check the kernel
    ///param data: pointer to the device data to work on 
    ///param size: size of the buffer
    ///param threads: how many threads per block
    template<typename T, T VALUE>
    std::unique_ptr<T[]> set_value_alloc(T*data, unsigned int size, unsigned int threads=512)
    {
         
        T* d_data;
        cudaMalloc( (void**)&d_data, size*sizeof(T));
        cudaMemcpy( d_data, data, size*sizeof(T), cudaMemcpyHostToDevice );
        
        //computing the wanted blocks
        unsigned int blocks = min((size+ threads - 1) / threads, 1024);
        set_value_kernel<T,VALUE><<<blocks,threads>>>(d_data,size);

        //alloc return memory
        auto ptr = std::unique_ptr<T[]>(new T[size]);
        cudaMemcpy( ptr.get(), d_data, size*sizeof(T), cudaMemcpyDeviceToHost);
        
        return ptr;
    }

}//namespace utils
}//namespace mg_gpu
