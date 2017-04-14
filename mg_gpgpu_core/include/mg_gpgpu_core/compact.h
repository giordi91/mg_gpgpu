#pragma once
#include <memory>
#include <mg_gpgpu_core/utils.h>
#include <mg_gpgpu_core/parallel_scan.h>
#include <limits>

namespace mg_gpgpu
{

    template<typename T>
    struct EvenPredicate
    {
        __inline__ __device__ static bool predicate( T d_in)
        { return ((d_in % 2) == 0); } 
    };

    template<typename T, typename PREDICATE>
    __global__ void predicate_array_kernel(const T* d_in, bool* d_out, uint32_t count)
    {
        uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        if(tid< count)
        {
            d_out[tid] = PREDICATE::predicate(d_in[tid]); 
        }
    }

    template<typename T, typename PREDICATE>
    void predicate_array(const T* d_in, bool* d_out, uint32_t count)
    {
        uint32_t threads = 1024;
        uint32_t blocks = ((count%threads) != 0)?(count/threads) +1 : (count/threads);
        if (blocks == 0)
        {blocks =1;}

        predicate_array_kernel<T,PREDICATE><<<blocks, threads>>>(d_in, d_out,count);
    }

    template<typename T , typename PREDICATE>
    std::unique_ptr<bool[]> predicate_array_alloc(T* data, uint32_t count)
    {
        T* d_in;
        bool* d_out;
        gpuErrchkDebug(cudaMalloc( (void**)&d_in,  count*sizeof(T)));
        gpuErrchkDebug(cudaMemcpy( d_in, data, count*sizeof(T), cudaMemcpyHostToDevice ));
        gpuErrchkDebug(cudaMalloc( (void**)&d_out,  count*sizeof(bool)));

        predicate_array<T,PREDICATE>(d_in, d_out,count);

        auto ptr =std::unique_ptr<bool[]>(new bool[count]);
        gpuErrchkDebug(cudaMemcpy( ptr.get(), d_out, count*sizeof(bool), cudaMemcpyDeviceToHost));

        cudaFree(d_in);
        cudaFree(d_out);
        return ptr;
    }

    template<typename T , typename PREDICATE>
    std::unique_ptr<T[]> compact_alloc(T* data, uint32_t count)
    {
        T* d_in;
        bool* d_predicate;
        T* d_out;
        gpuErrchkDebug(cudaMalloc( (void**)&d_in,  count*sizeof(T)));
        gpuErrchkDebug(cudaMemcpy( d_in, data, count*sizeof(T), cudaMemcpyHostToDevice ));
        gpuErrchkDebug(cudaMalloc( (void**)&d_predicate,  count*sizeof(bool)));
        gpuErrchkDebug(cudaMalloc( (void**)&d_out,  count*sizeof(T)));

        predicate_array<T,PREDICATE>(d_in, d_predicate,count);
        

        auto ptr =std::unique_ptr<bool[]>(new bool[count]);
        gpuErrchkDebug(cudaMemcpy( ptr.get(), d_out, count*sizeof(bool), cudaMemcpyDeviceToHost));

        cudaFree(d_in);
        cudaFree(d_predicate);
        cudaFree(d_out);
        return ptr;

    }

}//mg_gpgpu
