#include <memory>

namespace mg_gpgpu
{

struct ComputeBin
{
    __device__ static __inline__ int computeBin(uint32_t value)
    { return value%20; }; 

};

template<typename T>
__global__ void global_atomic_histogram_kernel(uint32_t* d_in, uint32_t* d_histo, uint32_t size)
{
    uint32_t t_id =  threadIdx.x + blockDim.x*blockIdx.x;
    uint32_t idx =  T::computeBin(d_in[t_id]);
    atomicAdd(&(d_histo[idx]),1);


}


std::unique_ptr<uint32_t[]> global_atomic_histogram_alloc(uint32_t* host_data, uint32_t count)
{
    using T = uint32_t;
    constexpr uint32_t BIN_SIZE=20;

    T* d_in;
    T* d_histo;
    cudaMalloc( (void**)&d_in,  count*sizeof(T));
    cudaMemcpy( d_in, host_data, count*sizeof(T), cudaMemcpyHostToDevice );

    uint32_t cpu_histo[BIN_SIZE];   
    for (uint32_t i =0; i < BIN_SIZE; ++i)
    { cpu_histo[i] = 0; }
    cudaMalloc( (void**)&d_histo,  BIN_SIZE*sizeof(T));
    cudaMemcpy( d_histo, cpu_histo, BIN_SIZE*sizeof(T), cudaMemcpyHostToDevice );

    uint32_t threads=512;
    uint32_t blocks = ((count%threads) != 0)?(count/threads) +1 : (count/threads);
    global_atomic_histogram_kernel<ComputeBin><<<blocks,threads>>>(d_in, d_histo, count);

    std::unique_ptr<T[]> ptr {new T[BIN_SIZE]};
    cudaMemcpy( ptr.get(), d_histo, BIN_SIZE*sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_histo);
    cudaFree(d_in);
    return ptr;
}





}//mg_gpgpu
