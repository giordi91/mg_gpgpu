#pragma once
#include <memory>
#include <mg_gpgpu_core/utils.h>



__global__ void parallel_scan_hillis_steel_kernel(unsigned int * in,unsigned int* out, unsigned int hop, unsigned int count)
{
    unsigned int tid =threadIdx.x + (blockDim.x * blockIdx.x);
    if( tid < count   )
    {
		if(tid >= hop)
		{
        	out[tid] = in[tid] + in[tid-hop];
		}
		else
		{
			out[tid] = in[tid];
		}
    }    
}

__global__ void copy(unsigned int * in , unsigned int * out,unsigned int count)
{
    unsigned int tid =threadIdx.x + (blockDim.x * blockIdx.x);
    if( tid < count  )
    {
        out[tid] = in[tid];
    }
}


std::unique_ptr<unsigned int[]> parallel_scan(unsigned int* data, unsigned int count)
{
    unsigned int* in;
    unsigned int* out;
    gpuErrchk(cudaMalloc( (void**)&in, count*sizeof(unsigned int)));
    gpuErrchk(cudaMalloc( (void**)&out, count*sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy( in, data, count*sizeof(unsigned int ), cudaMemcpyHostToDevice ));


    ////computing the wanted blocks
    unsigned int* t_in = out;;
    unsigned int* t_out = in;
    unsigned int threads = 512;
    unsigned int blocks = ((count/threads) != 0)?(count/threads) +1 : (count/threads);
    if (blocks ==0)
    {
        blocks =1;
    }
    for (int hop =1; hop<count;hop<<=1)
    {
        
        std::swap(t_in, t_out);
        std::cout<< hop <<" "<<(count/(hop))<<std::endl;
        parallel_scan_hillis_steel_kernel<<<blocks, threads>>>(t_in,t_out, hop, count);
        gpuErrchk( cudaPeekAtLastError() );
    }

    auto ptr =std::unique_ptr<unsigned int[]>(new unsigned int[count]);
    gpuErrchk(cudaMemcpy( ptr.get(), t_out, count*sizeof(unsigned int ), cudaMemcpyDeviceToHost));
    cudaFree(in);
    cudaFree(out);
    return ptr;

}
