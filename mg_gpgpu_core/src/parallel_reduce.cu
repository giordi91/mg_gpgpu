#include <iostream>

__global__ void global_reduce_kernel(unsigned int * d_out, unsigned int * d_in, unsigned int count)
{
    extern __shared__ unsigned int sdata[];
    

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    if (myId >= count)
    { sdata[tid] = 0;}
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


unsigned int parallel_reduce( unsigned int* host_data, unsigned int element_count)
{
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

    unsigned int block_size = 1024;
    unsigned int * device_data;
    unsigned int * out_device_data;

    unsigned int blocks = ((element_count%block_size) != 0)?(element_count/block_size) +1:
                                                            (element_count/block_size);
    unsigned int size = blocks * block_size* sizeof(unsigned int);

   

    std::cout<<"element count allocated"<< size/sizeof(unsigned int)<<std::endl;
    cudaMalloc( (void**)&device_data, size );
    cudaMalloc( (void**)&out_device_data, size );
    cudaMemcpy( device_data , host_data, size, cudaMemcpyHostToDevice );

    unsigned int * in = out_device_data;
    unsigned int * out = device_data;
    cudaEventRecord(start);
    for(int i =element_count; i>1; )
    {

        std::swap(in,out);
        //unsigned int * tmp = in;
        //in = out;
        //out = tmp;
        unsigned int ratio =i/ block_size;
        unsigned int grid_size = ((i%block_size) != 0)?(ratio) +1: (ratio);
        std::cout<<"grid_size "<<grid_size<<std::endl;
        dim3 dimBlock( block_size, 1 );
        dim3 dimGrid( grid_size, 1 );
        global_reduce_kernel<<<grid_size, block_size, sizeof(unsigned int)*element_count >>>( out, in, i);
        i = grid_size;
       
       
    }
    unsigned int result;
    cudaMemcpy(&result, out,sizeof(unsigned int), cudaMemcpyDeviceToHost);

cudaEventRecord(stop);
cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
std::cout<<"cuda took milliseconds  :"<<milliseconds<<std::endl;
    cudaFree(device_data);
    cudaFree(out_device_data);
    return result;

}

