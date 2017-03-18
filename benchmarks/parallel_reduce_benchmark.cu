#include <iostream>
#include <vector>
#include <mg_gpgpu_core/parallel_reduce.h>

using namespace mg_gpgpu;

template <unsigned int ITERATIONS>
void bench_reduce_algs()
{
    std::cout<<"================================================"<<std::endl;
    std::cout<<"                 REDUCE                         "<<std::endl;
    std::cout<<"================================================"<<std::endl;

    const unsigned int DATA_SIZE = 1024*1024;
    //generate random data
    std::vector<float>vec;
    vec.resize(DATA_SIZE);

    for (int i =0; i <DATA_SIZE;++i)
    {
        vec[i] =  static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    //copying/allocating memory to device
    float* in;
    float* out;

    //computing the wanted blocks

    cudaMalloc( (void**)&in,  vec.size()*sizeof(float));
    cudaMalloc( (void**)&out, vec.size()*sizeof(float));
    cudaMemcpy( in, vec.data(),vec.size()*sizeof(float), cudaMemcpyHostToDevice );


    //creating timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    
    float milliseconds = 0;


    cudaEventRecord(start);
    for(int i =0 ; i < ITERATIONS; ++i)
    {
	    parallel_reduce_shared<float >(in,out, vec.size());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"parallel_reduce shared took: "<<
                (milliseconds/ (static_cast<float>(ITERATIONS)))<< " ms"<<std::endl;

    cudaEventRecord(start);
    for(int i =0 ; i < ITERATIONS; ++i)
    {
	    parallel_reduce_shuffle<float >(in,out, vec.size());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"parallel_reduce_shuffle took: "<<
                (milliseconds/ (static_cast<float>(ITERATIONS)))<< " ms"<<std::endl;
        
    cudaEventRecord(start);
    for(int i =0 ; i < ITERATIONS; ++i)
    {
	    parallel_reduce_shuffle_atomic<float >(in,out, vec.size());
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"parallel_reduce_shuffle atomic took: "<<
                (milliseconds/ (static_cast<float>(ITERATIONS)))<< " ms"<<std::endl;


    cudaFree(in);
    cudaFree(out);
}
