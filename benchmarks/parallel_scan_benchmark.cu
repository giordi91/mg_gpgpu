#include <iostream>
#include <vector>
#include <mg_gpgpu_core/parallel_scan.h>

template <unsigned int ITERATIONS>
void bench_scan_algs()
{

    std::cout<<"================================================"<<std::endl;
    std::cout<<"                  SCAN                          "<<std::endl;
    std::cout<<"================================================"<<std::endl;
    std::vector<uint32_t > data;
    std::vector<uint32_t > original;
    uint32_t size = rand() %(1000000) ;

    data.resize(size);
    original.resize(size);
    for (int i =0 ; i <size; ++i)
    {
        data[i] = rand() % 2 + 1;
        original[i] = data[i];
    }
    auto ptr = data.data();
    uint32_t* in;
    uint32_t* out;

    cudaMalloc( (void**)&in,  data.size()*sizeof(uint32_t));
    cudaMalloc( (void**)&out,  data.size()*sizeof(uint32_t));
    cudaMemcpy( in, data.data(),data.size()*sizeof(uint32_t), cudaMemcpyHostToDevice );

    //creating timing stuff
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    
    float milliseconds = 0;

    cudaEventRecord(start);
    for (int i =0; i< ITERATIONS; ++i)
    {
        auto cudares = mg_gpgpu::parallel_scan_hillis_steel<uint32_t>(in,out, size);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"parallel_scan hilliss_steel took: "<<
                (milliseconds/ (static_cast<float>(ITERATIONS)))<< " ms"<<std::endl;


    milliseconds = 0;
    ////computing the wanted blocks
    uint32_t* d_intermediate;
    uint32_t threads = 512;
    uint32_t blocks = ((size%threads) != 0)?(size/threads) +1 : (size/threads);
    //here we have an extra one which will be our atomic value for blocks
    if (blocks == 0)
    {blocks =1;}

    //compute_blocks(threads, blocks,count);
    gpuErrchkDebug(cudaMalloc( (void**)&d_intermediate,  (blocks + 1)*sizeof(uint32_t)));

    constexpr uint32_t SENTINEL = std::numeric_limits<uint32_t>::max();
    cudaEventRecord(start);
    for (int i =0; i< ITERATIONS; ++i)
    {
        //mg_gpgpu::parallel_stream_scan<uint32_t,SENTINEL>(in,d_intermediate, size);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<"parallel stream scan took: "<<
                (milliseconds/ (static_cast<float>(ITERATIONS)))<< " ms"<<std::endl;



    //cudaFree(in);
    //cudaFree(out);
}
