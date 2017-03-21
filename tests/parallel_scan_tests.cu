#include <gmock/gmock.h>
#include <iostream>
#include <fstream>
#include <mg_gpgpu_core/parallel_scan.h>
#include <vector>
using namespace testing;

//commented for the time being, need to find a way to make a custom target for it
template<typename T>
void inclusive_scan(std::vector<T>& data)
{

    for (int i =1; i<data.size(); ++i)
    {
        data[i] += data[i-1];    
    }

}

TEST(cuda_parallel_scan, hillis_steel_uint32_t)
{
    std::vector<uint32_t > data;
    std::vector<uint32_t > original;
    uint32_t size = rand() %(100000) ;

    data.resize(size);
    original.resize(size);
    for (int i =0 ; i <size; ++i)
    {
        data[i] = rand() % 2 + 1;
        original[i] = data[i];
    }

    auto ptr = data.data();
    auto cudares = mg_gpgpu::parallel_scan_hillis_steel_alloc<uint32_t>(ptr, size);
    inclusive_scan<uint32_t>(data);
    for(int i =1; i < size; ++i)
    {
       ASSERT_TRUE( data[i] == cudares[i]);
    }
}

TEST(cuda_parallel_scan, hillis_steel_uint64_t)
{
    std::vector<uint64_t > data;
    std::vector<uint64_t > original;
    uint64_t size = rand() %(100000) ;

    data.resize(size);
    original.resize(size);
    for (int i =0 ; i <size; ++i)
    {
        data[i] = rand() % 2 + 1;
        original[i] = data[i];
    }

    auto ptr = data.data();
    auto cudares = mg_gpgpu::parallel_scan_hillis_steel_alloc<uint64_t>(ptr, size);
    inclusive_scan<uint64_t>(data);
    for(int i =1; i < size; ++i)
    {
        if (data[i] != cudares[i])
        {
            std::cout<<data[i] << " "<<cudares[i]<<std::endl; 
        }
       ASSERT_TRUE( data[i] == cudares[i]);
    }
}


TEST(cuda_parallel_scan,steam_scan_inclusive_32_bit_int )
{
    using lint = uint32_t;
    std::vector<lint > data;
    std::vector<lint> original;
    uint64_t size = rand() %(10000000) ;

    data.resize(size);
    original.resize(size);
    for (int i =0 ; i <size; ++i)
    {
        data[i] = rand() % 2 + 1;
        original[i] = data[i];
    }


    //uint32_t bs = 512;
    //uint32_t accum =0;
    //uint32_t mult =1;
    ////for(int i =0; i < size; ++i)
    ////{
    ////    if (!( i< (bs*mult)))
    ////    {
    ////        std::cout<<" "<<accum<< " "; 
    ////        accum=0;
    ////        mult++;
    ////    }
    ////    accum += data[i];
    ////    if (mult > 5)
    ////    {
    ////        break; 
    ////    }
    ////}
    //for(int i =0; i <512; ++i)
    //{
    //    accum += data[i]; 
    //}
    //std::cout<<accum<<" ";
    //accum=0;
    //for(int i =512; i <1024; ++i)
    //{
    //    accum += data[i]; 
    //}
    //std::cout<<accum<<" ";
    //accum=0;
    //for(int i =1024; i <1536; ++i)
    //{
    //    accum += data[i]; 
    //}
    //std::cout<<accum<<" ";
    //accum=0;
    //for(int i =1536; i <2048; ++i)
    //{
    //    accum += data[i]; 
    //}
    //std::cout<<accum<<" "<<std::endl;

    //std::cout<<"cpu: ";
    //for(int i =0; i <= bs; ++i)
    //{

    //    if (!( i< (32*mult)))
    //    {
    //        std::cout<<" "<<accum<< " "; 
    //        accum=0;
    //        mult++;
    //    }

    //        accum += data[i];
    //    
    //    if (mult > 17)
    //    {
    //        break; 
    //    }
    //}
    std::cout<<std::endl;
    auto ptr = data.data();
    auto cudares = mg_gpgpu::parallel_stream_scan_alloc<lint>(ptr, size);
    inclusive_scan<lint>(data);
    for(int i =1; i < size; ++i)
    {
        if (data[i] != cudares[i])
        {
            std::cout<<"ERROR index "<<i<< " "<<data[i] << " "<<cudares[i]<<std::endl; 
        }
       ASSERT_TRUE( data[i] == cudares[i]);
    }
}
//TEST(cuda_parallel_scan,steam_scan_inclusive_64_bit_int )
//{
//    using lint = unsigned long long int;
//    std::vector<lint > data;
//    std::vector<lint> original;
//    uint64_t size = rand() %(100000) ;
//
//    data.resize(size);
//    original.resize(size);
//    for (int i =0 ; i <size; ++i)
//    {
//        data[i] = rand() % 2 + 1;
//        original[i] = data[i];
//    }
//
//    auto ptr = data.data();
//    auto cudares = mg_gpgpu::parallel_stream_scan_alloc<lint>(ptr, size);
//    inclusive_scan<lint>(data);
//    for(int i =1; i < size; ++i)
//    {
//        if (data[i] != cudares[i])
//        {
//            std::cout<<data[i] << " "<<cudares[i]<<std::endl; 
//        }
//       ASSERT_TRUE( data[i] == cudares[i]);
//    }
//}



//TEST(cuda_parallel_scan,blelloc_block_scan_uint32_t_1024_block )
//{
//    using lint = uint32_t;
//    std::vector<lint > data;
//    std::vector<lint> original;
//    uint64_t size = rand() %(10000000) ;
//    std::cout<<"size of array "<<size<<std::endl;
//
//
//    data.resize(size);
//    original.resize(size);
//    for (int i =0 ; i <size; ++i)
//    {
//        data[i] = rand() % 2 + 1;
//        original[i] = data[i];
//    }
//
//}
