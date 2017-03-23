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
template<typename T,int BLOCK_SIZE>
void test_blelloc_block_scan()
{
    std::vector<T > data;
    std::vector<T> original;
    uint32_t size = BLOCK_SIZE;

    data.resize(size);
    original.resize(size);
    for (int i =0 ; i <size; ++i)
    {
        data[i] = i+1;
        original[i] = data[i];
    }

    auto cudares = mg_gpgpu::parallel_scan_blelloch_alloc<T>(data.data(), data.size());
    inclusive_scan<T>(data);

    for(int i =1; i < data.size(); ++i)
    {
        if (data[i-1] != cudares[i])
        {
            std::cout<<"ERROR index "<<i<< " "<<data[i] << " "<<cudares[i]<<std::endl; 
        }
       ASSERT_TRUE( data[i-1] == cudares[i]);
    }


}

TEST(cuda_parallel_scan,blelloc_block_scan_uint32_t_1024_block )
{
    test_blelloc_block_scan<uint32_t,1024>();
}

TEST(cuda_parallel_scan,blelloc_block_scan_uint32_t_512_block )
{
    test_blelloc_block_scan<uint32_t,512>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint32_t_256_block )
{
    test_blelloc_block_scan<uint32_t,256>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint32_t_128_block )
{
    test_blelloc_block_scan<uint32_t,128>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint32_t_64_block )
{
    test_blelloc_block_scan<uint32_t,64>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint32_t_32_block )
{
    test_blelloc_block_scan<uint32_t,32>();

}

//TODO check why 64 bit doesnt work
TEST(cuda_parallel_scan,blelloc_block_scan_uint64_t_1024_block )
{
    test_blelloc_block_scan<uint64_t ,1024>();
}

TEST(cuda_parallel_scan,blelloc_block_scan_uint64_t_512_block )
{
    test_blelloc_block_scan<uint64_t,512>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint64_t_256_block )
{
    test_blelloc_block_scan<uint64_t,256>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint64_t_128_block )
{
    test_blelloc_block_scan<uint64_t,128>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint64_t_64_block )
{
    test_blelloc_block_scan<uint64_t,64>();
}
TEST(cuda_parallel_scan,blelloc_block_scan_uint64_t_32_block )
{
    test_blelloc_block_scan<uint64_t,32>();
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
