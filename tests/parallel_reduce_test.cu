
#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this
                          // in one cpp file

#include <iostream>
#include <vector>
#include <cmath>
#include <mg_gpgpu_core/reduce.h>

#include "catch.hpp"
using mg_gpgpu::parallel_reduce_shared_alloc;
using mg_gpgpu::parallel_reduce_shuffle_alloc;
using mg_gpgpu::parallel_reduce_shuffle_atomic_alloc;

TEST_CASE("int reduce", "[reduce]") {
	REQUIRE(true);
    std::vector<unsigned int> vec; 
    unsigned int size (1024);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
    unsigned int res = parallel_reduce_shared_alloc<unsigned int>(vec.data(), vec.size());
    auto math_res = (size*(size +1) *0.5);
    REQUIRE(res ==math_res );
}

TEST_CASE("integer_numbers_from_1_to_n", "[reduce]") {
    std::vector<unsigned int> vec; 
    unsigned int size (1024);
    vec.resize(size);
    for (int i =0; i <size;++i) {
        vec[i] = i+1; 
    }
    unsigned int res = parallel_reduce_shared_alloc<unsigned int>(vec.data(), vec.size());
    auto math_res = (size*(size +1) *0.5);
    REQUIRE(res == math_res );
}

TEST_CASE("cuda_parallel_reduce_shared,integer_numbers_from_1_to_n_not_power_of_2","[reduce]")
{
    std::vector<unsigned int> vec; 
    unsigned int size (1300);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
    unsigned int res = parallel_reduce_shared_alloc<unsigned int>(vec.data(), vec.size());
    auto math_res = (size*(size +1) *0.5);
    REQUIRE(res == math_res );
}

TEST_CASE("cuda_parallel_reduce_shared,float_numbers_from_1_to_n", "[reduce]")
{
    std::vector<float> vec; 
    unsigned int size (1024);
    vec.resize(size);
    float accum=0;
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
        accum += float((i+1));
    }
    float res= parallel_reduce_shared_alloc<float >(vec.data(), vec.size());
    float math_res = (size*(size +1) *0.5);
    REQUIRE(res == Approx(math_res) );
}

TEST_CASE("cuda_parallel_reduce_shared_float_numbers_random","[reduce]")
{
    std::vector<float> vec; 
    unsigned int size (1024);
    vec.resize(size);

    float accum=0;
    for (int i =0; i <size;++i)
    {
        auto value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        vec[i] = value;
        accum += value;
    }

    float res= parallel_reduce_shared_alloc<float >(vec.data(), vec.size());
    //ASSERT_FLOAT_EQ(res , accum );
    REQUIRE(res ==Approx(accum).margin( 0.001f) );
}

TEST_CASE("cuda_parallel_reduce_shared_float_numbers_random_not_power_of_2","[reduce]")
{
    std::vector<float> vec; 
    unsigned int size (1121);
    vec.resize(size);

    float accum=0;
    for (int i =0; i <size;++i)
    {
        auto value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        vec[i] = value;
        accum += value;
    }

    float res= parallel_reduce_shared_alloc<float >(vec.data(), vec.size());
    REQUIRE(res ==Approx(accum).margin( 0.001f) );
}

TEST_CASE("cuda_parallel_reduce_shuffle_integer_numbers_from_1_to_n","[reduce]")
{
    std::vector<unsigned int> vec; 
    unsigned int size (1024);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
    unsigned int res = parallel_reduce_shuffle_alloc<unsigned int>(vec.data(), vec.size());
    auto math_res = (size*(size +1) *0.5);
    REQUIRE(res == math_res );
}

TEST_CASE("cuda_parallel_reduce_shuffle_integer_numbers_from_1_to_n_not_power_of_2")
{
    std::vector<unsigned int> vec; 
    unsigned int size (1300);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
    unsigned int res = parallel_reduce_shuffle_alloc<unsigned int>(vec.data(), vec.size());
    auto math_res = (size*(size +1) *0.5);
    REQUIRE(res == math_res );
}

TEST_CASE("cuda_parallel_reduce_shuffle_float_numbers_from_1_to_n", "[reduce]")
{
    std::vector<float> vec; 
    unsigned int size (1024);
    vec.resize(size);
    float accum=0;
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
        accum += float((i+1));
    }
    float res= parallel_reduce_shuffle_alloc<float >(vec.data(), vec.size());
    float math_res = (size*(size +1) *0.5);
    REQUIRE(res == math_res );
}

TEST_CASE("cuda_parallel_reduce_shuffle_float_numbers_random","[reduce]")
{
    std::vector<float> vec; 
    unsigned int size (1024);
    vec.resize(size);

    float accum=0;
    for (int i =0; i <size;++i)
    {
        auto value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        vec[i] = value;
        accum += value;
    }

    float res= parallel_reduce_shuffle_alloc<float >(vec.data(), vec.size());
    REQUIRE(res == Approx(accum).margin( 0.001f ));
}

TEST_CASE("cuda_parallel_reduce_shuffle,float_numbers_random_not_power_of_2", "[reduce]")
{
    std::vector<float> vec; 
    unsigned int size (1121);
    vec.resize(size);

    float accum=0;
    for (int i =0; i <size;++i)
    {
        auto value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        vec[i] = value;
        accum += value;
    }

    float res= parallel_reduce_shuffle_alloc<float >(vec.data(), vec.size());
    REQUIRE(res == Approx(accum) );
}


TEST_CASE("cuda_parallel_reduce_shuffle_atomic_integer_numbers_from_1_to_n", "[reduce]")
{
    std::vector<unsigned int> vec; 
    unsigned int size (1024);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
    unsigned int res = parallel_reduce_shuffle_atomic_alloc<unsigned int>(vec.data(), vec.size());
    auto math_res = (size*(size +1) *0.5);
    REQUIRE(res == math_res );
}

TEST_CASE("cuda_parallel_reduce_shuffle_atomic_integer_numbers_from_1_to_n_not_power_of_2", "[reduce]")
{
    std::vector<unsigned int> vec; 
    unsigned int size (1300);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
    unsigned int res = parallel_reduce_shuffle_alloc<unsigned int>(vec.data(), vec.size());
    auto math_res = (size*(size +1) *0.5);
    REQUIRE(res == math_res );
}

//TEST_CASE(cuda_parallel_reduce_shuffle_atomic, float_numbers_from_1_to_n)
//{
//    std::vector<float> vec; 
//    unsigned int size (1024);
//    vec.resize(size);
//    float accum=0;
//    for (int i =0; i <size;++i)
//    {
//        vec[i] = i+1; 
//        accum += float((i+1));
//    }
//    float res= parallel_reduce_shuffle_atomic_alloc<float >(vec.data(), vec.size());
//    float math_res = (size*(size +1) *0.5);
//    ASSERT_FLOAT_EQ(res , math_res );
//}

//TEST_CASE(cuda_parallel_reduce_shuffle_atomic,float_numbers_random)
//{
//    std::vector<float> vec; 
//    unsigned int size (1024);
//    vec.resize(size);

//    float accum=0;
//    for (int i =0; i <size;++i)
//    {
//        auto value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//        vec[i] = value;
//        accum += value;
//    }

//    float res= parallel_reduce_shuffle_atomic_alloc<float >(vec.data(), vec.size());
//    ASSERT_FLOAT_EQ(res , accum );
//}

//TEST(cuda_parallel_reduce_shuffle_atomic,float_numbers_random_not_power_of_2)
//{
//    std::vector<float> vec; 
//    unsigned int size (1921);
//    vec.resize(size);

//    float accum=0;
//    for (int i =0; i <size;++i)
//    {
//        auto value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//        vec[i] = value;
//        accum += value;
//    }

//    float res= parallel_reduce_shuffle_atomic_alloc<float >(vec.data(), vec.size());
//    ASSERT_NEAR(res , accum, 0.001f );

//}

//    template<typename T>
//void full_array_scan_serial( std::vector<T>& data)
//{
//    auto size = data.size();
//    uint32_t lg = log2(float(size)) ;
//    for(int i =0; i<lg; i++ )
//    {
//        for(int id =0; id < (size-1); id += pow(2,(i+1)))
//        {
//            data[id + pow(2,(i+1)) -1] += data[id + pow(2,i) -1];
//        }
//    }
//}

//    template<typename T, int BLOCK_SIZE>
//void run_increasing_reduce_test()
//{
//    std::vector<T > data;
//    std::vector<T> original;
//    uint64_t size = BLOCK_SIZE;


//    data.resize(size);
//    original.resize(size);
//    for (int i =0 ; i <size; ++i)
//    {
//        data[i] = i+1;
//        original[i] = data[i];
//    }

//    auto res =mg_gpgpu::parallel_reduce_full_array_alloc<T>(data.data(), data.size());
//    full_array_scan_serial(data);
//    for (int i =0; i<data.size(); i++)
//    {
//        if (data[i]!= res.get()[i])
//        {
//            std::cout<<"index "<<i <<" "<<data[i]<<" " <<res.get()[i]<<std::endl; 
//        }
//        REQUIRE(data[i], res.get()[i]);
//    }

//}
//    template<typename T, int BLOCK_SIZE>
//void run_random_reduce_test()
//{
//    std::vector<T > data;
//    std::vector<T> original;
//    uint64_t size = BLOCK_SIZE;


//    data.resize(size);
//    original.resize(size);
//    for (int i =0 ; i <size; ++i)
//    {
//        data[i] = rand() % 10;
//        original[i] = data[i];
//    }

//    auto res =mg_gpgpu::parallel_reduce_full_array_alloc<T>(data.data(), data.size());
//    full_array_scan_serial(data);
//    for (int i =0; i<data.size(); i++)
//    {
//        if (data[i]!= res.get()[i])
//        {
//            std::cout<<"index "<<i <<" "<<data[i]<<" " <<res.get()[i]<<std::endl; 
//        }
//        REQUIRE(data[i], res.get()[i]);
//    }

//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_1024 )
//{
//    run_increasing_reduce_test<uint32_t,1024>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_512)
//{
//    run_increasing_reduce_test<uint32_t,512>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_256 )
//{
//    run_increasing_reduce_test<uint32_t,256>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_128 )
//{
//    run_increasing_reduce_test<uint32_t,128>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_64 )
//{
//    run_increasing_reduce_test<uint32_t,64>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_32 )
//{
//    run_increasing_reduce_test<uint32_t,32>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_1024_uint64 )
//{
//    run_increasing_reduce_test<uint64_t,1024>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_512_uint64)
//{
//    run_increasing_reduce_test<uint64_t,512>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_256_uint64 )
//{
//    run_increasing_reduce_test<uint64_t,256>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_128_uint64 )
//{
//    run_increasing_reduce_test<uint64_t,128>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_64_uint64 )
//{
//    run_increasing_reduce_test<uint64_t,64>();
//}


//TEST(cuda_parallel_reduce,full_array_block_reduce_32_uint64 )
//{
//    run_increasing_reduce_test<uint64_t,32>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_random_1024 )
//{
//    run_random_reduce_test<uint32_t,1024>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_random_512)
//{
//    run_random_reduce_test<uint32_t,512>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_random_256 )
//{
//    run_random_reduce_test<uint32_t,256>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_random_128 )
//{
//    run_random_reduce_test<uint32_t,128>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_random_64 )
//{
//    run_random_reduce_test<uint32_t,64>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_random_32 )
//{
//    run_random_reduce_test<uint32_t,32>();
//}

//TEST(cuda_parallel_reduce,full_array_block_reduce_multi_block_1024 )
//{

//    using T = uint32_t;
//    std::vector<T > data;
//    std::vector<T> original;
//    uint64_t size = 1024*10;


//    data.resize(size);
//    original.resize(size);
//    for (int i =0 ; i <size; ++i)
//    {
//        data[i] = (i+1) % 1024;
//        original[i] = data[i];
//    }

//    T* d_in;
//    gpuErrchkDebug(cudaMalloc( (void**)&d_in,  size*sizeof(T)));
//    gpuErrchkDebug(cudaMemcpy( d_in, data.data(), size*sizeof(T), cudaMemcpyHostToDevice ));

//    //kicking only a single block, since this mainly for debugging purpose
//    uint32_t threads = 1024;
//    uint32_t blocks = size/1024;

//    mg_gpgpu::parallel_reduce_full_array_wrap_kernel<T><<<blocks,threads>>>(d_in,size); 

//    auto res =std::unique_ptr<T[]>(new T[size]);
//    gpuErrchkDebug(cudaMemcpy( res.get(), d_in, size*sizeof(T), cudaMemcpyDeviceToHost));



//    //serial hardcoded computation
//    auto c_size = 1024;
//    uint32_t lg = log2(float(c_size)) ;
//    for(int i =0; i<lg; i++ )
//    {
//        for(int id =0; id < (c_size-1); id += pow(2,(i+1)))
//        {
//            data[id + pow(2,(i+1)) -1] += data[id + pow(2,i) -1];
//        }
//    }
//    for (int i =0; i<data.size(); i++)
//    {
//        if (data[i%1024]!= res.get()[i])
//        {
//            std::cout<<"index "<<i <<" "<<data[i]<<" " <<res.get()[i]<<std::endl; 
//            REQUIRE(data[i%1024], res.get()[i]);
//            break;
//        }
//    }
//    cudaFree(d_in);
//}
