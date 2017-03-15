#include <gmock/gmock.h>
#include <iostream>
#include <mg_gpgpu_core/parallel_reduce.h>
#include <vector>
using namespace testing;

using mg_gpgpu::parallel_reduce_shared_alloc;
using mg_gpgpu::parallel_reduce_shuffle_alloc;
using mg_gpgpu::parallel_reduce_shuffle_atomic_alloc;

TEST(cuda_parallel_reduce_shared,integer_numbers_from_1_to_n)
{
    std::vector<unsigned int> vec; 
    unsigned int size (1024);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
	unsigned int res = parallel_reduce_shared_alloc<unsigned int>(vec.data(), vec.size());
	auto math_res = (size*(size +1) *0.5);
    ASSERT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shared,integer_numbers_from_1_to_n_not_power_of_2)
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
    ASSERT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shared,float_numbers_from_1_to_n)
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
    ASSERT_FLOAT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shared,float_numbers_random)
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
    ASSERT_FLOAT_EQ(res , accum );
}

TEST(cuda_parallel_reduce_shared,float_numbers_random_not_power_of_2)
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
    ASSERT_FLOAT_EQ(res , accum );
}

TEST(cuda_parallel_reduce_shuffle,integer_numbers_from_1_to_n)
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
    ASSERT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shuffle, integer_numbers_from_1_to_n_not_power_of_2)
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
    ASSERT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shuffle, float_numbers_from_1_to_n)
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
    ASSERT_FLOAT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shuffle,float_numbers_random)
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
    ASSERT_FLOAT_EQ(res , accum );
}

TEST(cuda_parallel_reduce_shuffle,float_numbers_random_not_power_of_2)
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
    ASSERT_FLOAT_EQ(res , accum );
}


TEST(cuda_parallel_reduce_shuffle_atomic,integer_numbers_from_1_to_n)
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
    ASSERT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shuffle_atomic, integer_numbers_from_1_to_n_not_power_of_2)
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
    ASSERT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shuffle_atomic, float_numbers_from_1_to_n)
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
	float res= parallel_reduce_shuffle_atomic_alloc<float >(vec.data(), vec.size());
	float math_res = (size*(size +1) *0.5);
    ASSERT_FLOAT_EQ(res , math_res );
}

TEST(cuda_parallel_reduce_shuffle_atomic,float_numbers_random)
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

	float res= parallel_reduce_shuffle_atomic_alloc<float >(vec.data(), vec.size());
    ASSERT_FLOAT_EQ(res , accum );
}

TEST(cuda_parallel_reduce_shuffle_atomic,float_numbers_random_not_power_of_2)
{
    std::vector<float> vec; 
    unsigned int size (1921);
    vec.resize(size);

	float accum=0;
    for (int i =0; i <size;++i)
    {
        auto value = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		vec[i] = value;
		accum += value;
    }

	float res= parallel_reduce_shuffle_atomic_alloc<float >(vec.data(), vec.size());
    ASSERT_NEAR(res , accum, 0.001f );

}

