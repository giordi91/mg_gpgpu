/*
#include <gmock/gmock.h>
#include <iostream>
#include <mg_gpgpu_core/utils.h>
#include <vector>
using namespace testing;

template<typename T>
struct RandomGenerator 
{
	RandomGenerator()=default;
 
	T operator()() {
		return static_cast<T>(rand() );
	}
};

template<typename T>
inline std::vector<T> gen_random_vector( unsigned int size)
{
    std::vector<T> vec;
    vec.resize(size);
    std::generate(vec.begin(), vec.end(), RandomGenerator<T>());
    return vec;
}
	

TEST(cuda_utils, zero_out_float)
{
    auto&& vec = gen_random_vector<float>(1000);
    auto res = mg_gpgpu::utils::zero_out_alloc<float>(vec.data(), vec.size());    
    for (int i = 0; i < vec.size(); ++i)
    {
        ASSERT_FLOAT_EQ(res[i],0.0f);
    }

}

TEST(cuda_utils, zero_out_int)
{
    auto&& vec = gen_random_vector<int>(1000);
    auto res = mg_gpgpu::utils::zero_out_alloc<int>(vec.data(), vec.size());    
    for (int i = 0; i < vec.size(); ++i)
    {
        ASSERT_EQ(res[i],0);
    }

}
TEST(cuda_utils, zero_out_uint64_t)
{
    auto&& vec = gen_random_vector<uint64_t>(10000);
    auto res = mg_gpgpu::utils::zero_out_alloc<uint64_t>(vec.data(), vec.size());    
    for (int i = 0; i < vec.size(); ++i)
    {
        ASSERT_EQ(res[i],0);
    }

}

*/