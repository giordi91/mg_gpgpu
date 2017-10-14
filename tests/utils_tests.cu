#include <iostream>
#include <mg_gpgpu_core/utils.h>
#include <vector>
#include "catch.hpp"

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
	

TEST_CASE("cuda_utils, zero_out_float" , "[utils]")
{
    auto&& vec = gen_random_vector<float>(1000);
    auto res = mg_gpgpu::utils::zero_out_alloc<float>(vec.data(), vec.size());    
    for (int i = 0; i < vec.size(); ++i)
    {
        REQUIRE(res[i]== Approx(0.0f));
    }

}

TEST_CASE("cuda_utils, zero_out_int" , "[utils]")
{
    auto&& vec = gen_random_vector<int>(1000);
    auto res = mg_gpgpu::utils::zero_out_alloc<int>(vec.data(), vec.size());    
    for (int i = 0; i < vec.size(); ++i)
    {
        REQUIRE(res[i]==0);
    }

}
TEST_CASE("cuda_utils, zero_out_uint64_t" , "[utils]")
{
    auto&& vec = gen_random_vector<uint64_t>(10000);
    auto res = mg_gpgpu::utils::zero_out_alloc<uint64_t>(vec.data(), vec.size());    
    for (int i = 0; i < vec.size(); ++i)
    {
        REQUIRE(res[i] == 0);
    }

}
