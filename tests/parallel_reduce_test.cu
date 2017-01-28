#include <gmock/gmock.h>
#include <iostream>
#include <mg_gpgpu_core/parallel_reduce.h>
#include <vector>
using namespace testing;

TEST(cuda_parallel_reduce,integer_test )
{
    std::vector<unsigned int> vec; 
    unsigned int size (1024);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i+1; 
    }
	unsigned int res = parallel_reduce<unsigned int>(vec.data(), vec.size());
	auto math_res = (size*(size +1) *0.5);
    ASSERT_TRUE((res )== math_res );
}

TEST(cuda_parallel_reduce,float_test)
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
	float res= parallel_reduce<float >(vec.data(), vec.size());
	float math_res = (size*(size +1) *0.5);
    ASSERT_FLOAT_EQ(res , math_res );
}
