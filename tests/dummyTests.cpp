#include <gmock/gmock.h>
#include <iostream>
#include <mg_gpgpu_core/dummy.h>
#include <mg_gpgpu_core/parallel_reduce.h>
#include <vector>
using namespace testing;



TEST(Attribute_test, basic_instances)
{
	test();
}

TEST(Attribute_test, cuda_run)
{
    std::vector<unsigned int> vec; 
    unsigned int size (1024*1024);
    vec.resize(size);
    for (int i =0; i <size;++i)
    {
        vec[i] = i; 
    }
	parallel_reduce(vec.data(), vec.size());
}
