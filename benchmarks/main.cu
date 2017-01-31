#include <iostream>
#include <parallel_reduce_benchmark.cu>


int main()
{
    const unsigned int ITERATIONS = 50;
    bench_reduce_algs<ITERATIONS>();
    return 0;
}
