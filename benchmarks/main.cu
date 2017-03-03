#include <iostream>
#include <parallel_reduce_benchmark.cu>
#include <parallel_scan_benchmark.cu>


int main()
{
    const unsigned int ITERATIONS = 500;
    bench_reduce_algs<ITERATIONS>();
    bench_scan_algs<ITERATIONS>();
    return 0;
}
