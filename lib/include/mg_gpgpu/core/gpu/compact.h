#pragma once
#include <cuda_runtime_api.h>
#include <limits>
#include <memory>
#include <mg_gpgpu/core/gpu/scan.h>
#include <mg_gpgpu/utils.h>

namespace mg_gpgpu {

/** Trick to pass a function as template parameter through a type */
// TODO(giordi) investigate if there is a way to pass the function drirectly
template <typename T> struct EvenPredicate {
  __inline__ __device__ static bool predicate(T d_in) {
    return ((d_in % 2) == 0);
  }
};

template <typename T> struct IsZeroPredicate {
  __inline__ __device__ static bool predicate(T d_in) { return (d_in == 0); }
};

/**
 *This is a simple kernel which performs a simple predicate on each of the given
 *element of the array, result is an int rather than a bool so it s easier to
 *use it later on on a scan
 *@parm d_in: input device pointer
 *@parm d_out: output device pointer
 *@parm count : size of the buffer
 */
template <typename T, typename PREDICATE>
__global__ void predicate_array_kernel(const T *d_in, uint32_t *d_out,
                                       uint32_t count) {
  // TODO(giordi) might be worth to have a grid stride algorithm
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < count) {
    d_out[tid] = int(PREDICATE::predicate(d_in[tid]));
  }
}

/**
 * @brief cpu wrap for the predicate_array_kernel
 * This function kicks the predicate array kenrnel on the input data
 *@parm d_in: input device pointer
 *@parm d_out: output device pointer
 *@parm count : size of the buffer
 */
template <typename T, typename PREDICATE>
void predicate_array(const T *d_in, uint32_t *d_out, uint32_t count) {
  // TODO(giordi) change this to be configurable
  uint32_t threads = 1024;
  uint32_t blocks =
      ((count % threads) != 0) ? (count / threads) + 1 : (count / threads);
  blocks = blocks == 0 ? 1 : blocks;

  predicate_array_kernel<T, PREDICATE><<<blocks, threads>>>(d_in, d_out, count);
}

/**
@brief compacts the given data in a secondary buffer
* This kernel uses the data from the predicate and address arrays
* to copy the needed data
*@parm d_in: input device pointer
*@parm d_addresses: array holding the indeces of destination for the data,
                    generally this array compes from a scanned predicate array
*@param d_predicate: the predicated array, will let us chose wheter or not we want to
                     copy the data
*@parm d_out: output device pointer
*@parm count : size of the buffer
*/
template <typename T>
__global__ void compact_to_location(const T *d_in, const uint32_t *d_addresses,
                                    const uint32_t *d_predicate, T *d_out,
                                    uint32_t count) {
  // TODO(giordi) might be worth to have a grid stride algorithm
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < count) {
    if (d_predicate[tid]) {
      d_out[d_addresses[tid]] = d_in[tid];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
//  ALLOCS VARIANTS
//////////////////////////////////////////////////////////////////////////////////////
template <typename T, typename PREDICATE>
std::unique_ptr<T[]> compact_alloc(T *data, uint32_t count) {
  T *d_in;
  uint32_t *d_predicate;
  uint32_t *d_scan;
  T *d_out;
  gpuErrchkDebug(cudaMalloc((void **)&d_in, count * sizeof(T)));
  gpuErrchkDebug(
      cudaMemcpy(d_in, data, count * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchkDebug(cudaMalloc((void **)&d_predicate, count * sizeof(uint32_t)));
  gpuErrchkDebug(cudaMalloc((void **)&d_scan, count * sizeof(uint32_t)));
  gpuErrchkDebug(cudaMalloc((void **)&d_out, count * sizeof(T)));

  uint32_t threads = 1024;
  uint32_t blocks =
      ((count % threads) != 0) ? (count / threads) + 1 : (count / threads);
  blocks = blocks == 0 ? 1 : blocks;
  // performing the predicate
  predicate_array<T, PREDICATE>(d_in, d_predicate, count);
  // TODO(giordi) having a version of exclusive scan that is not in place would 
  // allow us to avoid the specific copy
  // need to make a copy of the predicate
  utils::copy_kernel<<<blocks, threads>>>(d_predicate, d_scan, count);
  // performing the scan
  parallel_scan_blelloch<uint32_t>(d_scan, count);

  compact_to_location<T>
      <<<blocks, threads>>>(d_in, d_scan, d_predicate, d_out, count);

  auto ptr = std::unique_ptr<uint32_t[]>(new uint32_t[count]);
  gpuErrchkDebug(cudaMemcpy(ptr.get(), d_out, count * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_predicate);
  cudaFree(d_out);
  return ptr;
}

template <typename T, typename PREDICATE>
std::unique_ptr<uint32_t[]> predicate_array_alloc(T *data, uint32_t count) {
  T *d_in;
  uint32_t *d_out;
  gpuErrchkDebug(cudaMalloc((void **)&d_in, count * sizeof(T)));
  gpuErrchkDebug(
      cudaMemcpy(d_in, data, count * sizeof(T), cudaMemcpyHostToDevice));
  gpuErrchkDebug(cudaMalloc((void **)&d_out, count * sizeof(uint32_t)));

  predicate_array<T, PREDICATE>(d_in, d_out, count);

  auto ptr = std::unique_ptr<uint32_t[]>(new uint32_t[count]);
  gpuErrchkDebug(cudaMemcpy(ptr.get(), d_out, count * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost));

  cudaFree(d_in);
  cudaFree(d_out);
  return ptr;
}
} // namespace mg_gpgpu
