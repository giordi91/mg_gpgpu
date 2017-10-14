#include <cmath>
#include <iostream>
#include <mg_gpgpu_core/compact.h>
#include <vector>

#include "catch.hpp"
using mg_gpgpu::compact_alloc;
using mg_gpgpu::EvenPredicate;
using mg_gpgpu::IsZeroPredicate;
using mg_gpgpu::predicate_array_alloc;

TEST_CASE("predicate is equal to zero", "[compact]") {
  std::vector<uint32_t> vec;
  uint32_t size(1024);
  vec.resize(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = i % 3;
  }
  uint32_t *data = vec.data();
  auto res =
      predicate_array_alloc<uint32_t, IsZeroPredicate<uint32_t>>(data, size);
  REQUIRE(res[0] == 1ul);
  for (uint32_t i = 1; i < size; ++i) {
    if ((i % 3) == 0) {
      REQUIRE(res[i] == 1ul);
    } else {
      REQUIRE(res[i] == 0ul);
    }
  }
}
TEST_CASE("compact is is even", "[compact]") {
  std::vector<uint32_t> vec;
  uint32_t size(1024);
  vec.resize(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = i;
  }
  uint32_t *data = vec.data();
  auto res = compact_alloc<uint32_t, EvenPredicate<uint32_t>>(data, size);

  uint32_t dataToLoop = size / 2;
  for (uint32_t i = 0; i < dataToLoop; ++i) {
    REQUIRE(res[i] == i * 2);
  }
}
