#define DEBUG

#include "losses.hpp"
#include <cassert>
#include <chrono>

template <typename T>
bool cmp(std::vector<T> x, std::vector<T> y) {
  const T R = 1000; auto res = true;
  for (size_t i = 0; i < x.size(); ++i)
    if (std::ceil(x[i]*R)/R != std::ceil(y[i]*R)/R)
      res = false;
  return res;
}

#define TEST(x) std::cout << std::setw(10) << std::left << x << " ... " << std::flush; auto start = std::chrono::steady_clock::now();
#define END auto end = std::chrono::steady_clock::now(); auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start); std::cout << "OK, time taken: " << std::setw(10) << std::fixed << std::setprecision(0) << std::right << elapsed.count()/10e3 << "\u00B5s" << std::endl;

void test_MSE() {
  TEST("MSE")
  auto x = Tensor<float>({{2, 3, 1}, {-4, 2, -1}}, true);
  auto y = Tensor<float>({{2, 2, 1}, {-2, 2, -1}});
  auto z = nn::loss::MSE(x, y);
  assert(cmp(z.data().to_vector(), {0.83333}));
  z.backward();
  assert(cmp(x.grad().value().to_vector(), {0, 0.33333, 0, -0.6667, 0, 0}));
  END
}

int main() {
  // force compilation of needed kernels
  auto x = Tensor<float>();
  test_MSE();
}
