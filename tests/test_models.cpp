#define DEBUG

#include "models.hpp"
#include <cassert>
#include <chrono>

#define TEST(x) std::cout << std::setw(10) << std::left << x << " ... " << std::flush; auto start = std::chrono::steady_clock::now();
#define END auto end = std::chrono::steady_clock::now(); auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start); std::cout << "OK, time taken: " << std::setw(10) << std::fixed << std::setprecision(0) << std::right << elapsed.count()/10e3 << "\u00B5s" << std::endl;

void test_Sequential() {
  TEST("Sequential")
  auto model = nn::model::Sequential<float>({
    new nn::layer::Linear<float>(2, 3),
    new nn::layer::Linear<float>(3, 1)
  });
  auto x = Tensor<float>({{2, 4}, {2, -3}}, true);
  auto z = model(x);
  assert((z.shape() == std::vector<size_t>{2, 1}));
  assert(model.params().size() == 4);
  z.sum().backward();
  assert(x.requires_grad());
  END
}

int main() {
  auto x = Tensor<float>();
  test_Sequential();
}
