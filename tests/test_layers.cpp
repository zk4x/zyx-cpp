#define DEBUG

#include "layers.hpp"
#include "activations.hpp"
#include <cassert>
#include <chrono>

template <typename T>
bool cmp(std::vector<T> x, std::vector<T> y) {
  const T R = 1000; auto res = true;
  for (size_t i = 0; i < x.size(); ++i) {
    //std::cout << std::ceil(x[i]*R) << " cmp: " << std::ceil(y[i]*R) << std::endl;
    if (std::ceil(x[i]*R) != std::ceil(y[i]*R))
      res = false;
  }
  return res;
}

#define TEST(x) std::cout << std::setw(10) << std::left << x << " ... " << std::flush; auto start = std::chrono::steady_clock::now();
#define END auto end = std::chrono::steady_clock::now(); auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start); std::cout << "OK, time taken: " << std::setw(10) << std::fixed << std::setprecision(0) << std::right << elapsed.count()/10e3 << "\u00B5s" << std::endl;

void test_Linear() {
  TEST("Linear")
  auto x = Tensor<float>({{2, 4}, {2, -3}, {2, 1}}, true);
  auto l1 = nn::layer::Linear<float>(2, 2);
  auto z = l1(x);
  assert((z.shape() == std::vector<size_t>{3, 2}));
  z.sum().backward();
  assert((x.grad().value().shape() == std::vector<size_t>{3, 2}));
  assert(l1.params().size() == 2);
  assert((l1.params()[0].shape() == std::vector<size_t>{2, 2}));
  assert((l1.params()[1].shape() == std::vector<size_t>{2}));
  END
}

void test_Tanh() {
  TEST("Tanh")
  auto x = Tensor<float>({{{2, 4}, {2, -3}, {2, 1}}, {{-1, 3}, {4, -5}, {6, 1}}}, true);
  auto z = nn::Tanh(x);
  assert(cmp(z.data().to_vector(), {0.96403, 0.99933, 0.96403, -0.99505, 0.96403, 0.76159,
    -0.76159, 0.99505, 0.99933, -0.99991, 0.99999, 0.76159}));
  z.sum().backward();
  assert(cmp(x.grad().value().to_vector(), {0.07065, 0.00134, 0.07065, 0.00988, 0.07065, 0.41998,
    0.41998, 0.00988, 0.00134, 0.00018, 0.00002, 0.41998}));
  END
}

void test_Dropout() {
  TEST("Dropout")
  srand(1);
  auto x = Tensor<float>({{2, 4}, {2, -3}, {2, 1}}, true);
  auto z = nn::layer::Dropout<float>(0)(x);
  assert(cmp(z.data().to_vector(), {0, 0, 0, 0, 0, 0}));
  z.sum().backward();
  assert(cmp(x.grad().value().to_vector(), {0, 0, 0, 0, 0, 0}));
  END
}

int main() {
  auto x = Tensor<float>();
  test_Linear();
  test_Tanh();
  test_Dropout();
}
