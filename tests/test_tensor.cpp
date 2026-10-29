#define DEBUG

#include "tensor.hpp"
#include "nn.hpp"

#include <cassert>
#include <cmath>

template <typename T>
bool cmp(std::vector<T> x, std::vector<T> y) {
  const size_t R = 1000; auto res = true;
  for (size_t i = 0; i < x.size(); ++i)
    if (std::ceil(x[i]*R)/R != std::ceil(y[i]*R)/R) res = false;
  return res;
}

void test_matmul() {
  auto x = Tensor<float>({{2, 3, 4}, {3, 4, 1}});
  x.requires_grad(true);
  auto y = Tensor<float>({{2, 3}, {3, 4}, {4, 2}});
  y.requires_grad(true);
  auto z = x.matmul(y);
  //z = z.exp();
  //z = z.sum().relu() + 3;
  z.backward();
  assert(cmp(x.grad().value().to_vector(), {5, 7, 6, 5, 7, 6}));
  assert(cmp(y.grad().value().to_vector(), {5, 5, 7, 7, 5, 5}));
  std::cout << x << std::endl << y << std::endl << z << std::endl;
}

void test_exp() {
  auto x = Tensor<float>({{1, 3, 2}, {3, 2, 1}}, true);
  auto y = x.exp().sum();
  y.backward();
  assert(cmp(x.grad().value().to_vector(), {2.71828, 20.0855, 7.38906,
    20.0855, 7.38906, 2.71828}));
  std::cout << x << std::endl;
}

void test_broadcast() {
  auto x = Tensor<float>({{1, 2, 3}, {4, 5, 6}}, true);
  auto b = Tensor<float>({1, 2, 3}, true);
  (x + b).sum().backward();
  assert(cmp(x.grad().value().to_vector(), {1, 1, 1, 1, 1, 1}));
  assert(cmp(b.grad().value().to_vector(), {2, 2, 2}));
}

void test_sub() {
  auto x = Tensor<float>({1}, true);
  auto y = Tensor<float>({2}, true);
  (x - y).backward();
  assert(cmp(x.grad().value().to_vector(), {1}));
  assert(cmp(y.grad().value().to_vector(), {-1}));

  x.zero_grad();
  (2 - x).backward();
  assert(cmp(x.grad().value().to_vector(), {-1}));
}

void test_div() {
  auto x = Tensor<float>({6}, true);
  auto y = Tensor<float>({2}, true);
  (x / y).backward();
  assert(cmp(x.grad().value().to_vector(), {0.5}));
  assert(cmp(y.grad().value().to_vector(), {-1.5}));

  y.zero_grad();
  (6 / y).backward();
  assert(cmp(y.grad().value().to_vector(), {-1.5}));
}

int main() {
  auto x = Tensor<float>();
  test_matmul();
  test_exp();
  test_broadcast();
  test_sub();
  test_div();
}
