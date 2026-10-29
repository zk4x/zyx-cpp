//#define OPENCL
#define DEBUG

#include "buffer.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <functional>
#include <chrono>

template <typename T>
bool cmp(std::vector<T> x, std::vector<T> y) {
  const size_t R = 1000; auto res = true;
  for (size_t i = 0; i < x.size(); ++i)
    if (std::ceil(x[i]*R)/R != std::ceil(y[i]*R)/R) res = false;
  return res;
}

#define TEST(x) std::cout << std::setw(10) << std::left << x << " ... " << std::flush; auto start = std::chrono::steady_clock::now();
#define END auto end = std::chrono::steady_clock::now(); auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start); std::cout << "OK, time taken: " << std::setw(10) << std::fixed << std::setprecision(0) << std::right << elapsed.count()/10e3 << "\u00B5s" << std::endl;

void test_constructors() {
  TEST("Constructors")
  // in order to use other types than flaot,
  // you must change DTYPE const in header file
  auto x1 = Buffer<float>::with_shape({2, 3}, 2);
  //std::cout << x1 << std::endl;
  assert(x1.to_vector()[2] == 2);
  auto x2 = Buffer<float>::uniform({4, 5});
  //std::cout << x2 << std::endl;
  assert(x2.shape()[1] == 5);
  auto x3 = Buffer<int>::eye(5);
  //std::cout << x3 << std::endl;
  assert(x3.to_vector()[6] == 1);
  assert(x3.to_vector()[8] == 0);
  auto x4 = Buffer<float>({{2, 3, 4}, {9, 2, 5}});
  assert(x4.to_vector()[4] == 2);
  assert(x4.shape()[0] == 2);
  END
}

void test_methods() {
  TEST("Methods")
  auto x1 = Buffer<float>::with_shape({1}, 1);
  //std::cout << x1 << std::endl;
  x1.reshape({1, 1});
  //std::cout << x1 << std::endl;
  END
}

void test_transpose() {
  TEST("Transpose");
  auto x = Buffer<float>::with_shape({2, 7}, 3);
  auto z = x.transpose();
  assert(cmp(z.shape(), {7, 2}));
  auto x1 = Buffer<float>({{2, 4}, {2, 3}, {2, 1}, {1, 3}, {4, 5}, {6, 1}});
  auto z1 = x1.transpose();

  assert(cmp(z1.to_vector(), {2, 2, 2, 1, 4, 6, 4, 3, 1, 3, 5, 1}));
  auto x2 = Buffer<float>::with_shape({5, 7}, 3);
  auto z2 = x2.transpose();
  assert((z2.shape() == std::vector<size_t>{7, 5}));
  END
}

void test_exp() {
  TEST("Exp");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto z = x.exp();
  auto res = x.to_vector();
  for (auto &v : res)
    v = std::exp(v);
  assert(cmp(z.to_vector(), res));
  END
}

void test_ln() {
  TEST("Ln");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto z = x.ln();
  auto res = x.to_vector();
  for (auto &v : res)
    v = std::log(v);
  assert(cmp(z.to_vector(), res));
  END
}

void test_sqrt() {
  TEST("Sqrt");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto z = x.sqrt();
  auto res = x.to_vector();
  for (auto &v : res)
    v = std::sqrt(v);
  assert(cmp(z.to_vector(), res));
  END
}

void test_inv() {
  TEST("Inv");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto z = x.inv();
  assert(cmp(z.to_vector(), {0.5, 0.25, 0.5, 0.3333, 0.5, 1, 1, 0.3333, 0.25, 0.2, 0.1666, 1}));
  END
}

void test_max() {
  TEST("Max");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto z = x.max();
  assert(cmp(z.to_vector(), {6}));
  END
}

void test_sum() {
  TEST("Sum");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto z = x.sum();
  assert(cmp(z.to_vector(), {34}));

  auto x4 = Buffer<float>({{2, 3, 4}, {5, 1, 3}});
  auto z4 = x4.sum();
  assert((z4.to_vector()[0] == 18));
  END
}

void test_add() {
  TEST("Add");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto y = Buffer<float>({{{1, 10}, {8, 2}, {8, 0}}, {{-3, 4}, {-89, 9}, {8, 0}}});
  auto z = x + y;
  assert(cmp(z.to_vector(), {3, 14, 10, 5, 10, 1, -2, 7, -85, 14, 14, 1}));

  auto x2 = Buffer<float>({{2, 3, 4}, {5, 1, 3}});
  auto y2 = Buffer<float>({{2, 1, 7}, {0, 1, 9}});
  auto z2 = x2 + y2;
  assert((z2.to_vector() == std::vector<float>{4, 4, 11, 5, 2, 12}));
  END
}

void test_sub() {
  TEST("Sub");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto y = Buffer<float>({{{1, 10}, {8, 2}, {8, 0}}, {{-3, 4}, {-89, 9}, {8, 0}}});
  auto z = x - y;
  assert(cmp(z.to_vector(), {1, -6, -6, 1, -6, 1, 4, -1, 93, -4, -2, 1}));
  END
}

void test_mul() {
  TEST("Mul");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto y = Buffer<float>({{{1, 10}, {8, 2}, {8, 0}}, {{-3, 4}, {-89, 9}, {8, 0}}});
  auto z = x * y;
  assert(cmp(z.to_vector(), {2, 40, 16, 6, 16, 0, -3, 12, -356, 45, 48, 0}));

  auto x3 = Buffer<float>({{2, 3, 4}, {5, 1, 3}});
  auto y3 = Buffer<float>({{2, 1, 7}, {0, 1, 9}});
  auto z3 = x3 * y3;
  assert((z3.to_vector() == std::vector<float>{4, 3, 28, 0, 1, 27}));
  END
}

void test_div() {
  TEST("Div");
  auto x = Buffer<float>({{{2, 4}, {2, 3}, {2, 1}}, {{1, 3}, {4, 5}, {6, 1}}});
  auto y = Buffer<float>({{{1, 10}, {8, 2}, {8, 5}}, {{-4, 4}, {-1, 9}, {8, 1}}});
  auto z = x / y;
  assert(cmp(z.to_vector(), {2, 0.4, 0.25, 1.5, 0.25, 0.2, -0.25, 0.75, -4, 0.5555, 0.75, 1}));
  END
}

void test_matmul() {
  TEST("MatMul");
  auto x5 = Buffer<short>({{2, 3, 4}, {3, 4, 1}});
  auto y5 = Buffer<short>({{2, 3}, {3, 4}, {4, 2}});
  auto z5 = x5.matmul(y5);
  std::cout << z5 << std::endl;
  assert((z5.to_vector()[3] == 27));
  END
}

int main() {
  // force compilation of kernels for some basic types
  auto x = Buffer<float>();
  auto y = Buffer<short>();
  auto z = Buffer<int>();
  std::cout << std::endl;
  test_constructors();
  test_methods();
  test_transpose();
  test_exp();
  test_ln();
  test_sqrt();
  test_inv();
  test_max();
  test_sum();
  test_add();
  test_sub();
  test_mul();
  test_div();
  test_matmul();
}

