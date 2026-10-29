//#define OPENCL
//#define DEBUG
#include "nn.hpp"
#include "tensor.hpp"
#include <chrono>

void test_matmul()
{
  auto x = Tensor<float>({ { 2, 3 }, { 4, 5 } });
  auto y = Tensor<float>({ { 2, 3, 4 }, { 5, 7, 2 } });
  auto z = x.matmul(y);
  z.backward();
  std::cout << x << std::endl
            << y << std::endl
            << z << std::endl;
}

void test_constructors()
{
  auto x = Tensor<unsigned short>::with_shape({ 2, 3 });
  auto y = Tensor<long>::eye(3);
}

void test_relu()
{
  auto x = Tensor<short>({ { 2, -3, 4 }, { -1, 0, 0 } });
  auto z = nn::ReLU(x);
  z.backward();
  std::cout << x << std::endl
            << z << std::endl;
}

void test_ostream()
{
  auto x = Tensor<short>({ { { 1, 2 }, { 3, 4 }, { 5, 6 } }, { { 1, 2 }, { 3, 4 }, { 5, 6 } }, { { 1, 2 }, { 3, 4 }, { 5, 6 } } });
  std::cout << x;
}

int main()
{
  // opencl_settings::default_platform = 0;
  // auto a = Tensor<float>::uniform({ 2048, 1024 });
  auto x = Tensor<float>::uniform({ 2048, 1024 });
  x.requires_grad(true);
  auto y = Tensor<float>::uniform({ 1024, 1024 });
  y.requires_grad(true);
  auto start = std::chrono::steady_clock::now();
  // for (size_t i = 0; i < 1; ++i) {
  //  x = a * x;
  auto z = x.matmul(y); //.sum();
  z.backward();
  //}
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << std::setw(9) << elapsed.count() << "ms" << std::endl;
  // std::cout << x << std::endl;
}
