//#define OPENCL
#include "tensor.hpp"
#include <chrono>

int main()
{
  auto x = Tensor<float>({{2, 3, 4}, {3, 4, 5}});
  x.requires_grad(true);
  auto y = Tensor<float>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
  y.requires_grad(true);
  auto start = std::chrono::steady_clock::now();
  auto z = x.matmul(y); //.sum();
  //auto end = std::chrono::steady_clock::now();
  //auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
  //std::cout << std::setw(9) << elapsed.count() << "ns" << std::endl;
  //start = std::chrono::steady_clock::now();
  z.backward();
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start);
  std::cout << std::setw(9) << elapsed.count() << "ns" << std::endl;
}

