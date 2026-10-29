#define DEBUG

#include "nn.hpp"
#include "optim.hpp"
#include "tensor.hpp"

int main()
{
  auto bh = Tensor<float>::zeros({ 1, 20 }, true);
  auto whx = Tensor<float>::uniform({ 1, 20 }, 0, 1, true);
  auto wh = Tensor<float>::uniform({ 20, 20 }, 0, 1, true);
  // ah is tanh
  auto w = Tensor<float>::uniform({ 20, 1 }, 0, 1, true);
  // a is softmax
  auto optim_ = optim::SGD(std::vector<Tensor<float>> { bh, whx, wh, w }, 0.01);
  // loss is MSE

  auto x = Tensor<float>({ 0.3 });
  auto y = Tensor<float>({ 0.5 });

  auto a1 = x.matmul(whx);
  bh = nn::Tanh(a1 + bh.matmul(wh));
  auto y_x = nn::Tanh(bh.matmul(w));

  auto loss = nn::loss::MSE(y_x, y);
  optim_.zero_grad();
  loss.backward();
  optim_.step();

  std::cout << w << std::endl;
}
