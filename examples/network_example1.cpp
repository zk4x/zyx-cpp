#define DEBUG

#include "nn.hpp"
#include "tensor.hpp"
#include "optim.hpp"

class SimpleDenseNet {
public:
  SimpleDenseNet() {
    w1 = Tensor<float>::uniform({1, 20}, -1, 1);
    w1.requires_grad(true);
    b1 = Tensor<float>::uniform({1, 20}, 0, 1);
    b1.requires_grad(true);
    a1 = Tensor<float>::with_shape({1, 20});
    w2 = Tensor<float>::uniform({20, 1}, -1, 1);
    w2.requires_grad(true);
    b2 = Tensor<float>::uniform({1, 1}, 0, 1);
    b2.requires_grad(true);
    a2 = Tensor<float>::with_shape({1, 1});
  }

  Tensor<float> operator()(Tensor<float> &a0) {
    a1 = nn::ReLU(a0.matmul(w1) + b1);
    a2 = nn::Tanh(a1.matmul(w2) + b2);
    return a2;
  }

  std::vector<Tensor<float>> params() {
    return {w1, b1, a1, w2, b2, a2};
  }

private:
  Tensor<float> w1;
  Tensor<float> b1;
  Tensor<float> a1;
  Tensor<float> w2;
  Tensor<float> b2;
  Tensor<float> a2;
};

int main() {
  // input is 1 value, output is 1 value
  auto net = SimpleDenseNet();

  auto input = Tensor<float>({2});
  input.reshape({1, 1});
  auto correct_output = Tensor<float>({3});
  correct_output.reshape({1, 1});

  auto output = net(input);

  // optimizer needs to know parameters of our model
  // and learning rate (default is 0.01)
  auto optimizer = optim::SGD(net.params(), 0.001);

  optimizer.zero_grad();
  auto loss = nn::loss::MSE(output, correct_output);
  loss.backward();
  optimizer.step();
  for (auto param : net.params())
    std::cout << param << std::endl;
  //std::cout << loss << std::endl;
}

