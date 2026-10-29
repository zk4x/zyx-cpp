#ifndef OPTIM_HPP
#define OPTIM_HPP

#include "tensor.hpp"

#include <cmath>

namespace optim {
template <typename T> class Optim {
public:
  Optim(const std::vector<Tensor<T>> &inputs) {
    for (auto param : inputs)
      if (param.requires_grad())
        _params.push_back(param);
  }
  
  void zero_grad() {
    for (auto t : params())
      t.zero_grad();
  }

  virtual void step() = 0;

  std::vector<Tensor<T>> &params() {
    return _params;
  }
  
protected:
  std::vector<Tensor<T>> _params;
};

template <typename T> class SGD : public Optim<T> {
public:
  float lr = 0.01;

  SGD(const std::vector<Tensor<T>> &params, float lr=0.01) :
    Optim<T>(params), lr(lr) {}

  void step() {
    for (auto t : Optim<T>::params())
      t.mut_data() -= lr*t.grad().value();
  }
};

template <typename T> class Adam : public Optim<T> {
public:
  float lr;
  float b1;
  float b2;
  float eps;
  
  Adam(const std::vector<Tensor<T>> &params, float lr=0.01, float b1=0.9, float b2=0.999, float eps=1e-8) :
    Optim<T>(params), lr(lr), b1(b1), b2(b2), eps(eps), t(0) {
    for (auto t : params) {
      m.push_back(Tensor<T>::zeros(t.shape()));
      v.push_back(Tensor<T>::zeros(t.shape()));
    }
  }

  void step() {
    ++t;
    const auto a = lr*std::sqrt(1-std::pow(b2, t))/(1-std::pow(b1, t));
    for (size_t i = 0; i < Optim<T>::params().size(); ++i) {
      auto &param = Optim<T>::params()[i];
      const auto grad = param.grad().value();
      m[i] = b1*m[i]+(1-b1)*grad;
      v[i] = b2*v[i]+(1-b2)*grad*grad;
      param.mut_data() -= (a*m[i]/(v[i].sqrt() + eps)).data();
    }
  }
  
private:
  size_t t;
  
  std::vector<Tensor<T>> m;
  std::vector<Tensor<T>> v;
};
}

#endif // OPTIM_HPP
