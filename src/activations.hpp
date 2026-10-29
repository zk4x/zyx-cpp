#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "tensor.hpp"

namespace nn {
  namespace {
    template <typename T>
    void ReLU_backward(TensorData<T> *x) {
      if (x->prev[0]->grad)
        x->prev[0]->grad.value() += x->grad.value()*x->data.relu_grad();
    }
  }

  template <typename T>
  const Tensor<T> ReLU(const Tensor<T> x) {
    return Tensor<T>::save_for_backwards(x.data().relu(), ReLU_backward, {x});
  }

  namespace {
    template <typename T>
    void Tanh_backward(TensorData<T> *x) {
      if (x->prev[0]->grad)
        x->prev[0]->grad.value() += x->grad.value()*x->data.tanh_grad();
    }
  }

  template <typename T>
  const Tensor<T> Tanh(const Tensor<T> x) {
    return Tensor<T>::save_for_backwards(x.data().tanh(), Tanh_backward, {x});
  }
}

#endif // ACTIVATIONS_HPP
