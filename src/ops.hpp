#ifndef OPS_HPP
#define OPS_HPP

#include "buffer.hpp"

#include <stdexcept>
#include <utility>

template <typename T>
class Tensor;
template <typename T>
class TensorData;

namespace ops {
// Add a gradient that may have been broadcast to a parameter's shape.
// Buffer arithmetic broadcasts a smaller operand across a larger one, so when
// backpropagating we have to reduce the larger gradient back to the parameter.
template <typename T>
void add_grad(Buffer<T> &dst, const Buffer<T> &src)
{
  if (dst.shape() == src.shape()) {
    dst += src;
    return;
  }
  if (src.size() <= dst.size() && dst.size() % src.size() == 0) {
    dst += src;
    return;
  }
  if (dst.size() < src.size() && src.size() % dst.size() == 0) {
    auto src_values = src.to_vector();
    auto dst_values = dst.to_vector();
    for (size_t k = 0; k < src_values.size(); k += dst_values.size())
      for (size_t i = 0; i < dst_values.size(); ++i)
        dst_values[i] += src_values[k + i];
    dst.write_data(std::move(dst_values));
    return;
  }
  throw std::invalid_argument("ops::add_grad(): incompatible gradient shapes");
}

namespace {
  template <typename T>
  void Transpose_backward(TensorData<T>* x)
  {
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += x->grad.value().transpose();
  }
}

template <typename T>
const Tensor<T> Transpose(const Tensor<T> x)
{
  return Tensor<T>::save_for_backwards(x.data().transpose(), Transpose_backward, { x });
}

namespace {
  template <typename T>
  static void Sum_backward(TensorData<T>* x)
  {
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += x->grad.value();
  }
}

template <typename T>
const Tensor<T> Sum(const Tensor<T> x)
{
  return Tensor<T>::save_for_backwards(x.data().sum(), Sum_backward, { x });
}

namespace {
  template <typename T>
  void Ln_backward(TensorData<T>* x)
  {
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += x->grad.value() / x->prev[0]->data;
  }
}

template <typename T>
const Tensor<T> Ln(const Tensor<T> x)
{
  return Tensor<T>::save_for_backwards(x.data().ln(), Ln_backward, { x });
}

namespace {
  template <typename T>
  void Exp_backward(TensorData<T>* x)
  {
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += x->grad.value() * x->data;
  }
}

template <typename T>
const Tensor<T> Exp(const Tensor<T> x)
{
  return Tensor<T>::save_for_backwards(x.data().exp(), Exp_backward, { x });
}

namespace {
  template <typename T>
  void Sqrt_backward(TensorData<T>* x)
  {
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += 0.5 * x->grad.value() / x->data;
  }
}

template <typename T>
const Tensor<T> Sqrt(const Tensor<T> x)
{
  return Tensor<T>::save_for_backwards(x.data().sqrt(), Sqrt_backward, { x });
}

namespace {
  template <typename T>
  void Inv_backward(TensorData<T>* x)
  {
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() -= x->grad.value() * x->data * x->data;
  }
}

template <typename T>
const Tensor<T> Inv(const Tensor<T> x)
{
  return Tensor<T>::save_for_backwards(x.data().inv(), Inv_backward, { x });
}

namespace {
  template <typename T>
  static void Mean_backward(TensorData<T>* x)
  {
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += x->grad.value() * (1 / static_cast<T>(x->prev[0]->data.size()));
  }
}

template <typename T>
const Tensor<T> Mean(const Tensor<T> x)
{
  return Tensor<T>::save_for_backwards(x.data().mean(), Mean_backward, { x });
}

namespace {
  template <typename T>
  void Dot_backward(TensorData<T>* x)
  {
    buffer::par();
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += x->prev[1]->data * x->grad.value();
    if (x->prev[1]->grad)
      x->prev[1]->grad.value() += x->prev[0]->data * x->grad.value();
    buffer::end_par();
  }
}

template <typename T>
const Tensor<T> Dot(const Tensor<T> x, const Tensor<T> y)
{
  return Tensor<T>::save_for_backwards(x.data().dot(y.data()), Dot_backward, { x, y });
}

namespace {
  template <typename T>
  void Add_backward(TensorData<T>* x)
  {
    buffer::par();
    if (x->prev[0]->grad)
      add_grad(x->prev[0]->grad.value(), x->grad.value());
    if (x->prev[1]->grad)
      add_grad(x->prev[1]->grad.value(), x->grad.value());
    buffer::end_par();
  }
}

template <typename T>
const Tensor<T> Add(const Tensor<T> x, const Tensor<T> y)
{
  return Tensor<T>::save_for_backwards(x.data() + y.data(), Add_backward, { x, y });
}

namespace {
  template <typename T>
  void Sub_backward(TensorData<T>* x)
  {
    buffer::par();
    if (x->prev[0]->grad)
      add_grad(x->prev[0]->grad.value(), x->grad.value());
    if (x->prev[1]->grad)
      add_grad(x->prev[1]->grad.value(), x->grad.value() * static_cast<T>(-1));
    buffer::end_par();
  }
}

template <typename T>
const Tensor<T> Sub(const Tensor<T> x, const Tensor<T> y)
{
  return Tensor<T>::save_for_backwards(x.data() - y.data(), Sub_backward, { x, y });
}

namespace {
  template <typename T>
  void Mul_backward(TensorData<T>* x)
  {
    buffer::par();
    if (x->prev[0]->grad)
      add_grad(x->prev[0]->grad.value(), x->grad.value() * x->prev[1]->data);
    if (x->prev[1]->grad)
      add_grad(x->prev[1]->grad.value(), x->grad.value() * x->prev[0]->data);
    buffer::end_par();
  }
}

template <typename T>
const Tensor<T> Mul(const Tensor<T> x, const Tensor<T> y)
{
  return Tensor<T>::save_for_backwards(x.data() * y.data(), Mul_backward, { x, y });
}

namespace {
  template <typename T>
  void Div_backward(TensorData<T>* x)
  {
    buffer::par();
    if (x->prev[0]->grad)
      add_grad(x->prev[0]->grad.value(), x->grad.value() / x->prev[1]->data);
    if (x->prev[1]->grad) {
      const auto denom = x->prev[1]->data * x->prev[1]->data;
      const auto numerator = x->grad.value() * x->prev[0]->data;
      add_grad(x->prev[1]->grad.value(), (numerator / denom) * static_cast<T>(-1));
    }
    buffer::end_par();
  }
}

template <typename T>
const Tensor<T> Div(const Tensor<T> x, const Tensor<T> y)
{
  return Tensor<T>::save_for_backwards(x.data() / y.data(), Div_backward, { x, y });
}

namespace {
  template <typename T>
  void MatMul_backward(TensorData<T>* x)
  {
    auto x0_t = x->prev[0]->data.transpose();
    buffer::par();
    auto x1_t = x->prev[1]->data.transpose();
    if (x->prev[0]->grad)
      x->prev[0]->grad.value() += x->grad.value().matmul(x1_t);
    if (x->prev[1]->grad)
      x->prev[1]->grad.value() += x0_t.matmul(x->grad.value());
    buffer::end_par();
  }
}

template <typename T>
const Tensor<T> MatMul(const Tensor<T> x, const Tensor<T> y)
{
  return Tensor<T>::save_for_backwards(x.data().matmul(y.data()), MatMul_backward, { x, y });
}
}

#endif // OPS_HPP
