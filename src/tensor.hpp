#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <memory>
#include <set>
#include <optional>

#include "ops.hpp"

template <typename T> class TensorData;

template <typename T> class Tensor {
public:
  Tensor() {
    self = std::make_shared<TensorData<T>>();
  };

  Tensor(Buffer<T> data, bool requires_grad=false) {
    self = std::make_shared<TensorData<T>>();
    self->data = data;
    this->requires_grad(requires_grad);
  }

  Tensor<T> copy() const {
    auto res = Tensor<T>(self->data.copy());
    if (requires_grad()) {
      res.self->prev = self->prev;
      res.self->back_fn = self->back_fn;
      res.self->grad = self->grad->copy();
    }
    return res;
  }
  
  static Tensor<T> with_shape(std::vector<size_t> shape, bool requires_grad=false) {
    return Tensor<T>(Buffer<T>::with_shape(shape), requires_grad);
  }

  static Tensor<T> with_shape(std::vector<size_t> shape, T init_value, bool requires_grad=false) {
    return Tensor<T>(Buffer<T>::with_shape(shape, init_value), requires_grad);
  }

  static Tensor<T> eye(size_t a, bool requires_grad=false) {
    return Tensor<T>(Buffer<T>::eye(a), requires_grad);
  }
  
  static Tensor<T> zeros(std::vector<size_t> shape, bool requires_grad=false) {
    return Tensor<T>(Buffer<T>::with_shape(shape, 0), requires_grad);
  }

  static Tensor<T> uniform(const std::vector<size_t> &shape, T minval=0, T maxval=1, bool requires_grad=false) {
    return Tensor<T>(Buffer<T>::uniform(shape, minval, maxval), requires_grad);
  }

  void backward() const {
    if (!requires_grad())
      throw std::invalid_argument("\n\tCalled backward on node that does not require grad\n"
        "where():\n\tMethod void Tensor<T>::backward() const");
    std::vector<TensorData<T> *> graph;
    std::set<TensorData<T> *> visited;
    self->grad.value().fill(1);
    self->walk_back(graph, visited);
    for (auto node_iter = graph.rbegin(); node_iter != graph.rend(); ++node_iter)
      (*node_iter)->back_fn(*node_iter);
  }

  friend std::ostream &operator<<(std::ostream &stream, const Tensor<T> x) {
    stream << "Tensor at: " << &x.self->data << "  with shape: [";
    for (auto v : x.shape()) stream << v << ", ";
    stream << "\b\b]\n" << x.self->data;
    if (x.requires_grad()) stream << "  with grad:\n" << x.self->grad.value();
    return stream;
  }

  bool requires_grad() const { return self->grad.has_value(); };

  void requires_grad(bool requires_grad) const {
    if (requires_grad)// && shape() != self->grad->shape())
      self->grad = Buffer<T>::with_shape(shape(), 0);
    else
      self->grad = std::nullopt;
  };

  const std::optional<Buffer<T>> grad() const { return self->grad; }
  void zero_grad() const { if (requires_grad()) self->grad.value().fill(0); }

  const Buffer<T> &data() const { return self->data; }
  Buffer<T> &mut_data() const { return self->data; }

  size_t size() const { return self->data.size(); }
  const std::vector<size_t> &shape() const { return self->data.shape(); }

  void reshape(const std::vector<size_t> &shape) const {
    self->data.reshape(shape);
    if (requires_grad()) self->grad->reshape(shape);
  }

  static const Tensor<T> save_for_backwards(Buffer<T> data,
    void (*back_fn)(TensorData<T> *),
    std::vector<Tensor<T>> prev)
  {
    auto res = Tensor<T>(data);
    res.self->prev.reserve(prev.size());
    bool needs_grad = false;
    for (auto v : prev) {
      res.self->prev.push_back(v.self);
      needs_grad = needs_grad || v.requires_grad();
    }
    if (needs_grad) {
      res.requires_grad(true);
      res.self->back_fn = back_fn;
    }
    return res;
  }

  bool operator==(const Tensor<T> &other) const { return self == other.self; }
  bool operator<(const Tensor<T> &other) const { return self < other.self; }

  // unary operators
  const Tensor<T> transpose() const { return ops::Transpose<T>(*this); }
  const Tensor<T> sum() const { return ops::Sum<T>(*this); }
  const Tensor<T> mean() const { return ops::Mean<T>(*this); }
  const Tensor<T> ln() const { return ops::Ln<T>(*this); }
  const Tensor<T> exp() const { return ops::Exp<T>(*this); }
  const Tensor<T> sqrt() const { return ops::Sqrt<T>(*this); }
  const Tensor<T> inv() const { return ops::Inv<T>(*this); }

  // operators with constant
  const Tensor<T> operator+(T other) const { return ops::Add<T>(*this, Tensor({other})); }
  friend const Tensor<T> operator+(T other, const Tensor<T> x) { return x + other; }
  const Tensor<T> operator-(T other) const { return ops::Sub<T>(*this, Tensor({other})); }
  friend const Tensor<T> operator-(T other, const Tensor<T> x) { return ops::Sub<T>(Tensor({other}), x); }
  const Tensor<T> operator*(T other) const { return ops::Mul<T>(*this, Tensor({other})); }
  friend const Tensor<T> operator*(T other, const Tensor<T> x) { return x * other; }
  const Tensor<T> operator/(T other) const { return ops::Div<T>(*this, Tensor({other})); }
  friend const Tensor<T> operator/(T other, const Tensor<T> x) { return ops::Div<T>(Tensor({other}), x); }

  // binary operators
  const Tensor<T> operator+(const Tensor<T> other) const { return ops::Add<T>(*this, other); }
  const Tensor<T> operator-(const Tensor<T> other) const { return ops::Sub<T>(*this, other); }
  const Tensor<T> operator*(const Tensor<T> other) const { return ops::Mul<T>(*this, other); }
  const Tensor<T> operator/(const Tensor<T> other) const { return ops::Div<T>(*this, other); }
  const Tensor<T> dot(const Tensor<T> other) const { return ops::Dot<T>(*this, other); }
  const Tensor<T> matmul(const Tensor<T> other) const { return ops::MatMul<T>(*this, other); }

private:
  std::shared_ptr<TensorData<T>> self;
};

// this is only internal class
template <typename T> class TensorData : public std::enable_shared_from_this<TensorData<T>> {
public:
  void (*back_fn)(TensorData<T> *) = nullptr; // function that got us here
  std::vector<std::shared_ptr<TensorData<T>>> prev; // nodes that got us here
  Buffer<T> data; // main tensor storage
  std::optional<Buffer<T>> grad; // nullopt if requires_grad() == false

  void walk_back(std::vector<TensorData<T> *> &graph, std::set<TensorData<T> *> &visited) {
    if (back_fn == nullptr || visited.contains(this)) return;
    visited.insert(this);
    for (auto node : prev) node->walk_back(graph, visited);
    graph.push_back(this);
  }
};

#endif // TENSOR_HPP
