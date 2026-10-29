#ifndef BUFFER_HPP
#define BUFFER_HPP

#if defined(DEBUG)
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#endif

#include <functional>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace buffer {
  void par();
  void end_par();
}

// here come includes needed by implementations for buffer storage medium
#if defined(ZYX_USE_OPENCL) || defined(OPENCL)
#include "buffer_opencl.hpp"
#elif defined(CLBLAS)
#include "buffer_clblas.hpp"
#elif defined(ARRAYFIRE)
#include "buffer_arrayfire.hpp"
#else
#include "buffer_cpu.hpp"
#endif

template <typename T> class Buffer {
public:
  // each implementation can use it's own storage medium
#if defined(ZYX_USE_OPENCL) || defined(OPENCL)
  cl::Buffer data;
#elif defined(CLBLAS)
  cl::Buffer data;
#elif defined(ARRAYFIRE)
#else
  std::vector<T> data;
#endif

  static Buffer<T> with_shape(const std::vector<size_t> &shape) {
    auto res = Buffer<T>();
    res.reshape(shape);
    return res;
  }

  static Buffer<T> with_shape(const std::vector<size_t> &shape, T init) {
    auto res = Buffer<T>();
    res.reshape(shape);
    res.fill(init);
    return res;
  }

  static Buffer<T> uniform(const std::vector<size_t> &shape, T minval=0, T maxval=1) {
    auto res = Buffer<T>::with_shape(shape);
    std::random_device rnd_device;
    std::default_random_engine engine{rnd_device()};
    std::uniform_real_distribution<T> dist(minval, maxval);
    res.generate([&dist, &engine]() { return dist(engine); });
    return res;
  }

  static Buffer<T> eye(size_t a) {
    auto res = Buffer<T>::with_shape({a, a});
    std::vector<T> vec(res.size(), 0);
    for (size_t i = 0; i < a; ++i)
      vec[i*a+i] = 1;
    res.write_data(vec);
    return res;
  }

  Buffer() { impl::init<T>(); }

  Buffer(std::initializer_list<T> data) {
    impl::init<T>();
    _shape = {data.size()};
    write_data(data);
  }

  Buffer(std::initializer_list<std::initializer_list<T>> data) {
    impl::init<T>();
    _shape = {data.size(), data.begin()->size()};
    auto vec = std::vector<T>(size(), 0);
    size_t i = 0;
    for (auto &row : data)
      for (auto v : row)
        vec[i] = v, ++i;
    write_data(vec);
  }

  Buffer(std::initializer_list<std::initializer_list<std::initializer_list<T>>> data) {
    impl::init<T>();
    _shape = {data.size(), data.begin()->size(), data.begin()->begin()->size()};
    auto vec = std::vector<T>(size(), 0);
    size_t i = 0;
    for (auto &row1 : data)
      for (auto &row2 : row1)
        for (auto v : row2)
          vec[i] = v, ++i;
    write_data(vec);
  }

  // copy constructor and copy assignment are move
  Buffer(const Buffer<T> &other) {
    if (this != &other) {
      _shape = std::move(other._shape);
      data = std::move(other.data);
    }
  }

  Buffer<T> &operator=(const Buffer<T> &other) {
    if (this != &other) {
      _shape = std::move(other._shape);
      data = std::move(other.data);
    }
    return *this;
  }

  const std::vector<size_t> &shape() const { return _shape; }

  const size_t dim(const size_t i) const {
    if (i >= 0)
      return shape()[i];
    return *(shape().end()+i);
  }

  friend std::ostream &operator<<(std::ostream &stream, const Buffer<T> &x) {
    const size_t REAL_PRECISION = 2;
    auto n = x.size(), s = x._shape.size();
    if (s == 0) return stream;
    auto print_char = [&stream](char ch, size_t times) {
      for (size_t i = 0; i < times; ++i) stream << ch;
    };
    auto vec = x.to_vector();
    size_t w = 0;
    std::stringstream convertor;
    for (auto v : vec) {
      convertor << std::fixed << std::setprecision(REAL_PRECISION) << v;
      auto len = convertor.str().length();
      std::stringstream().swap(convertor);
      if (len > w) w = len;
    }
    for (size_t i = 0; i < n; ++i) {
      size_t v = n;
      for (size_t k = 0; k < s; ++k) {
        if (i % v == 0) { print_char(' ', k); print_char('[', s - k); break; };
        v /= x._shape[k];
      }
      stream << std::fixed << std::setprecision(REAL_PRECISION) << std::right << std::setw(w) << vec[i] << ", ";
      v = n;
      for (size_t k = 0; k < s; ++k) {
        if ((i + 1) % v == 0) { stream << "\b\b"; print_char(']', s - k); break; }
        v /= x._shape[k];
      }
      if ((i + 1) % x._shape.back() == 0 && i != n - 1) stream << ",\n";
    }
    return stream;
  }

  bool empty() const { return shape().empty(); }

  size_t size() const {
    if (shape().empty())
      return 0;
    return std::accumulate(shape().begin(), shape().end(), size_t{1}, std::multiplies<size_t>());
  }

  Buffer<T> transpose() const {
#if defined(DEBUG)
    if (shape().size() > 2)
      throw std::invalid_argument("\n\tCan only transpose buffers with 1 or 2 dimensions\n"
        "  where():\n\tMethod Buffer<T> Buffer<T>::transpose() const\n");
#endif
    return impl::Transpose<T>(*this);
  }

  Buffer<T> exp() const { return impl::Exp<T>(*this); }
  Buffer<T> ln() const { return impl::Log<T>(*this); }
  Buffer<T> sqrt() const { return impl::Sqrt<T>(*this); }
  Buffer<T> inv() const { return impl::Inv<T>(*this); }
  Buffer<T> relu() const { return impl::ReLU<T>(*this); }
  Buffer<T> relu_grad() const { return impl::ReLU_grad<T>(*this); }
  Buffer<T> tanh() const { return impl::Tanh<T>(*this); }
  Buffer<T> tanh_grad() const { return impl::Tanh_grad<T>(*this); }
  Buffer<T> max() const { return impl::Max<T>(*this); }
  Buffer<T> sum() const { return impl::Sum<T>(*this); }
  Buffer<T> mean() const { return sum()/size(); }
  Buffer<T> operator+(T other) const { return impl::Add(*this, Buffer({other})); }
  friend Buffer<T> operator+(T x, const Buffer<T> &y) { return y + x; }

  Buffer<T> operator+(const Buffer<T> &other) const {
#if defined(DEBUG)
    if (size() % other.size() != 0)
      throw std::invalid_argument("\n\tCan only add buffers with compatible shapes\n"
        "  where():\n\tMethod Buffer<T> Buffer<T>::operator+(const Buffer<T> &other) const\n");
#endif
    return impl::Add(*this, other);
  }

  void operator+=(const Buffer<T> &x) { *this = *this + x; }
  void operator+=(T x) { *this = *this + x; }
  Buffer<T> operator-(T x) const { return *this + (-x); }
  friend Buffer<T> operator-(T x, const Buffer<T> &y) { return x + y*(-1); }

  Buffer<T> operator-(const Buffer<T> &other) const {
#if defined(DEBUG)
    if (size() % other.size() != 0)
      throw std::invalid_argument("\n\tCan only subtract buffers with compatible shapes\n"
        "  where():\n\tMethod Buffer<T> Buffer<T>::operator-(const Buffer<T> &other) const\n");
#endif
    return impl::Sub(*this, other);
  }

  void operator-=(const Buffer<T> &other) { *this = *this - other; }
  void operator-=(T other) { *this = *this - other; }
  Buffer<T> operator*(T other) const { return impl::Mul(*this, Buffer({other})); }
  friend Buffer<T> operator*(T x, const Buffer<T> &y) { return y * x; }

  Buffer<T> operator*(const Buffer<T> &other) const {
#if defined(DEBUG)
    if (size() % other.size() != 0)
      throw std::invalid_argument("\n\tCan only multiply buffers with compatible shapes\n"
        "  where():\n\tMethod Buffer<T> Buffer<T>::operator*(const Buffer<T> &other) const\n");
#endif
    return impl::Mul(*this, other);
  }

  void operator*=(const Buffer<T> &other) { *this = *this * other; }
  void operator*=(T other) { *this = *this * other; }
  Buffer<T> operator/(T x) const { return impl::Div(*this, Buffer({x})); }
  friend Buffer<T> operator/(T x, const Buffer<T> &y) { return Buffer({x}) / y; }

  Buffer<T> operator/(const Buffer<T> &other) const {
#if defined(DEBUG)
    if (size() % other.size() != 0)
      throw std::invalid_argument("\n\tCan only multiply buffers with compatible shapes\n"
        "  where():\n\tMethod Buffer<T> Buffer<T>::operator/(const Buffer<T> &other) const\n");
#endif
    return impl::Div(*this, other);
  }

  void operator/=(const Buffer<T> &other) { *this = *this / other; }
  void operator/=(T other) { *this = *this / other; }

  Buffer<T> dot(const Buffer<T> &other) const {
#if defined(DEBUG)
    if (_shape != other._shape)
      throw std::invalid_argument("\n\tCan only calculate dot product for buffers with same shapes\n"
        "  where():\n\tMethod Buffer<T> Buffer<T>::dot(const Buffer<T> &other) const\n");
#endif
    return (*this*other).sum();
  }

  Buffer<T> matmul(const Buffer<T> &other) const {
#if defined(DEBUG)
    if (_shape.size() > 2 || other._shape.size() > 2)
      throw std::invalid_argument("\n\tCan only multiply buffers with less than 3 dimensions\n"
        "  where():\n\tMethod Buffer<T> matmul(const Buffer<T> &other) const\n");
    if (_shape.back() != other._shape.front()) {
      //std::cout << "Trying to multiply these buffers: " << *this << std::endl << other << std::endl;
      throw std::invalid_argument("\n\tCan only multiply buffers with compatible dimensions\n"
        "  where():\n\tMethod Buffer<T> Buffer<T>::matmul(const Buffer<T> &other) const\n");
    }
#endif
    return impl::MatMul(*this, other);
  }

  Buffer<T> copy() const { return impl::copy(*this); }
  void reshape(std::vector<size_t> shape) { impl::reshape(*this, shape); }
  void fill(T value) { impl::fill(*this, value); }
  template <typename F> void generate(F gen_fn) { impl::generate(*this, gen_fn); }
  std::vector<T> to_vector() const { return impl::to_vector(*this); }

  void write_data(std::vector<T> data) { impl::write_data(*this, data); }

  void _set_shape_directly(std::vector<size_t> shape) { _shape = shape; }

private:
  std::vector<size_t> _shape; // implementation agnostic shape storage
};

#endif // BUFFER_HPP

