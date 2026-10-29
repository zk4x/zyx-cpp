#ifndef BUFFER_ARRAYFIRE_HPP
#define BUFFER_ARRAYFIRE_HPP

namespace buffer {
  void par() {
  }

  void end_par() {
  }
}

namespace impl {
  template <typename T> void init() {
  }

  template <typename T> void write_data(Buffer<T> &self, std::vector<T> data) {
  }

  template <typename T> Buffer<T> copy(const Buffer<T> &self) {
  }

  template <typename T> void reshape(Buffer<T> &self, std::vector<size_t> shape) {
  }

  template <typename T> void fill(Buffer<T> &self, T value) {
  }

  template <typename T, typename F> void generate(Buffer<T> &self, F gen_fn) {
  }

  template <typename T> std::vector<T> to_vector(const Buffer<T> &self) {
  }

  template <typename T> Buffer<T> Transpose(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Exp(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Log(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Sqrt(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Inv(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Max(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Sum(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Add(const Buffer<T> &x, const Buffer<T> &y) {
  }

  template <typename T> Buffer<T> Sub(const Buffer<T> &x, const Buffer<T> &y) {
  }

  template <typename T> Buffer<T> Mul(const Buffer<T> &x, const Buffer<T> &y) {
  }

  template <typename T> Buffer<T> Div(const Buffer<T> &x, const Buffer<T> &y) {
  }

  template <typename T> Buffer<T> MatMul(const Buffer<T> &x, const Buffer<T> &y) {
  }

  template <typename T> Buffer<T> ReLU(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> ReLU_grad(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Tanh(const Buffer<T> &x) {
  }

  template <typename T> Buffer<T> Tanh_grad(const Buffer<T> &x) {
  }
}

#endif // BUFFER_ARRAYFIRE_HPP

