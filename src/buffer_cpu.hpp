#ifndef BUFFER_CPU_HPP
#define BUFFER_CPU_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

template <typename T>
class Buffer;

namespace buffer {
inline void par() { }
inline void end_par() { }
}

namespace impl {
template <typename T>
void init()
{
  // why should I use it? ;-)
}

template <typename T>
void write_data(Buffer<T>& self, std::vector<T> data)
{
  self.data = data;
}

template <typename T>
Buffer<T> copy(const Buffer<T>& self)
{
  auto res = Buffer<T>();
  res._shape = self.shape();
  res.data = self.data;
  return res;
}

template <typename T>
void reshape(Buffer<T>& self, std::vector<size_t> shape)
{
  self._set_shape_directly(shape);
  self.data.resize(self.size());
}

template <typename T>
void fill(Buffer<T>& self, T value)
{
  std::fill(self.data.begin(), self.data.end(), value);
}

template <typename T, typename F>
void generate(Buffer<T>& self, F gen_fn)
{
  std::generate(self.data.begin(), self.data.end(), gen_fn);
}

template <typename T>
std::vector<T> to_vector(const Buffer<T>& self)
{
  return self.data;
}

template <typename T>
Buffer<T> Transpose(const Buffer<T>& x)
{
  auto rows = x.shape().front(), cols = x.shape().back();
  if (x.shape().size() == 1)
    cols = 1;
  auto res = Buffer<T>::with_shape({ cols, rows });
  for (size_t i = 0; i < cols; ++i)
    for (size_t k = 0; k < rows; ++k)
      res.data[i * rows + k] = x.data[k * cols + i];
  return res;
}

template <typename T>
Buffer<T> Exp(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  std::transform(x.data.begin(), x.data.end(), res.data.begin(), static_cast<T (*)(T)>(std::exp));
  return res;
}

template <typename T>
Buffer<T> Log(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  std::transform(x.data.begin(), x.data.end(), res.data.begin(), static_cast<T (*)(T)>(std::log));
  return res;
}

template <typename T>
Buffer<T> Sqrt(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  std::transform(x.data.begin(), x.data.end(), res.data.begin(), static_cast<T (*)(T)>(std::sqrt));
  return res;
}

template <typename T>
Buffer<T> Inv(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t i = 0; i < res.data.size(); ++i)
    res.data[i] = 1 / x.data[i];
  return res;
}

template <typename T>
Buffer<T> Max(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape({ 1 });
  res.data[0] = *std::max_element(x.data.begin(), x.data.end());
  return res;
}

template <typename T>
Buffer<T> Sum(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape({ 1 });
  res.data[0] = std::accumulate(x.data.begin(), x.data.end(), 0, std::plus<T>());
  return res;
}

template <typename T>
Buffer<T> Add(const Buffer<T>& x, const Buffer<T>& y)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t k = 0; k < x.data.size(); k += y.data.size())
    std::transform(&x.data[k], &x.data[k + y.data.size()], &y.data[0], &res.data[k], std::plus<T>());
  return res;
}

template <typename T>
Buffer<T> Sub(const Buffer<T>& x, const Buffer<T>& y)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t k = 0; k < x.data.size(); k += y.data.size())
    std::transform(&x.data[k], &x.data[k + y.data.size()], &y.data[0], &res.data[k], std::minus<T>());
  return res;
}

template <typename T>
Buffer<T> Mul(const Buffer<T>& x, const Buffer<T>& y)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t k = 0; k < x.data.size(); k += y.data.size())
    std::transform(&x.data[k], &x.data[k + y.data.size()], &y.data[0], &res.data[k], std::multiplies<T>());
  return res;
}

template <typename T>
Buffer<T> Div(const Buffer<T>& x, const Buffer<T>& y)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t k = 0; k < x.data.size(); k += y.data.size())
    std::transform(&x.data[k], &x.data[k + y.data.size()], &y.data[0], &res.data[k], std::divides<T>());
  return res;
}

template <typename T>
Buffer<T> MatMul(const Buffer<T>& x, const Buffer<T>& y)
{
  const auto x_rows = x.shape().size() > 1 ? *(x.shape().end() - 2) : 1;
  const auto x_cols = x.shape().back(), o_cols = y.shape().back();

  auto res = Buffer<T>::with_shape({ x_rows, o_cols }, 0);

  const size_t x_size = x.data.size();

  // what is SIMD width?
  const size_t SIMD_WIDTH_BYTES = 8;
  const size_t SIMD_WIDTH = SIMD_WIDTH_BYTES / sizeof(T);

  // rewrite matrix y, store in Y
  auto Y = std::vector<T>();
  Y.reserve(y.size());

  const size_t remaining_rows = x_cols % SIMD_WIDTH;
  const size_t y_cols = x_cols >= SIMD_WIDTH ? o_cols * SIMD_WIDTH : o_cols * remaining_rows;
  const size_t y_rows_limit = x_cols / SIMD_WIDTH * y_cols;

  size_t _yr = 0;
  for (; _yr < x_cols - remaining_rows; _yr += SIMD_WIDTH)
    for (size_t oc = 0; oc < o_cols; ++oc)
      for (size_t i = 0; i < SIMD_WIDTH; ++i)
        Y.push_back(y.data[(_yr + i) * o_cols + oc]);

  // add remaining values in matrix y to matrix Y
  for (size_t oc = 0; oc < o_cols; ++oc)
    for (size_t i = 0; i < remaining_rows; ++i)
      Y.push_back(y.data[(_yr + i) * o_cols + oc]);

  // SATURATE SIMD UNITS !!! ;-)

  T x_block[SIMD_WIDTH];
  T* x_block2 = new T[remaining_rows];
  T* res_row = new T[o_cols];
  const size_t row_len = sizeof(T) * o_cols;
  // for each row in x
  //for (size_t xr = 0, resr = 0; xr < x_size; xr += x_cols, resr += o_cols) {
  size_t xr = 0, resr = 0;
  while (xr < x_size) {
    // create single row in result
    std::memset(&res_row[0], 0, row_len);
    // for each row in matrix Y
    size_t yr = 0, xc = xr;
    for (; yr < y_rows_limit; yr += y_cols, xc += SIMD_WIDTH) {
      // copy SIMD_WIDTH block from x
      std::copy(&x.data[xc], &x.data[xc + SIMD_WIDTH], &x_block[0]);
      // for each SIMD_WIDTH block in Y
      for (size_t yc = yr, rc = 0; rc < o_cols; ++rc, yc += SIMD_WIDTH) {
        // do the calculation inside SIMD block
        T var = 0;
        for (size_t xi = 0, yi = yc; xi < SIMD_WIDTH; ++xi, ++yi)
          var += x_block[xi] * Y[yi];
        res_row[rc] += var;
      }
    }
    // resolve remaining_columns in x
    if (remaining_rows > 0) {
      std::copy(&x.data[xc], &x.data[xc + remaining_rows], &x_block2[0]);
      for (size_t yc = yr, rc = 0; rc < o_cols; ++rc, yc += remaining_rows) {
        T var = 0;
        for (size_t xi = 0, yi = yc; xi < remaining_rows; ++xi, ++yi)
          var += x_block2[xi] * Y[yi];
        res_row[rc] += var;
      }
    }
    // write res_row into matrix res
    std::copy(&res_row[0], &res_row[o_cols], &res.data[resr]);
    xr += x_cols;
    resr += o_cols;
  }
  delete[] x_block2;
  delete[] res_row;

  return res;
}

template <typename T>
Buffer<T> ReLU(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t i = 0; i < res.data.size(); ++i)
    res.data[i] = std::max<T>(0, x.data[i]);
  return res;
}

template <typename T>
Buffer<T> ReLU_grad(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t i = 0; i < res.data.size(); ++i)
    res.data[i] = x.data[i] > 0;
  return res;
}

template <typename T>
Buffer<T> Tanh(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t i = 0; i < res.data.size(); ++i)
    res.data[i] = std::tanh(x.data[i]);
  return res;
}

template <typename T>
Buffer<T> Tanh_grad(const Buffer<T>& x)
{
  auto res = Buffer<T>::with_shape(x.shape());
  for (size_t i = 0; i < res.data.size(); ++i)
    res.data[i] = 1 - x.data[i] * x.data[i];
  return res;
}
}

#endif // BUFFER_CPU_HPP
