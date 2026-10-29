#ifndef LOSSES_HPP
#define LOSSES_HPP

#include <cmath>
#include "tensor.hpp"

namespace nn {
  namespace loss {
    enum class Reduction: short {NONE, MEAN, SUM};

    template <typename T>
    const Tensor<T> reduce(const Tensor<T> x, Reduction reduction) {
      switch (reduction) {
        case nn::loss::Reduction::MEAN:
          return x.mean();
        case nn::loss::Reduction::SUM:
          return x.sum();
        default:
          return x;
      }
    }

    template <typename T>
    const Tensor<T> MSE(const Tensor<T> x, const Tensor<T> y,
        Reduction reduction=nn::loss::Reduction::MEAN) {
      auto diff = x-y;
      return reduce(diff*diff, reduction);
    }
  }
}

#endif // LOSSES_HPP
