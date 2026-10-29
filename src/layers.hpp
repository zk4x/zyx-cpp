#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor.hpp"

#include <cstdlib>
#include <vector>

namespace nn {
namespace layer {
  template <typename T>
  class Layer {
public:
    virtual ~Layer() {};
    virtual const Tensor<T> operator()(const Tensor<T> input) const = 0;
    virtual std::vector<Tensor<T>> params() const = 0;
  };

  template <typename T>
  class Linear : public Layer<T> {
public:
    Linear(size_t in, size_t out)
    {
      // in is the last dimension of input tensor
      // out is the dimension of output (1D Tensor)
      weight = Tensor<T>::uniform({ in, out }, 0, 1, true);
      bias = Tensor<T>::uniform({ out }, 0, 1, true);
    }

    ~Linear() {};

    const Tensor<T> operator()(const Tensor<T> input) const
    {
#if defined(DEBUG)
      if (input.shape().back() != weight.shape().front())
        throw std::invalid_argument("\n\tThe input Tensor has wrong shape\n"
                                    "  where():\n\tMethod const Tensor<T> operator()(const Tensor<T> input) const\n");
#endif
      // std::cout << std::endl << input << std::endl << weight << std::endl;
      return input.matmul(weight) + bias;
    }

    std::vector<Tensor<T>> params() const
    {
      return { weight, bias };
    }

private:
    Tensor<T> weight;
    Tensor<T> bias;
  };

  template <typename T>
  class Dropout : public Layer<T> {
public:
    Dropout(float p)
    {
      this->p = p;
    }

    ~Dropout() {};

    const Tensor<T> operator()(const Tensor<T> input) const
    {
      const auto n = input.size();
      auto vec = std::vector<T>(n, T(1));
      for (size_t i = 0; i < n; ++i)
        if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) > p)
          vec[i] = 0;
      auto drop = Buffer<T>::with_shape(input.shape());
      drop.write_data(vec);
      return input * drop;
    }

    std::vector<Tensor<T>> params() const
    {
      return {};
    }

    float p;
  };
}
}

#endif // LAYERS_HPP
