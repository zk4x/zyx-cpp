#ifndef MODELS_HPP
#define MODELS_HPP

#include <stdexcept>
#include <vector>

#include "layers.hpp"
#include "tensor.hpp"

namespace nn {
  namespace model {
    template <typename T> class Sequential : public nn::layer::Layer<T> {
    public:
      Sequential(std::vector<nn::layer::Layer<T> *> layers) {
        for (size_t i = 0; i < layers.size(); ++i)
          push_back(layers[i]);
      }

      ~Sequential() {
        for (auto layer : layers)
          delete layer;
      }

      const Tensor<T> operator()(const Tensor<T> input) const {
        if (layers.empty())
          throw std::invalid_argument("nn::model::Sequential: model has no layers");
        auto res = (*layers[0])(input);
        for (size_t i = 1; i < layers.size(); ++i)
          res = (*layers[i])(res);
        return res;
      }

      std::vector<Tensor<T>> params() const {
        std::vector<Tensor<T>> params;
        for (auto layer : layers)
          for (auto param : layer->params())
            params.push_back(param);
        return params;
      }

      void push_back(nn::layer::Layer<T> *layer) {
        layers.push_back(layer);
      }

    private:
      std::vector<nn::layer::Layer<T> *> layers;
    };
  }
}

#endif // MODELS_HPP
