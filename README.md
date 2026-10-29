# zyx-cpp

A header-only tensor and automatic differentiation library written in C++.

Tensors can hold one-, two-, or three-dimensional values. Operations currently support 1D and 2D data, including matrix multiplication and the usual element-wise arithmetic.

## Features

- `Tensor` with reverse-mode automatic differentiation and `backward()`
- CPU storage by default; optional OpenCL backend
- `nn::layer::Linear` and `nn::layer::Dropout`
- `nn::ReLU` and `nn::Tanh`
- `nn::loss::MSE`
- `optim::SGD` and `optim::Adam`

## Requirements

- A C++23 compiler
- `make`
- OpenCL is optional

## Usage

Add the `src` directory to the include path and include the headers you need:

```sh
g++ -std=c++23 -Isrc main.cpp
```

```cpp
#include "nn.hpp"
#include "optim.hpp"

int main() {
  auto x = Tensor<float>({{1, 2}, {3, 4}}, true);
  auto y = Tensor<float>({{1, 0}}, true);

  auto linear = nn::layer::Linear<float>(2, 2);
  auto prediction = nn::Tanh(linear(x));
  auto loss = nn::loss::MSE(prediction, y);

  auto optimizer = optim::SGD(linear.params(), 0.01);
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
}
```

## Tests

```sh
cd tests
make -j
./test_buffer
./test_tensor
./test_activations
./test_losses
./test_layers
./test_models
```

Or run the whole suite from the repository root:

```sh
./run_tests.sh
```

## Examples

```sh
cd examples
make -j
./example1
./example2
./network_example1
./network_example2
```

## OpenCL

Build and test with the OpenCL backend:

```sh
cd tests
make clean
make -j OPENCL=1
```

This enables `ZYX_USE_OPENCL`. You need the OpenCL loader and the Khronos C++ header `CL/opencl.hpp`.

On Debian/Ubuntu:

```sh
sudo apt install opencl-clhpp-headers ocl-icd-opencl-dev
```

For a non-standard OpenCL installation, pass the paths explicitly:

```sh
make OPENCL=1 \
  OPENCL_INCLUDE="-I/path/to/include" \
  OPENCL_LIBS="-L/path/to/lib -lOpenCL"
```

The CUDA toolkit ships OpenCL's C header but not the Khronos C++ header `CL/opencl.hpp`. That header comes from `opencl-clhpp-headers` or another CLHPP package.

## Documentation

- `how_it_works.txt` describes the autograd design.
- `accelerators.txt` describes the OpenCL requirements.

## License

GPL-3.0. See [LICENSE](LICENSE).
