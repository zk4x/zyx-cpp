# zyx-cpp

The original C++ implementation of [zyx](https://github.com/zk4x/zyx), written about five years ago and preserved here as a portfolio artifact.

It is a small, header-only tensor and automatic-differentiation library. The Rust version is the successor and is the one that is actively developed; this repository is not.

## Status

This is the code as it was originally written, with only the build fixes and bug fixes required to make it compile and pass its tests again. There is no new feature work and no modernization.

The following unfinished or non-functional pieces were removed rather than published as if they worked:

- the empty `conv2d` implementation,
- the no-op `Pooling` layer,
- the unfinished `L1` and `CrossEntropy` losses.

## What is here

- `Tensor`, a shared-pointer-backed N-dimensional value with reverse-mode autograd (`backward()`).
- A `Buffer` abstraction with a CPU backend by default and an OpenCL backend behind `ZYX_USE_OPENCL`.
- The original experimental ArrayFire/clBLAS backend files, kept as part of the artifact.
- `nn::layer::Linear` and `nn::layer::Dropout`.
- `nn::ReLU` and `nn::Tanh`.
- `nn::loss::MSE`.
- `optim::SGD` and `optim::Adam`.
- Tests and small examples.

The library supports 1D and 2D tensors. `matmul` is 2D, and broadcasting is the flat "repeat the smaller buffer" form used by the original code (scalars and bias vectors).

## Requirements

- A C++23 compiler (the Makefiles use `-std=c++23`).
- `make`.
- OpenCL is optional. To build with it you need the OpenCL loader and the Khronos C++ header `CL/opencl.hpp`.

On Debian/Ubuntu, the OpenCL C++ header is in `opencl-clhpp-headers`:

```sh
sudo apt install opencl-clhpp-headers ocl-icd-opencl-dev
```

## Build and test

CPU:

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

Or from the repository root:

```sh
./run_tests.sh
```

OpenCL:

```sh
cd tests
make clean
make -j OPENCL=1
```

If OpenCL lives in a non-standard prefix (for example the CUDA toolkit), pass the paths explicitly:

```sh
make OPENCL=1 \
  OPENCL_INCLUDE="-I/path/to/include" \
  OPENCL_LIBS="-L/path/to/lib -lOpenCL"
```

Note that the CUDA toolkit ships OpenCL's C header but not the Khronos C++ header `CL/opencl.hpp`. That header has to come from `opencl-clhpp-headers` or another Khronos CLHPP package.

Examples use the same flags:

```sh
cd examples
make -j
./example1
./network_example1
```

## A small example

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

## Notes

- `how_it_works.txt` is the original short explanation of the autograd design.
- `accelerators.txt` and `readme.txt` are the original project notes, left as they were.
- OpenCL support was experimental. It is tested on the CUDA OpenCL implementation.
- This is an educational library, not a production numerical framework.

## License

GPL-3.0. See [LICENSE](LICENSE).
