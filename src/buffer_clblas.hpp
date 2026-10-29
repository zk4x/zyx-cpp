#ifndef BUFFER_CLBLAS_HPP
#define BUFFER_CLBLAS_HPP

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#endif

#include <CL/opencl.hpp>
#include <set>

template <typename T> class Buffer;

namespace opencl_settings {
  bool par_queues = false;
  size_t default_platform = 0;
  size_t WGS = 0; // preferred work group size
}

namespace {
  // default implementation
  template <typename T> struct TypeName { static const char* Get() { return "You are using unsupported data type. Please use one out of supported ones"; } };
  template <> struct TypeName<bool> { static const char* Get() { return "bool"; } };
  template <> struct TypeName<char> { static const char* Get() { return "char"; } };
  template <> struct TypeName<unsigned char> { static const char* Get() { return "unsigned char"; } };
  template <> struct TypeName<short> { static const char* Get() { return "short"; } };
  template <> struct TypeName<unsigned short> { static const char* Get() { return "unsigned short"; } };
  template <> struct TypeName<int> { static const char* Get() { return "int"; } };
  template <> struct TypeName<unsigned int> { static const char* Get() { return "unsigned int"; } };
  template <> struct TypeName<long> { static const char* Get() { return "long"; } };
  template <> struct TypeName<unsigned long> { static const char* Get() { return "unsigned long"; } };
  //template <> struct TypeName<float16_t> { static const char* Get() { return "half"; } };
  template <> struct TypeName<float> { static const char* Get() { return "float"; } };
  template <> struct TypeName<double> { static const char* Get() { return "double"; } };

  std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
  }

  cl::Context ctx;
  cl::Device device;
  bool INIT_GPU = false;

  std::set<std::string> used_types;
  std::vector<cl::Program> programs;
  std::vector<std::string> types;
  std::vector<cl::CommandQueue> queues;

  cl::CommandQueue &queue() {
    if (!opencl_settings::par_queues) {
      // wait for all queues to finish before starting new one
      for (auto queue : queues) queue.finish();
      queues.clear();
    }
    queues.push_back(cl::CommandQueue(ctx, device));
    return queues.back();
  }

  template <typename T> cl::Program &program() {
    auto i = std::distance(types.begin(), std::find(types.begin(), types.end(), TypeName<T>::Get()));
    return programs[i];
  }

  template <typename T> Buffer<T> kernel(const Buffer<T> &x, const std::string &name) {
    auto res = Buffer<T>::with_shape(x.shape());
    const auto local = x.size() % opencl_settings::WGS == 0 ? opencl_settings::WGS : 1;
    cl::KernelFunctor<const cl::Buffer &, cl::Buffer &> functor(program<T>(), name);
    cl::EnqueueArgs args(queue(), cl::NDRange(x.size()), cl::NDRange(local));
    functor(args, x.data, res.data);
    return res;
  }

  template <typename T> Buffer<T> kernel(const Buffer<T> &x, const Buffer<T> &y, const std::string &name) {
    auto res = Buffer<T>::with_shape(x.shape());
    const size_t n2 = y.size(), n1 = x.size()/n2, gs = static_cast<size_t>(std::sqrt(opencl_settings::WGS));
    const auto local_x = n1 % gs == 0 ? gs : 1, local_y = n2 % gs == 0 ? gs : 1;
    cl::KernelFunctor<const cl::Buffer &, const cl::Buffer &, cl::Buffer &> functor(program<T>(), name);
    cl::EnqueueArgs args(queue(), cl::NDRange(n1, n2), cl::NDRange(local_x, local_y));
    functor(args, x.data, y.data, res.data);
    return res;
  }

  template <typename T> Buffer<T> reduce_kernel(const Buffer<T> &x, const std::string &name) {
    auto res = Buffer<T>::with_shape({1});
    cl::KernelFunctor<const unsigned long, const cl::Buffer &, cl::Buffer &> functor(program<T>(), name);
    cl::EnqueueArgs args(queue(), cl::NDRange(1), cl::NDRange(1));
    functor(args, x.size(), x.data, res.data);
    return res;
  }

  template <typename T> Buffer<T> reduce_kernel(const Buffer<T> &x, const Buffer<T> &y, const std::string &name) {
    auto res = Buffer<T>::with_shape(x.shape());
    cl::KernelFunctor<const unsigned long, const cl::Buffer &, const cl::Buffer &, cl::Buffer &> functor(program<T>(), name);
    cl::EnqueueArgs args(queue(), cl::NDRange(1), cl::NDRange(1));
    functor(args, x.size(), x.data, y.data, res.data);
    return res;
  }
}

namespace buffer {
  void par() {
    opencl_settings::par_queues = true;
  }

  void end_par() {
    opencl_settings::par_queues = false;
  }
}

namespace impl {
  template <typename T> void init() {
    std::string type = TypeName<T>::Get();
    if (used_types.contains(type))
      return;
    used_types.insert(type);
    if (!INIT_GPU) {
      //get all platforms (drivers)
      std::vector<cl::Platform> all_platforms;
      cl::Platform::get(&all_platforms);
      if (all_platforms.size() == 0) {
#if defined(DEBUG)
        std::cout<<" No platforms found. Check OpenCL requirements!\n";
#endif
        exit(1);
      }
#if defined(DEBUG)
      else {
        std::cout << "Available platforms are: ";
        for (const auto &platform : all_platforms)
          std::cout << platform.getInfo<CL_PLATFORM_NAME>() << ", ";
        std::cout << "\b\b \n";
      }
#endif
      cl::Platform default_platform=all_platforms[opencl_settings::default_platform];
#if defined(DEBUG)
      std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
#endif
      //get default device of the default platform
      std::vector<cl::Device> all_devices;
      default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
      if (all_devices.size()==0) {
#if defined(DEBUG)
          std::cout<<" No devices found. Check OpenCL requirements!\n";
#endif
          exit(1);
      }
#if defined(DEBUG)
      else {
        std::cout << "Available devices are: ";
        for (const auto &device : all_devices)
          std:: cout << device.getInfo<CL_DEVICE_NAME>() << ", ";
        std::cout << "\b\b \n";
      }
#endif
      device=all_devices[0];
#if defined(DEBUG)
      std::cout << "Using device: "<< device.getInfo<CL_DEVICE_NAME>() << "\n";
#endif
      ctx = cl::Context({device});
      if (opencl_settings::WGS == 0)
        device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &opencl_settings::WGS);
#if defined(DEBUG)
      std::cout << "Using work group size: " << opencl_settings::WGS << std::endl;
#endif
      INIT_GPU = true;
    }
    // build all of the kernels
    std::vector<std::string> kernels;
    kernels.push_back(
      "void kernel Transpose(global const DTYPE *x, global DTYPE *res) {"
      "  size_t rows = get_global_size(0);"
      "  size_t cols = get_global_size(1);"
      "  size_t i = get_global_id(0);"
      "  size_t k = get_global_id(1);"
      "  res[k*rows+i] = x[i*cols+k];"
      "}");
    kernels.push_back(
      "void kernel Exp(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = (DTYPE)exp((float)x[i]);"
      "}");
    kernels.push_back(
      "void kernel Log(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = (DTYPE)log((float)x[i]);"
      "}");
    kernels.push_back(
      "void kernel Sqrt(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = (DTYPE)sqrt((float)x[i]);"
      "}");
    kernels.push_back(
      "void kernel Inv(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = 1/x[i];"
      "}");
    kernels.push_back(
      "void kernel Max(const unsigned long n, global const DTYPE *x, global DTYPE *res) {"
      "  DTYPE var = 0;"
      "  for (size_t i = 0; i < n; ++i)"
      "    if (var < x[i]) var = x[i];"
      "  res[0] = var;"
      "}");
    kernels.push_back(
      "void kernel Sum(const unsigned long n, global const DTYPE *x, global DTYPE *res) {"
      "  DTYPE var = 0;"
      "  for (size_t i = 0; i < n; ++i)"
      "    var += x[i];"
      "  res[0] = var;"
      "}");
    kernels.push_back(
      "void kernel Add(global const DTYPE *x, global const DTYPE *y, global DTYPE *res) {"
      "  size_t n1 = get_global_id(0);"
      "  size_t n2 = get_global_id(1);"
      "  res[n1+n2] = x[n1+n2] + y[n2];"
      "}");
    kernels.push_back(
      "void kernel Sub(global const DTYPE *x, global const DTYPE *y, global DTYPE *res) {"
      "  size_t n1 = get_global_id(0);"
      "  size_t n2 = get_global_id(1);"
      "  res[n1+n2] = x[n1+n2] - y[n2];"
      "}");
    kernels.push_back(
      "void kernel Mul(global const DTYPE *x, global const DTYPE *y, global DTYPE *res) {"
      "  size_t n1 = get_global_id(0);"
      "  size_t n2 = get_global_id(1);"
      "  res[n1+n2] = x[n1+n2] * y[n2];"
      "}");
    kernels.push_back(
      "void kernel Div(global const DTYPE *x, global const DTYPE *y, global DTYPE *res) {"
      "  size_t n1 = get_global_id(0);"
      "  size_t n2 = get_global_id(1);"
      "  res[n1+n2] = x[n1+n2] / y[n2];"
      "}");
    kernels.push_back(
      "void kernel MatMul(const unsigned long x_cols, global const DTYPE *x,"
      "                   global const DTYPE *y, global DTYPE *res) {"
      "  size_t x_rows = get_global_size(0);"
      "  size_t xr = get_global_id(0);"
      "  size_t y_cols = get_global_size(1);"
      "  size_t yc = get_global_id(1);"
      "  DTYPE var = 0;"
      "  for (size_t i = 0; i < x_cols; ++i)"
      "    var += x[xr*x_cols + i] * y[i*y_cols + yc];"
      "  res[xr*y_cols + yc] = var;"
      "}");
    kernels.push_back(
      "void kernel ReLU(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = (DTYPE)max((DTYPE)0, x[i]);"
      "}");
    kernels.push_back(
      "void kernel ReLU_grad(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = (DTYPE)(x[i] > 0);"
      "}");
    kernels.push_back(
      "void kernel Tanh(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = (DTYPE)tanh((float)x[i]);"
      "}");
    kernels.push_back(
      "void kernel Tanh_grad(global const DTYPE *x, global DTYPE *res) {"
      "  size_t i = get_global_id(0);"
      "  res[i] = 1 - x[i]*x[i];"
      "}");

    cl::Program::Sources sources;
    for (auto code : kernels) {
      code = replace_all(code, "DTYPE", type);
      sources.push_back({code.c_str(), code.length()});
    }
    auto program = cl::Program(ctx, sources);
#if defined(DEBUG)
    std::cout << "Compiling OpenCL kernels for type " << type << std::flush;
#endif
    if (program.build({device}) != CL_SUCCESS) {
#if defined(DEBUG)
      std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
#endif
      exit(1);
    }
#if defined(DEBUG)
    std::cout << " OK" << std::endl;
#endif
    types.push_back(type);
    programs.push_back(program);
  }

  template <typename T> void write_data(Buffer<T> &self, std::vector<T> data) {
    self.data = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T)*self.size());
    queue().enqueueWriteBuffer(self.data, CL_TRUE, 0, sizeof(T)*self.size(), &data[0]);
  }

  template <typename T> Buffer<T> copy(const Buffer<T> &self) {
    auto res = Buffer<T>::with_shape(self.shape());
    queue().enqueueCopyBuffer(self.data, res.data, 0, 0, sizeof(T)*self.size());
    return res;
  }

  template <typename T> void reshape(Buffer<T> &self, std::vector<size_t> shape) {
    auto n_prev = self.size();
    self._set_shape_directly(shape);
    if (self.size() != n_prev)
      self.data = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(T)*self.size());
  }

  template <typename T> void fill(Buffer<T> &self, T value) {
    queue().enqueueFillBuffer(self.data, value, 0, sizeof(T)*self.size());
  }

  template <typename T, typename F> void generate(Buffer<T> &self, F gen_fn) {
    auto vec = std::vector<T>(self.size(), 0);
    std::generate(vec.begin(), vec.end(), gen_fn);
    self.write_data(vec);
  }

  template <typename T> std::vector<T> to_vector(const Buffer<T> &self) {
    auto vec = std::vector<T>(self.size(), 0);
    queue().enqueueReadBuffer(self.data, CL_TRUE, 0, sizeof(T)*self.size(), &vec[0]);
    return vec;
  }

  template <typename T> Buffer<T> Transpose(const Buffer<T> &x) {
    const auto rows = x.shape().size() > 1 ? *(x.shape().end()-2) : 1;
    const auto cols = x.shape().back();
    auto res = Buffer<T>::with_shape({cols, rows});
    const size_t gs = std::sqrt(opencl_settings::WGS);
    const auto local_x = rows % gs == 0 ? gs : 1;
    const auto local_y = cols % gs == 0 ? gs : 1;
    cl::KernelFunctor<const cl::Buffer &, cl::Buffer &> functor(program<T>(), "Transpose");
    cl::EnqueueArgs args(queue(), cl::NDRange(rows, cols), cl::NDRange(local_x, local_y));
    functor(args, x.data, res.data);
    return res;
  }

  template <typename T> Buffer<T> Exp(const Buffer<T> &x) { return kernel(x, "Exp"); }
  template <typename T> Buffer<T> Log(const Buffer<T> &x) { return kernel(x, "Log"); }
  template <typename T> Buffer<T> Sqrt(const Buffer<T> &x) { return kernel(x, "Sqrt"); } 
  template <typename T> Buffer<T> Inv(const Buffer<T> &x) { return kernel(x, "Inv"); } 
  template <typename T> Buffer<T> Max(const Buffer<T> &x) { return reduce_kernel(x, "Max"); } 
  template <typename T> Buffer<T> Sum(const Buffer<T> &x) { return reduce_kernel(x, "Sum"); } 
  template <typename T> Buffer<T> Add(const Buffer<T> &x, const Buffer<T> &y) { return kernel(x, y, "Add"); } 
  template <typename T> Buffer<T> Sub(const Buffer<T> &x, const Buffer<T> &y) { return kernel(x, y, "Sub"); } 
  template <typename T> Buffer<T> Mul(const Buffer<T> &x, const Buffer<T> &y) { return kernel(x, y, "Mul"); } 
  template <typename T> Buffer<T> Div(const Buffer<T> &x, const Buffer<T> &y) { return kernel(x, y, "Div"); } 

  template <typename T> Buffer<T> MatMul(const Buffer<T> &x, const Buffer<T> &y) {
    const auto x_rows = x.shape().size() > 1 ? *(x.shape().end()-2) : 1;
    const auto x_cols = x.shape().back(), y_cols = y.shape().back();
    auto res = Buffer<T>::with_shape({x_rows, y_cols});
    cl::KernelFunctor<const unsigned long, const cl::Buffer &, const cl::Buffer &, cl::Buffer &> functor(program<T>(), "MatMul");
    const size_t gs = std::sqrt(opencl_settings::WGS);
    const auto local_x = x_rows % gs == 0 ? gs : 1, local_y = y_cols % gs == 0 ? gs : 1;
    cl::EnqueueArgs args(queue(), cl::NDRange(x_rows, y_cols), cl::NDRange(local_x, local_y));
    functor(args, x_cols, x.data, y.data, res.data);
    buffer::join_queues();
    return res;
  }

  template <typename T> Buffer<T> ReLU(const Buffer<T> &x) { return kernel(x, "ReLU"); } 
  template <typename T> Buffer<T> ReLU_grad(const Buffer<T> &x) { return kernel(x, "ReLU_grad"); } 
  template <typename T> Buffer<T> Tanh(const Buffer<T> &x) { return kernel(x, "Tanh"); } 
  template <typename T> Buffer<T> Tanh_grad(const Buffer<T> &x) { return kernel(x, "Tanh_grad"); }
}

#endif // BUFFER_CLBLAS_HPP
