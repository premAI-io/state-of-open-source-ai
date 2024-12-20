# Hardware

```{admonition} Work in Progress
:class: attention
{{ wip_chapter }}

Some ideas:

- [AI and Memory Wall](https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8)
- https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e#deployment
- https://www.youtube.com/watch?v=r5NQecwZs1A
```

## Machine Learning and GPUs

% TODO: add links/citations

GPUs are particularly well-suited for the types of computations required in AI for several reasons:

1. **Parallelisation**: Deep learning models involve a lot of matrix multiplications and other operations that can be parallelised. A single GPU can have thousands of cores, allowing it to execute many operations simultaneously, which can lead to a significant speedup in training and inference times.
2. **Specialised Hardware**: Modern GPUs have specialised hardware for performing certain types of operations that are common in deep learning, such as matrix multiplications and convolutions. For example, NVIDIA's Volta and Turing architectures include Tensor Cores, which are specialised hardware units designed to accelerate mixed-precision matrix multiply-and-accumulate operations.
3. **High Memory Bandwidth**: GPUs have much higher memory bandwidth compared to CPUs, which allows them to transfer data to and from memory much more quickly. This is important for deep learning models, which often involve large amounts of data.
4. **Software Support**: There is a lot of software support for GPU computing in popular deep learning frameworks like TensorFlow and PyTorch. These frameworks provide high-level APIs that make it easy to develop models and run them on GPUs, without having to write low-level GPU code.
5. **Energy Efficiency**: Training deep learning models can be very computationally intensive, and GPUs are generally more energy-efficient than CPUs for these types of computations.
6. **Availability**: Unlike much other specialized numerical computing hardware, GPUs are mass produced for the consumer market. Although specialized data-center and embedded variants exist, mid-tier GPUs can be [easily purchased](https://www.amazon.com/s?k=nvidia+24gb) by consumers and installed in a workstation or PC.

For these reasons, GPUs are often the preferred hardware for training and deploying deep learning models. That said, there are other types of hardware that can also be used for deep learning, such as TPUs (Tensor Processing Units), which are custom accelerators designed by Google specifically for deep learning.

## Types of GPUs

1. **NVIDIA GPUs**: NVIDIA is currently the dominant player in the GPU market for machine learning applications. Their GPUs are widely used in both research and commercial applications. NVIDIA provides a comprehensive ecosystem of software tools and libraries for machine learning, including CUDA and cuDNN (CUDA Deep Neural Network library), which are essential for training deep neural networks. The NVIDIA A100 GPU, for example, is designed specifically for AI and data analytics.
2. **AMD GPUs**: AMD GPUs are also used for machine learning, but they are not as popular as NVIDIA GPUs. AMD provides the ROCm (Radeon Open Compute) platform, which is an open-source software platform for GPU-enabled HPC (High-Performance Computing) and machine learning applications. However, the software ecosystem for AMD GPUs is not as mature as for NVIDIA GPUs.
3. **Apple Silicon GPUs**: Apple has developed its own GPUs for its Apple Silicon chips, like the M1. These GPUs are optimised for low power consumption and are used in Apple devices like the MacBook Air, MacBook Pro, Mac Mini, and iPad Pro. The performance of these GPUs is quite good for mobile and integrated GPUs, but they are not suitable for high-performance machine learning tasks.
4. **Intel GPUs**: Intel is also developing GPUs for machine learning applications. Their upcoming Intel Xe GPUs are expected to provide competitive performance for machine learning tasks. Intel also provides the oneAPI toolkit, which includes a library (oneDNN) for deep neural networks.
5. **Google TPUs (Tensor Processing Units)**: Although not technically GPUs, Google's TPUs are custom accelerators for machine learning tasks. They are designed to provide high performance and efficiency for both training and inference of machine learning models. TPUs are available through Google's cloud computing services.

Each of these options has its own advantages and disadvantages in terms of performance, power consumption, software support, and cost. NVIDIA GPUs are currently the most popular choice for machine learning applications due to their high performance and mature software ecosystem.

## Programming for GPUs

### NVIDIA GPUs

#### CUDA

To interact with NVIDIA GPUs, you will primarily use CUDA. CUDA is a parallel computing platform & programming model developed by NVIDIA for general computing on its GPUs {cite}`cuda-gpus`.

Here are the main components you will interact with:

1. [**CUDA Toolkit**](https://developer.nvidia.com/cuda-downloads), which includes:
   - **CUDA libraries**: e.g. `cuBLAS` for linear algebra, `cuDNN` for deep learning, and others for FFTs, sparse matrices, and more
   - [**CUDA runtime**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-runtime) (`cudart`)
   - [**CUDA compiler**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#compilation-with-nvcc) (`nvcc`)
   - [**NVIDIA drivers**](https://www.nvidia.com/Download/index.aspx): allow your operating system & programs to communicate with your NVIDIA graphics card
2. [**CUDA Language**](https://docs.nvidia.com/cuda/cuda-c-programming-guide): an extension of the C/C++ programming language which includes [some additional keywords & constructs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#c-language-extensions) for writing parallel code.

Here is a basic workflow for using NVIDIA GPUs:

1. **Install NVIDIA drivers & CUDA Toolkit**, using one of the following (depending on your taste):
   - [Developer download matrix (recommended)](https://developer.nvidia.com/cuda-downloads)
   - [Quickstart guide (slightly more detailed)](https://docs.nvidia.com/cuda/cuda-quick-start-guide)
   - [Quickstart videos (if you prefer eye-candy)](https://developer.nvidia.com/how-to-cuda-c-cpp)
   - Full Guide for [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux) or [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows)
2. [**Write your code**](https://docs.nvidia.com/cuda/cuda-c-programming-guide): Use the CUDA programming language (an extension of C/C++) to write your code. This will involve writing kernel functions that will be executed on the GPU, and host code that will be executed on the CPU.
3. **Compile your code**: Use the NVCC compiler (included in the CUDA Toolkit) to compile your code.
4. **Run your code**: Run your compiled code on an NVIDIA GPU.

For example, here is a simple CUDA program that adds two vectors:

```cpp
#include "cuda_runtime.h"
#include <cstdio>

/// CUDA kernel function for vector addition (dst = srcA + srcB)
__global__ void vectorAdd(float *const dst, const float *const srcA, const float *const srcB, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) dst[i] = srcA[i] + srcB[i];
}

int main(void) {
  // Allocate & initialise host (CPU) & device (GPU) memory
  const int numElements = 1337;
  float *srcA;
  cudaMallocManaged((void **)&srcA, numElements);
  for(int i=0; i<numElements; ++i) srcA[i] = i;
  cudaDeviceSynchronize();
  // ...

  // Launch the vectorAdd kernel
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(dst, srcA, srcB, numElements);
  cudaDeviceSynchronize();

  // clean up memory
  cudaFree((void *)a);
  // ...
}
```

In this example, `srcA`, `srcB`, and `dst` are memory pointers to linear vectors (of size `numElements`). Note that the CUDA compiler automatically converts these to host (CPU) or device (GPU) memory pointers (and copies data between host & device) when appropriate. The `vectorAdd` "kernel" (GPU function) is launched with `blocksPerGrid` blocks, each containing `threadsPerBlock` threads. Each thread computes the sum of one pair of elements from `srcA` and `srcB`, and stores the result in `dst`.

```{admonition} High-level wrappers
:class: seealso
Note that wrappers for other programming languages exists (e.g. [Python](https://developer.nvidia.com/how-to-cuda-python)), allowing control of CUDA GPUs while writing code in more concise & user-friendly languages.

% TODO: RAPIDS, CuPy, CuVec etc
```

#### Vulkan

Vulkan is a low-level graphics and compute API developed by the Khronos Group. It provides fine-grained control over the GPU and is designed to minimise CPU overhead and provide more consistent performance. Vulkan can be used for a variety of applications, including gaming, simulation, and scientific computing.

Vulkan is supported on a wide variety of platforms, including Windows, Linux, macOS (via MoltenVK, a Vulkan implementation that runs on top of Metal), Android, and iOS. Vulkan has a somewhat steep learning curve because it is a very low-level API, but it provides a lot of flexibility and can lead to very high performance.

### AMD GPUs

For AMD GPUs, you can use the ROCm (Radeon Open Compute) platform, which is an open-source software platform for GPU-enabled HPC (High-Performance Computing) and machine learning applications.

Here are the main components of the ROCm platform:

1. **ROCm Runtime**: This is the core of the ROCm platform. It includes the ROCr System Runtime, which is a user-space system runtime for managing GPU applications, and the ROCt Thunk Interface, which provides a low-level interface to the GPU kernel driver.
2. **ROCm Driver**: This is the kernel driver for AMD GPUs. It includes the AMDGPU driver, which is the open-source kernel driver for AMD Radeon graphics cards.
3. **ROCm Libraries**: These are a set of libraries optimised for AMD GPUs. They include rocBLAS for basic linear algebra, rocFFT for fast Fourier transforms, and rocRAND for random number generation.
4. **ROCm Tools**: These are a set of tools for developing and debugging applications on AMD GPUs. They include the ROCm SMI (System Management Interface) for monitoring and managing GPU resources, and the ROCgdb debugger for debugging GPU applications.

To develop applications for AMD GPUs using the ROCm platform, you will need to:

1. **Install the necessary software**: This includes the ROCm platform, and any other libraries or tools you need.
2. **Write your code**: You can use the HIP programming language, which is a C++ runtime API and kernel language that allows you to write portable GPU code that can run on both AMD and NVIDIA GPUs. HIP code can be compiled to run on AMD GPUs using the HIP-Clang compiler, or on NVIDIA GPUs using the NVCC compiler.
3. **Compile your code**: Use the HIP-Clang compiler to compile your code for AMD GPUs, or the NVCC compiler for NVIDIA GPUs.
4. **Run your code**: Run your compiled code on an AMD or NVIDIA GPU.

For example, here is a simple HIP program that adds two vectors:

```cpp
#include "hip/hip_runtime.h"
#include <cstdio>

/// HIP kernel function for vector addition (dst = srcA + srcB)
__global__ void vectorAdd(float *const dst, const float *const srcA, const float *const srcB, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) dst[i] = srcA[i] + srcB[i];
}

int main(void) {
  // Allocate and initialise host (CPU) & device (GPU) memory
  // ...

  // Launch the vectorAdd kernel
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  hipLaunchKernelGGL(
    vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, dst, srcA, srcB, numElements);

  // Copy result from device to host & clean up memory
  // ...
}
```

In this example, `d_A`, `d_B`, and `d_C` are pointers to device memory, and `numElements` is the number of elements in each vector. The `vectorAdd` kernel is launched with `blocksPerGrid` blocks, each containing `threadsPerBlock` threads. Each thread computes the sum of one pair of elements from `d_A` and `d_B`, and stores the result in `d_C`.

Note that this example is very similar to the CUDA example I provided earlier. This is because the HIP programming language is designed to be similar to CUDA, which makes it easier to port CUDA code to run on AMD GPUs.

### Apple Silicon GPUs

#### Metal

Apple Silicon GPUs, which are part of Apple's custom M1 chip, can be programmed using the Metal framework. Metal is a graphics and compute API developed by Apple, and it's available on all Apple devices, including Macs, iPhones, and iPads.

Here are the main components of the Metal framework:

1. **Metal API**: This is a low-level API that provides access to the GPU. It includes functions for creating and managing GPU resources, compiling shaders, and submitting work to the GPU.
2. **Metal Shading Language (MSL)**: This is the programming language used to write GPU code (shaders) in Metal. It is based on the C++14 programming language and includes some additional features and keywords for GPU programming.
3. **MetalKit and Metal Performance Shaders (MPS)**: These are higher-level frameworks built on top of Metal. MetalKit provides functions for managing textures, meshes, and other graphics resources, while MPS provides highly optimised functions for common image processing and machine learning tasks.

Here is a basic workflow for using Metal to perform GPU computations on Apple Silicon:

1. **Install the necessary software**: This includes the Xcode development environment, which includes the Metal framework and compiler.
2. **Write your code**: Write your GPU code using the Metal Shading Language, and your host code using Swift or Objective-C. Your host code will use the Metal API to manage GPU resources and submit work to the GPU.
3. **Compile your code**: Use the Xcode development environment to compile your code.
4. **Run your code**: Run your compiled code on an Apple device with an Apple Silicon GPU.

For example, here is a simple Metal program that adds two vectors:

```swift
import Metal

// Create a Metal device and command queue
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// Create a Metal library and function
let library = device.makeDefaultLibrary()!
let function = library.makeFunction(name: "vector_add")!

// Create a Metal compute pipeline
let pipeline = try! device.makeComputePipelineState(function: function)

// Allocate and initialise host and device memory
let numElements = 1024
let bufferSize = numElements * MemoryLayout<Float>.size
let h_A = [Float](repeating: 1.0, count: numElements)
let h_B = [Float](repeating: 2.0, count: numElements)
let d_A = device.makeBuffer(bytes: h_A, length: bufferSize, options: [])!
let d_B = device.makeBuffer(bytes: h_B, length: bufferSize, options: [])!
let d_C = device.makeBuffer(length: bufferSize, options: [])!

// Create a Metal command buffer and encoder
let commandBuffer = commandQueue.makeCommandBuffer()!
let commandEncoder = commandBuffer.makeComputeCommandEncoder()!

// Set the compute pipeline and buffers
commandEncoder.setComputePipelineState(pipeline)
commandEncoder.setBuffer(d_A, offset: 0, index: 0)
commandEncoder.setBuffer(d_B, offset: 0, index: 1)
commandEncoder.setBuffer(d_C, offset: 0, index: 2)

// Dispatch the compute kernel
let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
let numThreadgroups = MTLSize(width: (numElements + 255) / 256, height: 1, depth: 1)
commandEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

// End the command encoder and commit the command buffer
commandEncoder.endEncoding()
commandBuffer.commit()

// Wait for the command buffer to complete
commandBuffer.waitUntilCompleted()

// Copy the result from device to host
let h_C = UnsafeMutablePointer<Float>.allocate(capacity: numElements)
d_C.contents().copyMemory(to: h_C, byteCount: bufferSize)

// ...
// Clean up
// ...
```

In this example, `d_A`, `d_B`, and `d_C` are Metal buffers, and `numElements` is the number of elements in each vector. The `vector_add` function is a Metal shader written in the Metal Shading Language, and it is executed on the GPU using a Metal compute command encoder.

Note that this example is written in Swift, which is the recommended programming language for developing Metal applications. You can also use Objective-C, but Swift is generally preferred for new development.

This example is quite a bit more complex than the earlier CUDA and HIP examples, because Metal is a lower-level API that provides more fine-grained control over the GPU. This can lead to more efficient code, but it also requires more boilerplate code to set up and manage GPU resources.

#### Metal Performance Shaders (MPS)

**Metal Performance Shaders (MPS)** is a framework that provides highly optimised functions for common image processing and machine learning tasks. MPS is built on top of the Metal framework and is available on all Apple devices, including Macs, iPhones, and iPads.

MPS includes a variety of functions for image processing (e.g., convolution, resizing, and histogram calculation), as well as a set of neural network layers (e.g., convolution, pooling, and normalisation) that can be used to build and run neural networks on the GPU.

MPS is a higher-level API than Metal, which makes it easier to use, but it provides less flexibility. If you are developing an application for Apple devices and you need to perform image processing or machine learning tasks, MPS is a good place to start.

### Cross Platform Graphics APIs

#### Vulkan

**Vulkan** is a low-level graphics and compute API developed by the Khronos Group. It provides fine-grained control over the GPU and is designed to minimise CPU overhead and provide more consistent performance. Vulkan can be used for a variety of applications, including gaming, simulation, and scientific computing.

Vulkan is supported on a wide variety of platforms, including Windows, Linux, macOS (via MoltenVK, a Vulkan implementation that runs on top of Metal), Android, and iOS. Vulkan has a somewhat steep learning curve because it is a very low-level API, but it provides a lot of flexibility and can lead to very high performance.

Vulkan is designed to be a cross-platform API. It is supported on a wide variety of platforms, including Windows, Linux, macOS (via MoltenVK, a layer that maps Vulkan to Metal), Android, and iOS. This makes it a good choice for developing applications that need to run on multiple platforms.

#### OpenGL

**OpenGL** is a cross-platform graphics API developed by the Khronos Group. It is widely used for developing graphics applications, including games, simulations, and design tools. OpenGL is a higher-level API than Vulkan, which makes it easier to use, but it provides less control over the GPU and may have more CPU overhead.

OpenGL is supported on a wide variety of platforms, including Windows, macOS, Linux, and Android. However, Apple has deprecated OpenGL on its platforms in favor of Metal, so if you are developing an application for Apple devices, it is recommended to use Metal instead of OpenGL.

Each of these APIs has its own strengths and weaknesses, and the best one to use depends on your specific application and requirements. If you are developing a cross-platform application and need a low-level API, Vulkan is a good choice. If you are developing an application for Apple devices and need to perform image processing or machine learning tasks, MPS is a good choice. If you are developing a graphics application and need a higher-level API, OpenGL may be a good choice, although you should consider using Metal on Apple devices.

#### DirectX

**DirectX** is a collection of APIs for handling tasks related to multimedia, game programming, and video, on Microsoft platforms. While it's most commonly associated with Windows, it is also available on Xbox. Note that DirectX is not fully cross-platform, as it doesn't support macOS or Linux.

#### OpenCL

**OpenCL** is a framework for writing programs that execute across heterogeneous platforms consisting of CPUs, GPUs, and other processors. OpenCL includes a language (based on C99) for writing kernels (i.e., functions that run on the hardware devices), plus APIs that are used to define and then control the platforms. OpenCL provides parallel computing using task-based and data-based parallelism.

#### WebGL and WebGPU

**WebGL** is a web-based graphics API that is based on OpenGL ES. It allows you to create 3D graphics in a web browser. Since it's web-based, it is supported on all major platforms and web browsers. While on the other hand, **WebGPU** is a new web-based graphics and compute API that is currently being developed by the W3C GPU for the Web Community Group. It is designed to provide modern 3D graphics and computation capabilities in web browsers, and it is intended to be the successor to WebGL.

WebGPU aims to provide a more modern and lower-level API than WebGL, which will allow for better performance and more flexibility. It is designed to be a web-friendly API that can be implemented on top of other graphics APIs, such as Vulkan, Metal, and DirectX.

WebGPU is still in development, and it is not yet widely supported in web browsers. However, it is an exciting development for web-based graphics and computation, and it is worth keeping an eye on if you are developing web applications that require high-performance graphics or computation.

WebGPU will be a cross-platform API because it will be supported in web browsers on multiple platforms. However, the actual implementation of WebGPU in the browser may use different underlying graphics APIs, depending on the platform. For example, a browser on Windows may use a DirectX-based implementation of WebGPU, while a browser on macOS may use a Metal-based implementation. This will be transparent to the application developer, who will just use the WebGPU API.

```{admonition} Work in Progress
:class: attention
An entire chapter will be dedicated to WebGPU (coming soon!)
```

### Benchmarks

```{admonition} Work in Progress
:class: attention
Table with benchmarks
```

### Acceleration Libraries

- **OpenBLAS**
- **CuBLAS**
- **cuDNN**
- **OpenCL**

## Cloud

- cost comparisons
  + user-friendly: https://fullstackdeeplearning.com/cloud-gpus
  + less user-friendly but more comprehensive: https://cloud-gpus.com
  + comparisons of both features and pricing for GPU cloud providers: https://www.gpucloudpricing.com
  + LLM-specific advice: https://gpus.llm-utils.org/cloud-gpu-guide/#which-gpu-cloud-should-i-use

## Future

One problem with using current {term}`LLMs <LLM>` is the high GPU memory requirements. One popular work-around is {term}`quantisation`. However, this requires hardware manufacturers to build support for quantised operations ({term}`SIMD` instruction sets), and ML libraries to rewrite/reimplement core parts of their codebase to support the new operations. Also recall that CPU-based SIMD instruction sets (e.g. [SSE4](https://en.wikipedia.org/wiki/SSE4) & [AVX10](https://en.wikipedia.org/wiki/AVX10) for PCs and [NEON](<https://en.wikipedia.org/wiki/ARM_architecture_family#Advanced_SIMD_(Neon)>) for mobiles) took many years to develop, and are still actively evolving. By comparison, GPU architectures have much less adoption & development, so new arithmetic operations will take years to be widely supported.

{{ comments }}
