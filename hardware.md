# Hardware

See also:
- "AI and Memory Wall" https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8
- https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e#deployment
- https://www.youtube.com/watch?v=r5NQecwZs1A

## Machine Learning and GPUs

GPUs are particularly well-suited for the types of computations required in AI for several reasons:

1. **Parallelization**: Deep learning models involve a lot of matrix multiplications and other operations that can be parallelized. A single GPU can have thousands of cores, allowing it to execute many operations simultaneously, which can lead to a significant speedup in training and inference times.
2. **Specialized Hardware**: Modern GPUs have specialized hardware for performing certain types of operations that are common in deep learning, such as matrix multiplications and convolutions. For example, NVIDIA's Volta and Turing architectures include Tensor Cores, which are specialized hardware units designed to accelerate mixed-precision matrix multiply-and-accumulate operations.
3. **High Memory Bandwidth**: GPUs have much higher memory bandwidth compared to CPUs, which allows them to transfer data to and from memory much more quickly. This is important for deep learning models, which often involve large amounts of data.
4. **Software Support**: There is a lot of software support for GPU computing in popular deep learning frameworks like TensorFlow and PyTorch. These frameworks provide high-level APIs that make it easy to develop models and run them on GPUs, without having to write low-level GPU code.
5. **Energy Efficiency**: Training deep learning models can be very computationally intensive, and GPUs are generally more energy-efficient than CPUs for these types of computations.

For these reasons, GPUs are often the preferred hardware for training and deploying deep learning models. That said, there are other types of hardware that can also be used for deep learning, such as TPUs (Tensor Processing Units), which are custom accelerators designed by Google specifically for deep learning.

## Types of GPUs

1. **NVIDIA GPUs**: NVIDIA is currently the dominant player in the GPU market for machine learning applications. Their GPUs are widely used in both research and commercial applications. NVIDIA provides a comprehensive ecosystem of software tools and libraries for machine learning, including CUDA and cuDNN (CUDA Deep Neural Network library), which are essential for training deep neural networks. The NVIDIA A100 GPU, for example, is designed specifically for AI and data analytics.
2. **AMD GPUs**: AMD GPUs are also used for machine learning, but they are not as popular as NVIDIA GPUs. AMD provides the ROCm (Radeon Open Compute) platform, which is an open-source software platform for GPU-enabled HPC (High-Performance Computing) and machine learning applications. However, the software ecosystem for AMD GPUs is not as mature as for NVIDIA GPUs.
3. **Apple Silicon GPUs**: Apple has developed its own GPUs for its Apple Silicon chips, like the M1. These GPUs are optimized for low power consumption and are used in Apple devices like the MacBook Air, MacBook Pro, Mac Mini, and iPad Pro. The performance of these GPUs is quite good for mobile and integrated GPUs, but they are not suitable for high-performance machine learning tasks.
4. **Intel GPUs**: Intel is also developing GPUs for machine learning applications. Their upcoming Intel Xe GPUs are expected to provide competitive performance for machine learning tasks. Intel also provides the oneAPI toolkit, which includes a library (oneDNN) for deep neural networks.
5. **Google TPUs (Tensor Processing Units)**: Although not technically GPUs, Google's TPUs are custom accelerators for machine learning tasks. They are designed to provide high performance and efficiency for both training and inference of machine learning models. TPUs are available through Google's cloud computing services.

Each of these options has its own advantages and disadvantages in terms of performance, power consumption, software support, and cost. NVIDIA GPUs are currently the most popular choice for machine learning applications due to their high performance and mature software ecosystem.

## Programming for GPUs

### NVIDIA GPUs

To interact with NVIDIA GPUs, you will primarily use the NVIDIA CUDA platform. CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on its GPUs.

Here are the main components you will interact with:

1. **CUDA Toolkit**: Install the CUDA Toolkit, which includes GPU-accelerated libraries, a compiler, development tools and the CUDA runtime.
2. **CUDA Language**: This is an extension of the C/C++ programming language. You will write your code in CUDA C/C++, which includes some additional keywords and constructs for writing parallel code.
3. **CUDA Libraries**: NVIDIA provides a variety of libraries that are optimized for their GPUs, such as cuBLAS for linear algebra, cuDNN for deep learning, and others for FFTs, sparse matrices, and more.
4. **NVIDIA GPU Drivers**: Make sure you have the correct drivers installed for your GPU. These drivers allow your operating system and programs to communicate with your NVIDIA graphics card.

Here is a basic workflow for using NVIDIA GPUs:

1. **Install the necessary software**: This includes the NVIDIA GPU drivers, the CUDA Toolkit, and any other libraries or software development kits (SDKs) you need.
2. **Write your code**: Use the CUDA programming language (an extension of C/C++) to write your code. This will involve writing kernel functions that will be executed on the GPU, and host code that will be executed on the CPU.
3. **Compile your code**: Use the NVCC compiler (included in the CUDA Toolkit) to compile your code.
4. **Run your code**: Run your compiled code on an NVIDIA GPU.

For example, here is a simple CUDA program that adds two vectors:

```C
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel function for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    // ...
    // Allocate and initialize host and device memory
    // ...

    // Launch the vectorAdd kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    // ...
    // Copy result from device to host and clean up
    // ...
}
```

In this example, `d_A`, `d_B`, and `d_C` are pointers to device memory, and `numElements` is the number of elements in each vector. The `vectorAdd` kernel is launched with `blocksPerGrid` blocks, each containing `threadsPerBlock` threads. Each thread computes the sum of one pair of elements from `d_A` and `d_B`, and stores the result in `d_C`.

### AMD GPUs

For AMD GPUs, you can use the ROCm (Radeon Open Compute) platform, which is an open-source software platform for GPU-enabled HPC (High-Performance Computing) and machine learning applications.

Here are the main components of the ROCm platform:

1. **ROCm Runtime**: This is the core of the ROCm platform. It includes the ROCr System Runtime, which is a user-space system runtime for managing GPU applications, and the ROCt Thunk Interface, which provides a low-level interface to the GPU kernel driver.
2. **ROCm Driver**: This is the kernel driver for AMD GPUs. It includes the AMDGPU driver, which is the open-source kernel driver for AMD Radeon graphics cards.
3. **ROCm Libraries**: These are a set of libraries optimized for AMD GPUs. They include rocBLAS for basic linear algebra, rocFFT for fast Fourier transforms, and rocRAND for random number generation.
4. **ROCm Tools**: These are a set of tools for developing and debugging applications on AMD GPUs. They include the ROCm SMI (System Management Interface) for monitoring and managing GPU resources, and the ROCgdb debugger for debugging GPU applications.

To develop applications for AMD GPUs using the ROCm platform, you will need to:

1. **Install the necessary software**: This includes the ROCm platform, and any other libraries or tools you need.
2. **Write your code**: You can use the HIP programming language, which is a C++ runtime API and kernel language that allows you to write portable GPU code that can run on both AMD and NVIDIA GPUs. HIP code can be compiled to run on AMD GPUs using the HIP-Clang compiler, or on NVIDIA GPUs using the NVCC compiler.
3. **Compile your code**: Use the HIP-Clang compiler to compile your code for AMD GPUs, or the NVCC compiler for NVIDIA GPUs.
4. **Run your code**: Run your compiled code on an AMD or NVIDIA GPU.

For example, here is a simple HIP program that adds two vectors:

```C
#include <hip/hip_runtime.h>
#include <stdio.h>

// HIP kernel function for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    // ...
    // Allocate and initialize host and device memory
    // ...

    // Launch the vectorAdd kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, numElements);

    // ...
    // Copy result from device to host and clean up
    // ...
}
```

In this example, `d_A`, `d_B`, and `d_C` are pointers to device memory, and `numElements` is the number of elements in each vector. The `vectorAdd` kernel is launched with `blocksPerGrid` blocks, each containing `threadsPerBlock` threads. Each thread computes the sum of one pair of elements from `d_A` and `d_B`, and stores the result in `d_C`.

Note that this example is very similar to the CUDA example I provided earlier. This is because the HIP programming language is designed to be similar to CUDA, which makes it easier to port CUDA code to run on AMD GPUs.

## NVIDIA GPUs

## Apple GPUs

### Mac

### iPhone

## Cloud

- cost comparisons
  + user-friendly: https://fullstackdeeplearning.com/cloud-gpus
  + less user-friendly but more comprehensive: https://cloud-gpus.com
  + LLM-specific advice: https://gpus.llm-utils.org/cloud-gpu-guide/#which-gpu-cloud-should-i-use

{{ comments }}
