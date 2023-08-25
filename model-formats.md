# Model Formats

## GGML
[GGML](https://github.com/ggerganov/ggml) is a tensor library for machine learning to enable large models and high performance on commodity hardware - the "GG" refers to the initials of its originator [Georgi Gerganov](https://github.com/ggerganov). In addition to defining low-level machine learning primitives like a tensor type, GGML defines a binary format for distributing large language models (LLMs). [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) are based on it.

### Features and Benefits
- Written in C
- 16-bit float support
- Integer quantization support (e.g. 4-bit, 5-bit, 8-bit)
- Automatic differentiation
- Built-in optimization algorithms (e.g. ADAM, L-BFGS)
- Optimized for Apple Silicon
- On x86 architectures utilizes AVX / AVX2 intrinsics
- Web support via WebAssembly and WASM SIMD
- No third-party dependencies
- Zero memory allocations during runtime
- Guided language output support

To know more, see their [manifesto here](https://github.com/ggerganov/llama.cpp/discussions/205)

### Usage

Here's an example inference of GPT-2 GGML:
```python
git clone https://github.com/ggerganov/ggml
cd ggml
mkdir build && cd build
cmake ..
make -j4 gpt-2

# Run the GPT-2 small 117M model
../examples/gpt-2/download-ggml-model.sh 117M
./bin/gpt-2 -m models/gpt-2-117M/ggml-model.bin -p "This is an example"
```

### Working
For usage, the model should be saved in the particular GGML file format which consists binary-encoded data that has a particular format specifying what kind of data is present in the file, how it is represented, and the order in which it appears.

For a valid GGML file the following pieces of information should be present in order:
1. **GGML version number:** To support rapid development without sacrificing backwards-compatibility, GGML uses versioning to introduce improvements that may change the format of the encoding. The first value present in a valid GGML file is a "magic number" that indicates the GGML version that was used to encode the model.
Here's a [GPT-2 conversion example](https://github.com/ggerganov/ggml/blob/6319ae9ad7bdf9f834b2855d7e9fa70508e82f57/examples/gpt-2/convert-cerebras-to-ggml.py#L67) where it's getting written.
1. **Components of LLMs:**
   1. **Hyperparameters:** These are parameters which configures the behaviour of models. Valid GGML files lists these values in the correct order, and each value represented using the correct data type. Here's an [example for GPT-2](https://github.com/ggerganov/ggml/blob/6319ae9ad7bdf9f834b2855d7e9fa70508e82f57/examples/gpt-2/convert-cerebras-to-ggml.py#L68-L72).
   2. **Vocabulary:**: These are all supported tokens for a model. Here's an [example for GPT-2](https://github.com/ggerganov/ggml/blob/6319ae9ad7bdf9f834b2855d7e9fa70508e82f57/examples/gpt-2/convert-cerebras-to-ggml.py#L78-L83).
   3. **Weights:**: These are also called parameters of the model. The total number of weights in a model are referred to as the "size" of that model. In GGML format a tensor consists of few components:
     - Name
     - 4 element list representing number of dimensions in the tensor and their lengths
     - List of weights in the tensor
      
      Let's consider the following weights:
      ```
      weight_1 = [[0.334, 0.21], [0.0, 0.149]]
      weight_2 = [0.123, 0.21, 0.31]
      ```
      Then GGML representation would be:
      ```
      {"weight_1", [2, 2, 1, 1], [0.334, 0.21, 0.0, 0.149]}
      {"weight_2", [3, 1, 1, 1], [0.123, 0.21, 0.31]}
      ```
      For each weight representation the first list denotes dimensions and second list denotes weights. Dimensions list uses `1` as a placeholder for unused dimensions.

#### Quantization
[Quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) is a process where high-precision foating point values are converted to low-precision values. This overall reduces the resources required to use the values in Tensor, making model easier to run on low resources. GGML supports a number of different quantization strategies (e.g. 4-bit, 5-bit, and 8-bit quantization), each of which offers different trade-offs between efficiency and performance.

### Support

```{admonition} New GGUF format
There's a new successor format to GGML named `GGUF` which is designed to be extensible and unambiguous by containing all the information needed to load a model. To read more about `GGUF` check [this PR](https://github.com/ggerganov/llama.cpp/pull/2398) and read in detail about it [here](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
```

Currently **No GPU support** is present for GGML format models (CPU only), discussion happening [here](https://github.com/ggerganov/llama.cpp/discussions/915).

It's most used projects include:
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

  High-performance inference of [OpenAI's Whisper automatic speech recognition model](https://openai.com/research/whisper)
  The project provides a high-quality speech-to-text solution that runs on Mac, Windows, Linux, iOS, Android, Raspberry Pi, and Web. Used by [rewind.ai](https://www.rewind.ai/)

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
  
  Inference of Meta's LLaMA large language model

  The project demonstrates efficient inference on Apple Silicon hardware and explores a variety of optimization techniques and applications of LLMs

Inference and training of many open sourced models ([StarCoder](https://github.com/ggerganov/ggml/tree/master/examples/starcoder), [Falcon](https://github.com/cmp-nct/ggllm.cpp), [Replit](https://github.com/ggerganov/ggml/tree/master/examples/replit), [Bert](https://github.com/skeskinen/bert.cpp), etc.) are already supported in GGML. Track the full list of updates [here](https://github.com/ggerganov/ggml#updates).

```{tip}
[TheBloke](https://huggingface.co/TheBloke) currently has lots of LLM variants already converted to GGML format.
```

#### Resources
- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md): a description of the GGML format provided by the maintainers of the `llm` Rust crate, which provides Rust bindings for GGML
- [marella/ctransformers](https://github.com/marella/ctransformers): Python bindings for GGML models.
- [go-skynet/go-ggml-transformers.cpp](https://github.com/go-skynet/go-ggml-transformers.cpp): Golang bindings for GGML models
- [smspillaz/ggml-gobject](https://github.com/smspillaz/ggml-gobject): GObject-introspectable wrapper for use of GGML on the GNOME platform.
- [Hackernews discussion thread on GGML](https://news.ycombinator.com/item?id=36215651)

### License
The library and related projects are freely available under the [MIT license](https://github.com/ggerganov/ggml/blob/master/LICENSE).

## ONNX
[ONNX (Open Neural Network Exchange)](https://onnx.ai/) provides an open source format for AI models by defining an extensible computation graph model, as well as definitions of built-in operators and standard data types. It is [widely supported](https://onnx.ai/supported-tools) and can be found in many frameworks, tools, and hardware enabling interoperability between different frameworks. ONNX is an intermediary representation of your model that lets you easily go from one environment to the next.

### Features and Benefits
![onnx interoperability](assets/model-formats-onnx.png)(source: https://cms-ml.github.io/documentation/inference/onnx.html)
- **Model Interoperability:** ONNX bridges AI frameworks, allowing seamless model transfer between them, eliminating the need for complex conversions.

- **Computation Graph Model:** ONNX's core is a graph model, representing AI models as directed graphs with nodes for operations, offering flexibility.

- **Standardized Data Types:** ONNX establishes standard data types, ensuring consistency when exchanging models, reducing data type issues.

- **Built-in Operators:** ONNX boasts a rich library of operators for common AI tasks, enabling consistent computation across frameworks.

- **ONNX Ecosystem:**
  - **[ONNX Runtime](https://github.com/microsoft/onnxruntime):** A high-performance inference engine for cross-platform ONNX models.
  - **[ONNX ML Tools](https://github.com/onnx/onnxmltools):** Tools for ONNX model conversion and compatibility with frameworks like TensorFlow and PyTorch.
  - **[ONNX Model Zoo](https://github.com/onnx/models):** A repository of pre-trained models converted to ONNX format for various tasks.
  - **[ONNX Hub](https://github.com/onnx/onnx/blob/main/docs/Hub.md):** Helps sharing and collaborating on ONNX models within the community.

### Usage

Firstly the model needs to be converted to ONNX format using a relevant [converter](https://onnx.ai/onnx/intro/converters.html), for example if our model is created using Pytorch, for conversion we can use:
-  [`torch.onnx.export`](https://pytorch.org/docs/master/onnx.html)
-  [`optimum`](https://github.com/huggingface/optimum#onnx--onnx-runtime) by [huggingface](https://huggingface.co/docs/transformers/serialization#export-to-onnx)

Once exported we can load, manipulate, and run ONNX models. Let's take a Python example:

To install the official `onnx` python package:
```sh
pip install onnx
```

To load, manipulate, and run ONNX models in your Python applications:
```python
import onnx

# Load an ONNX model
model = onnx.load("your_awesome_model.onnx")

# Perform inference with the model
# (Specific inference code depends on your application and framework)
```

### Working

### Support

#### Updates
TODO: add updates from https://wiki.lfaidata.foundation/display/DL/ONNX+Community+Day+2023+-+June+28

#### Resources

### License
It's freely available under [Apache License 2.0](https://github.com/onnx/onnx/blob/main/LICENSE).


## FasterTransformer

## TVM

{{ comments }}

See also:
- ["Optimizing for Faster Inference"](https://cameronrwolfe.substack.com/i/135439692/optimizing-for-faster-inference)
- https://github.com/imaurer/awesome-decentralized-llm#training-and-quantization