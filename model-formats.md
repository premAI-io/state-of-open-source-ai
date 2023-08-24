# Model Formats

## GGML
[GGML](https://github.com/ggerganov/ggml) is a tensor library for machine learning to enable large models and high performance on commodity hardware - the "GG" refers to the initials of its originator [Georgi Gerganov](https://github.com/ggerganov). In addition to defining low-level machine learning primitives like a tensor type, GGML defines a binary format for distributing large language models (LLMs). [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) are based on it.
+++
It features:
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

### Inference

Here's an example inference with GPT-2 GGML:
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
   2. **Vocabulary:**: These are all the tokens supported for the current model. Here's an [example for GPT-2](https://github.com/ggerganov/ggml/blob/6319ae9ad7bdf9f834b2855d7e9fa70508e82f57/examples/gpt-2/convert-cerebras-to-ggml.py#L78-L83).
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
Currently **No GPU support** present for GGML format models (CPU only), discussion happening [here](https://github.com/ggerganov/llama.cpp/discussions/915).

It's most used projects include:
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

  High-performance inference of [OpenAI's Whisper automatic speech recognition model](https://openai.com/research/whisper)
  The project provides a high-quality speech-to-text solution that runs on Mac, Windows, Linux, iOS, Android, Raspberry Pi, and Web. Used by rewind.ai

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

## FasterTransformer

## TVM

{{ comments }}

See also:
- ["Optimizing for Faster Inference"](https://cameronrwolfe.substack.com/i/135439692/optimizing-for-faster-inference)
- https://github.com/imaurer/awesome-decentralized-llm#training-and-quantization