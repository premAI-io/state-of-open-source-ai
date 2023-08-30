# Model Formats

Current usecases for different ML models in our lives is at an all-time high and it's only going to increase, this gives rise to an ever increasing need for optimizing the models for specific usecases and unique environments it will run on to extract out the highest possible performance out of it. There's recently been rise of various model formats in community, and we will go through few of the most popular ones this year.

## ONNX
[ONNX (Open Neural Network Exchange)](https://onnx.ai/) provides an open source format for AI models by defining an extensible computation graph model, as well as definitions of built-in operators and standard data types. It is [widely supported](https://onnx.ai/supported-tools) and can be found in many frameworks, tools, and hardware enabling interoperability between different frameworks. ONNX is an intermediary representation of your model that lets you easily go from one environment to the next.

### Features and Benefits

```{figure-md} fig-ref
  ![onnx interoperability](assets/model-formats-onnx.png){align=center}

  https://cms-ml.github.io/documentation/inference/onnx.html
```
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

### How to make ONNX compatible?

| Framework / Tool | Installation | Tutorial |
| --- | --- | --- |
| [Caffe](https://github.com/BVLC/caffe) | [apple/coremltools](https://github.com/apple/coremltools) and [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/caffe_coreml_onnx.ipynb) |
| [Caffe2](https://caffe2.ai) | [part of caffe2 package](https://github.com/pytorch/pytorch/tree/master/caffe2/python/onnx) | [Example](https://github.com/onnx/tutorials/blob/main/tutorials/Caffe2OnnxExport.ipynb) |
| [Chainer](https://chainer.org/) | [chainer/onnx-chainer](https://github.com/chainer/onnx-chainer) | [Example](https://github.com/onnx/tutorials/blob/main/tutorials/ChainerOnnxExport.ipynb) |
| [Cognitive Toolkit (CNTK)](https://www.microsoft.com/en-us/cognitive-toolkit/) | [built-in](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine) | [Example](https://github.com/onnx/tutorials/blob/main/tutorials/CntkOnnxExport.ipynb) |
| [CoreML (Apple)](https://developer.apple.com/documentation/coreml) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/coreml_onnx.ipynb) |
| [Keras](https://github.com/keras-team/keras) | [onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) | [Example](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/keras-resnet50.ipynb) | n/a |
| [LibSVM](https://github.com/cjlin1/libsvm) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/libsvm_onnx.ipynb) | n/a |
| [LightGBM](https://github.com/Microsoft/LightGBM) | [onnx/onnxmltools](https://github.com/onnx/onnxmltools) | [Example](https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/converter_scripts/lightgbm_onnx.ipynb) | n/a |
| [MATLAB](https://www.mathworks.com/) | [Deep Learning Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/67296) | [Example](https://www.mathworks.com/help/deeplearning/ref/exportonnxnetwork.html) |
| [ML.NET](https://github.com/dotnet/machinelearning/) | [built-in](https://www.nuget.org/packages/Microsoft.ML/) | [Example](https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/OnnxConversionTest.cs) |
| [MXNet (Apache)](https://mxnet.incubator.apache.org/) | part of mxnet package [docs](https://mxnet.incubator.apache.org/api/python/contrib/onnx.html) [github](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/contrib/onnx) | [Example](https://github.com/onnx/tutorials/blob/main/tutorials/MXNetONNXExport.ipynb) |
| [PyTorch](https://pytorch.org/) | [part of pytorch package](https://pytorch.org/docs/master/onnx.html) | [Example1](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html), [Example2](https://github.com/onnx/tutorials/blob/main/tutorials/PytorchOnnxExport.ipynb), [export for Windows ML](https://github.com/onnx/tutorials/blob/main/tutorials/ExportModelFromPyTorchForWinML.md), [Extending support](https://github.com/onnx/tutorials/blob/main/tutorials/PytorchAddExportSupport.md) |
| [SciKit-Learn](https://scikit-learn.org/) | [onnx/sklearn-onnx](https://github.com/onnx/sklearn-onnx) | [Example](https://onnx.ai/sklearn-onnx/index.html) | n/a |
| [SINGA (Apache)](https://singa.apache.org/) - [Github](https://github.com/apache/incubator-singa/blob/master/python/singa/sonnx.py) (experimental) | [built-in](https://singa.apache.org/docs/installation/) | [Example](https://github.com/apache/incubator-singa/tree/master/examples/onnx) |
| [TensorFlow](https://www.tensorflow.org/) | [onnx/tensorflow-onnx](https://github.com/onnx/tensorflow-onnx) | [Examples](https://github.com/onnx/tutorials/blob/master/tutorials/TensorflowToOnnx-1.ipynb) |

source: https://github.com/onnx/tutorials#converting-to-onnx-format

Many onnx related tutorials can be found under their official [tutorials repository](https://github.com/onnx/tutorials#onnx-tutorials).

### Support

It has support for Inference runtime binding APIs written in [few programming languages](https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages) ([python](https://onnxruntime.ai/docs/install/#python-installs), [rust](https://github.com/microsoft/onnxruntime/tree/main/rust), [js](https://github.com/microsoft/onnxruntime/tree/main/js), [java](https://github.com/microsoft/onnxruntime/tree/main/java), [C#](https://github.com/microsoft/onnxruntime/tree/main/csharp)).

ONNX model's inference depends on the platform which runtime library supports, called Execution Provider. Currently there are few ranging from CPU based, GPU based, IoT/edge based and few others. A full list can be found [here](https://onnxruntime.ai/docs/execution-providers/#summary-of-supported-execution-providers).

Also there are few visualization tools support like [Netron](https://github.com/lutzroeder/Netron) and [more](https://github.com/onnx/tutorials#visualizing-onnx-models) for models converted to ONNX format, highly recommended for debugging purposes.

#### How's ONNX looking for Tomorrow?
Currently ONNX is part of [LF AI Foundation](https://wiki.lfaidata.foundation/pages/viewpage.action?pageId=327683), conducts regular [Steering committee meetings](https://wiki.lfaidata.foundation/pages/viewpage.action?pageId=18481196) and community meetups are held atleast once a year.
Few notable presentations from this year's meetup:
- [ONNX 2.0 Ideas](https://www.youtube.com/watch?v=A3NwCnUOUaU).
- [Analysis of Failures and Risks in Deep Learning Model Converters: A Case Study in the ONNX Ecosystem](https://www.youtube.com/watch?v=2TFP517aoKo).
- [On-Device Training with ONNX Runtime](https://www.youtube.com/watch?v=_fUslaITI2I): enabling training models on edge devices without the data ever leaving the device.
  
Checkout the [full list here](https://wiki.lfaidata.foundation/display/DL/ONNX+Community+Day+2023+-+June+28).


### Limitations
Onnx uses [Opsets](https://onnx.ai/onnx/intro/converters.html#opsets) (Operator sets) number which changes with each ONNX package minor/major releases, new opsets usually introduces new [operators](https://onnx.ai/onnx/operators/index.html). Proper opset needs to be used while creating the onnx model graph.

There are lots of open issues ([1](https://github.com/microsoft/onnxruntime/issues/12880), [2](https://github.com/microsoft/onnxruntime/issues/10303), [3](https://github.com/microsoft/onnxruntime/issues/7233), [4](https://github.com/microsoft/onnxruntime/issues/17116)) where users are getting slower inference speed after converting their models to ONNX format when compared to base model format, it shows that conversion might not be easy for all models. On similar grounds an user comments 3 years ago [here](https://www.reddit.com/r/MachineLearning/comments/lyem1l/discussion_pros_and_cons_of_onnx_format/gqlh8d3) though it's old, few points still seems relevant. [The troubleshooting guide](https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html) by ONNX runtime community can help with commonly faced issues.

Usage of Protobuf for storing/reading of ONNX models also seems to be causing few limitations which is discussed [here](https://news.ycombinator.com/item?id=36870731).

There's a detailed failure analysis ([video](https://www.youtube.com/watch?v=Ks3rPKfiE-Y), [ppt](https://wiki.lfaidata.foundation/download/attachments/84705448/02_pu-ONNX%20Day%20Presentation%20-%20Jajal-Davis.pdf)) done by [James C. Davis](https://davisjam.github.io/) and [Purvish Jajal](https://www.linkedin.com/in/purvish-jajal-989774190/) on ONNX converters.

````{subfigure} AB
:subcaptions: above
:class-grid: outline

```{image} assets/model-formats_onnx-issues.png
:align: left
```
```{image} assets/model-formats_onnx-issues-table.png
:align: right
```
.[Analysis of Failures and Risks in Deep Learning Model Converters](https://arxiv.org/abs/2303.17708.pdf)
````
The top findings were:
- Crash (56%) and Wrong Model (33%) are the most
common symptoms
- The most common failure causes are Incompatibility
and Type problems, each making up ∼25% of causes
- The majority of failures are located with the Node
Conversion stage (74%), with a further 10% in the Graph optimization stage (mostly from tf2onnx).


### License
It's freely available under [Apache License 2.0](https://github.com/onnx/onnx/blob/main/LICENSE).
### Read more
- [How to add support for new ONNX Operator](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md).
- [ONNX Backend Scoreboard](https://onnx.ai/backend-scoreboard/).
- [Intro to ONNX](https://onnx.ai/onnx/intro/).
- [ONNX Runtime](https://onnxruntime.ai/).
- [WONNX: GPU based ONNX inference runtime in Rust](https://github.com/webonnx/wonnx).
- [Hackernews discussion thread on ONNX runtimes and ONNX](https://news.ycombinator.com/item?id=36863522).


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
   2. **Vocabulary:** These are all supported tokens for a model. Here's an [example for GPT-2](https://github.com/ggerganov/ggml/blob/6319ae9ad7bdf9f834b2855d7e9fa70508e82f57/examples/gpt-2/convert-cerebras-to-ggml.py#L78-L83).
   3. **Weights:** These are also called parameters of the model. The total number of weights in a model are referred to as the "size" of that model. In GGML format a tensor consists of few components:
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
[Quantization](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) is a process where high-precision foating point values are converted to low-precision values. This overall reduces the resources required to use the values in Tensor, making model easier to run on low resources. GGML supports a number of different quantization strategies (e.g. 4-bit, 5-bit, and 8-bit quantization), each of which offers different trade-offs between efficiency and performance. Check out [this amazing article](https://huggingface.co/blog/merve/quantization) by [Merve](https://huggingface.co/merve) for a quick walkthrough.

### Support

```{admonition} New GGUF format
There's a new successor format to GGML named `GGUF` which is designed to be extensible and unambiguous by containing all the information needed to load a model. To read more about `GGUF` check [this PR](https://github.com/ggerganov/llama.cpp/pull/2398) and read in detail about it [here](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
```

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

### Limitations
- Currently **No GPU support** is present for GGML format models (CPU only), discussion happening [here](https://github.com/ggerganov/llama.cpp/discussions/915).
- Models are mostly quantised versions of actual models, taking slight hit from quality side if not much.

### License
The library and related projects are freely available under the [MIT license](https://github.com/ggerganov/ggml/blob/master/LICENSE).

### Read more
- [GGML - Large Language Models for Everyone](https://github.com/rustformers/llm/blob/main/crates/ggml/README.md): a description of the GGML format provided by the maintainers of the `llm` Rust crate, which provides Rust bindings for GGML
- [marella/ctransformers](https://github.com/marella/ctransformers): Python bindings for GGML models.
- [go-skynet/go-ggml-transformers.cpp](https://github.com/go-skynet/go-ggml-transformers.cpp): Golang bindings for GGML models
- [smspillaz/ggml-gobject](https://github.com/smspillaz/ggml-gobject): GObject-introspectable wrapper for use of GGML on the GNOME platform.
- [Hackernews discussion thread on GGML](https://news.ycombinator.com/item?id=36215651)


## TensorRT

### Features and Benefits

### Usage and Workings


### Support

### Limitations

### License

### Read more




## FasterTransformer (WIP)

Feel free to open a PR :)

## TVM (WIP)

Feel free to open a PR :)

{{ comments }}

See also:
- ["Optimizing for Faster Inference"](https://cameronrwolfe.substack.com/i/135439692/optimizing-for-faster-inference)
- https://github.com/imaurer/awesome-decentralized-llm#training-and-quantization



TODO: write top level content around what all formats are and why it's booming (likely add a picture), update future developments for each section of formats

TODO: add a section at the end saying feel free to make a pr if you want to extend anything, specially `To read more` parts
TODO: thoughts - onnx being truely open sourced, it can be so much more compared to other formats, since there's no single-entity/company benefit kind of situation around it.