# MLOps Engines

See also:

- "7 Frameworks for Serving LLMs" ("comprehensive guide & detailed comparison") https://betterprogramming.pub/frameworks-for-serving-llms-60b7f7b23407
- ["Trends: Optimizing for Faster Inference"](https://cameronrwolfe.substack.com/i/135439692/optimizing-for-faster-inference)
- https://github.com/imaurer/awesome-decentralized-llm

## Difficulties of Working with OpenSource MLOps

## Python Bindings and More

## PyTorch Toolchain - From C/C++ to Python

## Examples

vis. https://docs.bentoml.org esp. https://docs.bentoml.org/en/latest/overview/what-is-bentoml.html#build-applications-with-any-ai-models

### llama.cpp

See also https://finbarr.ca/how-is-llama-cpp-possible

### ONNX Runtime

See also:
- https://onnxruntime.ai/docs/execution-providers
- https://onnxruntime.ai/docs/ecosystem/

### Apache TVM

{{ comments }}

<br />

## (Rough Draft):

# MLOps Engines
<!--truncate-->

## The LLM Lifecycle

![](https://static.premai.io/book/mlops-engines-LLMOps-diagram.jpg)

[MLOps(Machine Learning Operations)](https://blogs.nvidia.com/blog/2020/09/03/what-is-mlops/) is a set of best practices for companies to run their AI in production. There are several components in the MLOps lifecycle, with each component trying to address problems with using AI in an enterprise setting. 

With large language models, the traditional MLOps landscape shifts a little bit and we encounter new problems. While MLOps focuses on model training LLMOps focuses on fine-tuning. Model inference is also integral to both lifecycles and will be a focus of this chapter. 

For this section, we'll be exploring the various open-source runtime engines for LLMs and the potential challenges with running these models in production.

## Challenges With Open-Source MLOps

MLOps has typically been available in two flavors. One is the managed version, where all the components are provided out of the box for a price. The other is a DIY setup where you stitch together various open-source components. [Citation](https://valohai.com/managed-vs-open-source-mlops/)

With large language models, the story is no different. Companies like Hugging Face are pushing for open-source models and datasets whereas closed-source competitors like OpenAI and Anthropic are doing the exact opposite. The three main challenges with open-source MLOps are maintenance, performance, and cost.

![](https://static.premai.io/book/mlops-engines-table.jpg)

### 1. Maintenance

When you use open-source components a lot of the setup and configuration has to be done in-house. Whether is downloading the model, fine-tuning, evaluating, or inferencing, everything has to be done manually. When there are multiple open-source components companies tend to write "glue" code to connect the components together.

If a component goes down or becomes unavailable, it is up to the team to resolve the issue. Because of this, teams have to stay on their toes to quickly fix issues to avoid prolonged periods of downtime for the applications. In the long run with robust and scalable pipelines, this becomes less of an issue, but in the early stages, there is a lot of firefighting for developers to do.

### 2. Performance

"Performance" for AI models could mean multiple things. Performance could mean output quality: how close is the output of the model in comparison to human expectation. Or it could be an operational metric like latency, how much time does it take the model to complete a single request.

To measure the output quality or accuracy of an LLM, there are various datasets the model gets tested on. For an in-depth guide, please refer to this [blog post](https://dev.premai.io/blog/evaluating-open-source-llms) which explains the popular datasets used to benchmark open-source models. For a quick snapshot, the [hugging face leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) can be a good place to start when evaluating the accuracy of an LLM.

For operational metrics like latency and throughput, the hardware being used and the runtime enviroment for the application can play a large role. Many AI models, especially LLMs, run faster on a GPU enabled device. [Citation](https://developer.nvidia.com/blog/supercharging-ai-video-and-ai-inference-performance-with-nvidia-l4-gpus) The same GPU enabled model may have different latency and throughput numbers when tested on an optimized inference server such as [Nvidia Triton](https://developer.nvidia.com/triton-inference-server).

Closed-source models like [Cohere](https://txt.cohere.com/nvidia-boosts-inference-speed-with-cohere/) tend to give better performance from an operational perspective because they come with many of the inference optimizations out of the box. Open-source models on the other hand, need to be manually integrated with inference servers to obtain similar performance.

### 3. Cost

One of the reasons companies prefer to choose an open-source solution is for cost savings. If done correctly, the savings can be huge in the long run. However, many firms underestimate the amount of work required to make an open-source ecosystem work seamlessly. 

Oftentimes, teams have to pay a larger cost upfront when working with open-source LLMs. For example, if you purchased a single GPU enabled node with the lowest configuration from GCP(a2-highgpu-1g (vCPUs: 12, RAM: 85GB, GPU: 1 x A100)) to run an open-source model, it would cost you about $2500 per month. On the flip side, flexible pricing models like ChatGPT cost $0.002 for 1K tokens. The monthly cost for infrastructure is expensive and difficult to maintain. Along with that, teams are constantly experimenting since the technology is so new, which further adds to the cost. 

Due to more maintenance and decreased baseline operational performance, enterprises looking to adopt open-source AI technology would need to make their system highly efficient.

## Inference

Inference is one of the hot topics currently with LLMs in general. Large models like ChatGPT have very low latency and great performance but become more expensive with more usage.

On the flip side, open-source models like [Llama-2](https://registry.premai.io/detail.html?service=llama-2-7b) or [Falcon](https://registry.premai.io/detail.html?service=falcon-7b-instruct) have variants that are much smaller in size, yet it's difficult to match the latency and throughput that ChatGPT provides, while still being cost efficient. [Citation](https://www.cursor.so/blog/llama-inference)

Models that are hosted on Hugging Face do not have the necessary optimizations to run in a production environment. The open-source LLM inferencing market is still evolving so currently there's no silver bullet that can run any open-source LLM at blazing-fast speeds.

Here are a few reasons for why inferencing is slow:
### 1. Models are growing larger in size
* As models grow in size and neural networks become more complex it's no surprise that it's taking longer to get an output

### 2. Python as the choice of programming language for AI
* Python, is inherently slow compared to compiled languages like C++
* The developer-friendly syntax and vast array of libraries have put Python in the spotlight, but when it comes to sheer performance it falls behind many other languages
* To compensate for its performance many inferencing servers convert the Python code into an optimized module. For example, Nvidia's [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) can take a Pytorch model and compile it into [TensorRT](https://developer.nvidia.com/tensorrt-getting-started), which has a much higher performance than native Pytorch
* Similarly, [Llama.cpp](https://github.com/ggerganov/llama.cpp) optimizes the Llama inference code to run in raw C++. Using this optimization, people can run a large language model on their laptops without a dedicated GPU.

### 3. Larger inputs
* Not only do LLMs have billions of parameters, but they perform millions of mathematical calculations for each inference
* To do these massive calculations in a timely manner, GPUs are required to help speed up the process. GPUs have much more memory bandwidth and processing power compared to a CPU. This is why GPUs are in such high demand when it comes to running large language models.

## LLM Inference Optimizers

The previous section explained why LLM inferencing is so difficult. In this section we'll look at some  open-source optimizers that can help make inferencing faster and easier.

### 1. Nvidia Triton Inference Server

![](https://static.premai.io/book/mlops-engines-triton-architecture.png)
[Image Source](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/jetson.html)

This [inference server](https://developer.nvidia.com/triton-inference-server) offers support for multiple model formats such as Pytorch, Tensorflow, ONNX, TensorRT, etc. It uses GPUs efficiently to boost the performance of deep learning models.
* <strong>Concurrent model execution:</strong> This allows multiple models to be executed on 1 or many GPUs in parallel. Multiple requests are routed to each model to execute the tasks in parallel
* <strong>Dynamic Batching:</strong> Combines multiple inference requests into a batch to increase throughput. Requests in each batch can be processed in parallel instead of handling each request sequentially.

Pros:
* High throughput, low latency for serving LLMs on a GPU
* Supports multiple frameworks/backends
* Production level performance
* Works with non-LLM models such as image generation or speech to text

Cons:
* Difficult to set up
* Not compatible with many of the newer LLMs

### 2. Text Generation Inference

![](https://static.premai.io/book/mlops-engines-tgi-architecture.png)
[Image Source](https://github.com/huggingface/text-generation-inference)

[Text Generation Inference](https://github.com/huggingface/text-generation-inference) is an open-source project developed by Hugging Face which optimizes Hugging Face models for inference. Unlike Triton, it's much easier to set up and it supports most of the popular LLMs on Hugging Face.

Pros:
* Supports newer models on Hugging Face
* Easy setup via docker container
* Production-ready

Cons:
* Open-source license has restrictions on commercial usage
* Only works with Hugging Face models

### 3. [vLLM](https://vllm.ai/)

This is an open-source project created by researchers at Berkeley to improve the performance of LLM inferencing. vLLM primarily optimizes LLM throughput via methods like PagedAttention and Continuous Batching. The project is fairly new and there is ongoing development.

Pros:
* Can be used commercially
* Supports many popular Hugging Face models
* Easy to setup

Cons:
* Takes a while for new LLMs to be supported


Many other open-source projects like [BentoML](https://www.bentoml.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Flask](https://flask.palletsprojects.com/en/2.3.x/) have been used for serving models in the past. The reason I have not included them on this list is that these open-source tools do not provide the optimizations you need to run LLMs in production.

LLM inference is quite different from ML inference in the past. These models are much larger and require an extraordinary amount of computing power. 

To meet these requirements, there is ongoing development in both the open-source and private sectors to improve the performance of LLMs. It's up to the community to test out different services to see which one works best for their use case.

## Thoughts About The Future

Due to the challenge of running LLMs, enterprises will opt to use an inference server instead of containerizing the model in-house. Most companies don't have the expertise to optimize these models, but they still want the performance benefits. Inference servers, whether they are open-source or not, will be the path forward.

Another pattern that's emerging is that models will move to the data instead of the data moving to the model. Right now if you call the ChatGPT API, you would be sending your data to the model. Enterprises have worked very hard over the past decade to set up robust data infrastructure in the cloud. It makes a lot more sense to bring the model into the same cloud environment where the data is. This is where open-source models being cloud agnostic have a huge advantage.

## Conclusion:

Before the word "MLOps" was coined, data scientists would manually train and run their models locally. At that time, data scientists were mostly experimenting with smaller statistical models. When they tried to bring this technology into production, they ran into many problems around data storage, data processing, model training, model deployment, and model monitoring. Companies started addressing these challenges and came up with a solution for running AI in production, hence "MLOps". 

Currently, we are in the experimental stage with LLMs. When companies try to use this technology in production, they will encounter a new set of challenges. Building solutions to address these challenges will build on the existing concept of MLOps. 

## (Still need to fix make it Bibtex)
## Citations:

1. Supercharging AI Video and AI Inference Performance with NVIDIA L4 GPUs. https://developer.nvidia.com/blog/supercharging-ai-video-and-ai-inference-performance-with-nvidia-l4-gpus
2. Why GPT-3.5 is (mostly) cheaper than Llama 2. https://www.cursor.so/blog/llama-inference
3. Pros and Cons of Open-Source and Managed MLOps Platforms. https://valohai.com/managed-vs-open-source-mlops
4. Cohere Boosts Inference Speed With NVIDIA Triton Inference Server. https://txt.cohere.com/nvidia-boosts-inference-speed-with-cohere