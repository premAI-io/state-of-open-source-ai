# MLOps Engines

```{figure-md} llm-lifecycle
:class: caption
![](https://static.premai.io/book/mlops-engines-LLMOps-diagram.jpg)

The LLM Lifecycle
```

[MLOps(Machine Learning Operations)](https://blogs.nvidia.com/blog/2020/09/03/what-is-mlops/) is a set of best practices for companies to run their AI in production. There are several components in the MLOps lifecycle, with each component trying to address problems with using AI in an enterprise setting. 

With large language models, the traditional MLOps landscape shifts a little bit and we encounter new problems. While MLOps focuses on model training LLMOps focuses on fine-tuning. Model inference is also integral to both lifecycles and will be a focus of this chapter. 

For this section, we'll be exploring the various open-source runtime engines for LLMs and the potential challenges with running these models in production.

## Challenges With Open-Source MLOps

MLOps has typically been available in two flavors. One is the managed version, where all the components are provided out of the box for a price. The other is a DIY setup where you stitch together various open-source components. {cite}`mlops-challenges`

With large language models, the story is no different. Companies like Hugging Face are pushing for open-source models and datasets whereas closed-source competitors like OpenAI and Anthropic are doing the exact opposite. The three main challenges with open-source MLOps are maintenance, performance, and cost.



```{figure-md} open-vs-closed-mlops
:class: caption
![](https://static.premai.io/book/mlops-engines-table.jpg)

Open-Source vs Closed-Source MLOps
```

### 1. Maintenance

When using open-source components, most of the setup and configuration has to be done in-house. Whether is downloading the model, fine-tuning, evaluating, or inferencing, everything has to be done manually. When there are multiple open-source components companies tend to write "glue" code to connect the components together.

If a component goes down or becomes unavailable, it is up to the team to resolve the issue. Because of this, teams have to stay on their toes to quickly fix issues to avoid prolonged periods of downtime for the applications. In the long run with robust and scalable pipelines, this becomes less of an issue, but in the early stages, there is a lot of firefighting for developers to do.

### 2. Performance

"Performance" for AI models could mean multiple things. Performance could mean output quality: how close is the output of the model in comparison to human expectation. Or it could be an operational metric like latency, how much time does it take the model to complete a single request.

To measure the output quality or accuracy of an LLM, there are various datasets the model gets tested on. For an in-depth guide, please refer to this [blog post](https://dev.premai.io/blog/evaluating-open-source-llms) which explains the popular datasets used to benchmark open-source models. For a quick snapshot, the [hugging face leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) can be a good place to start when evaluating the accuracy of an LLM.

For operational metrics like latency and throughput, the hardware being used and the runtime enviroment of the application can play a large role. Many AI models, especially LLMs, run faster on a GPU enabled device. {cite}`nvidia-gpu-inference` The same GPU enabled model may have different latency and throughput numbers when tested on an optimized inference server such as [Nvidia Triton](https://developer.nvidia.com/triton-inference-server).

Closed-source models like [Cohere](https://txt.cohere.com/nvidia-boosts-inference-speed-with-cohere/) tend to give better baseline performance from an operational perspective because they come with many of the inference optimizations out of the box. Open-source models on the other hand, need to be manually integrated with inference servers to obtain similar performance. {cite}`cohere-trition`

### 3. Cost

One of the reasons companies prefer to choose an open-source solution is for cost savings. If done correctly, the savings can be huge in the long run. However, many firms underestimate the amount of work required to make an open-source ecosystem work seamlessly. 

Oftentimes, teams have to pay a larger cost upfront when working with open-source LLMs. For example, if you purchased a single GPU enabled node with the lowest configuration from GCP(a2-highgpu-1g (vCPUs: 12, RAM: 85GB, GPU: 1 x A100)) to run an open-source model, it would cost you about $2500 per month. On the flip side, flexible pricing models like ChatGPT cost $0.002 for 1K tokens. The monthly cost for infrastructure is expensive and difficult to maintain. Along with that, teams are constantly experimenting since the technology is so new, which further adds to the cost. 

Due to more maintenance and decreased baseline operational performance, enterprises looking to adopt open-source AI technology would need to make their system highly efficient.

## Inference

Inference is one of the hot topics currently with LLMs in general. Large models like ChatGPT have very low latency and great performance but become more expensive with more usage.

On the flip side, open-source models like [Llama-2](https://registry.premai.io/detail.html?service=llama-2-7b) or [Falcon](https://registry.premai.io/detail.html?service=falcon-7b-instruct) have variants that are much smaller in size, yet it's difficult to match the latency and throughput that ChatGPT provides, while still being cost efficient. {cite}`cursor-llama`

Models that are run using Hugging Face pipelines do not have the necessary optimizations to run in a production environment. The open-source LLM inferencing market is still evolving so currently there's no silver bullet that can run any open-source LLM at blazing-fast speeds.

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
* To do these massive calculations in a timely manner, GPUs are required to help speed up the process. GPUs have much more memory bandwidth and processing power compared to a CPU which is why they are in such high demand when it comes to running large language models.

## LLM Inference Optimizers

The previous section explained why LLM inferencing is so difficult. In this section we'll look at some  open-source optimizers that can help make inferencing faster and easier.

### 1. Nvidia Triton Inference Server

```{figure-md} mlops-engines-triton-architecture
:class: caption
![](https://static.premai.io/book/mlops-engines-triton-architecture.png)

[Nvidia Triton Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/jetson.html)
```

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

```{figure-md} tgi-architecture
:class: caption
![](https://static.premai.io/book/mlops-engines-tgi-architecture.png)

[Text Generation Inference Architecture](https://github.com/huggingface/text-generation-inference)
```

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
* Not all LLM models are supported


Many other open-source projects like [BentoML](https://www.bentoml.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Flask](https://flask.palletsprojects.com/en/2.3.x/) have been used for serving models in the past. These frameworks work just fine for traditional ML related tasks, but fall behind when it comes to generative AI. The reason for this is that traditional ML serving solutions don't come with the necessary optimizations(tensor parallelism, continuous batching, flash attention, etc.)  to run generative AI models in production. 

There is ongoing development in both the open-source and private sectors to improve the performance of LLMs. It's up to the community to test out different services to see which one works best for their use case.

## Future

Due to the challenge of running LLMs, enterprises will likely opt to use an inference server instead of containerizing the model in-house. Optimizing LLMs for inference requires a high level of expertise, which most companies many not have. Inference servers can help solve this problem by providing a simple and unified interface to deploy AI models at scale, while still being cost effective.

Another pattern that's emerging is that models will move to the data instead of the data moving to the model. Currently, when calling the ChatGPT API data is sent to the model. Enterprises have worked very hard over the past decade to set up robust data infrastructure in the cloud. It makes a lot more sense to bring the model into the same cloud environment where the data is. This is where open-source models being cloud agnostic can have a huge advantage.

## Conclusion:

Before the word "MLOps" was coined, data scientists would manually train and run their models locally. At that time, data scientists were mostly experimenting with smaller statistical models. When they tried to bring this technology into production, they ran into many problems around data storage, data processing, model training, model deployment, and model monitoring. Companies started addressing these challenges and came up with a solution for running AI in production, hence "MLOps". 

Currently, we are in the experimental stage with LLMs. When companies try to use this technology in production, they will encounter a new set of challenges. Building solutions to address these challenges will build on the existing concept of MLOps. 
