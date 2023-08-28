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

## The MLOps Lifecycle

![](https://static.premai.io/book/mlops-engines-LLMOps-diagram.jpg)

[MLOps(Machine Learning Operations)](https://blogs.nvidia.com/blog/2020/09/03/what-is-mlops/) is a set of best practices for companies to run their AI in production. There are several components in the MLOps lifecycle, with each component trying to address problems with using AI in an enterprise setting. 

With large language models, the traditional MLOps landscape shifts a little bit and we encounter new problems. While MLOps focuses on model training LLMOps focuses on fine-tuning. Model inference is also integral to both lifecycles and will be a focus of this chapter. 

For this chapter, we'll be exploring the various open-source runtime engines for LLMs and the potential challenges with running these models in production.

## Challenges With Open-Source MLOps

MLOps has always been available in two flavors. One is the managed version, where all the components are provided out of the box for a steep price. The other is a DIY setup where you stitch together various open-source components. 

With large language models, the story is no different. Companies like Hugging Face are pushing for open-source models and datasets whereas closed-source competitors like OpenAI and Anthropic are doing the exact opposite. The three main challenges with open-source MLOps are maintenance, performance, and cost.

![](https://static.premai.io/book/mlops-engines-table.jpg)

1. <strong>Maintenance</strong>

When you use open-source components you have to do everything yourself. Whether is downloading the model, fine-tuning, evaluating, or inferencing, everything has to be done manually. When you have multiple open-source components companies tend to write "glue" code to connect the components together.

When a component is updated, the glue code connecting the component also needs to be updated. For example, if there is a new version of [BentoML](https://www.bentoml.com/) that your company is using you need to update the code that connects BentoML with [MLflow](https://mlflow.org/). Whenever an open-source component changes, everything the component is connected to could break. 

Because of this, teams have to stay on their toes to quickly fix their code to avoid breaking changes. In the long run with robust and scalable pipelines, this becomes less of an issue, but in the early stages, there is a lot of firefighting for developers to do.

2. <strong>Performance</strong>

One of the key differences between open-source and closed-source software is the level of optimization. Closed-source tools almost always have greater optimization whether it's a data pipeline, machine learning model, or runtime environment. 

With LLMs, the baseline difference between open/closed source is quite stark. OpenAI and Anthropic's models are not only larger but also run faster than open-source models. Despite what various benchmarks claim to say, a model like ChatGPT has higher quality output than even the best open-source model on the market. 

On top of that, OpenAI's inferencing is exponentially faster compared to a model provided by Hugging Face. The reason is optimization. Closed-source LLM providers have optimized every step of their pipeline from data collection all the way to model serving. With open-source NONE of those optimizations exist out of the box. It's up to companies to experiment with various open-source tools to figure out which one gives them the best performance for the problem they are solving.

3. <strong>Cost</strong>

One of the reasons companies prefer to choose an open-source solution is for cost savings. If done correctly, the savings can be huge in the long run. However, many firms underestimate the amount of work required to make an open-source ecosystem work seamlessly. 

Oftentimes, teams have to pay a larger cost upfront when working with open-source LLMs. Infrastructure is expensive and difficult to maintain. Along with that, teams are constantly experimenting since the technology is so new, which further adds to the cost. 

Due to more maintenance and decreased baseline performance, enterprises looking to adopt open-source LLM technology will need to make their system highly efficient.

## Let's talk about inference

Inference is one of the hot topics currently with LLMs in general. Large models like ChatGPT have very low latency and great performance but are extremely expensive to run. 

On the flip side, open-source models like [Llama-2](https://registry.premai.io/detail.html?service=llama-2-7b) or [Falcon](https://registry.premai.io/detail.html?service=falcon-7b-instruct) have variants that are much smaller in size, yet they cannot match the latency and throughput that ChatGPT provides. In terms of performance metrics at a system level, the main difference between closed and open-source models is optimization.

Models that are hosted on Hugging Face do not have the necessary optimizations to run in a production environment. The open-source LLM inferencing market is still evolving so currently there's no silver bullet that can run any open-source LLM at blazing-fast speeds.

## Why is inferencing so challenging to begin with?

There are a plethora of reasons why inferencing is slow. As models grow in size and neural networks become more complex it's no surprise that it's taking longer to get an output. 

On top of that, the programming language of choice for AI, Python, is inherently slow compared to compiled languages like C++. The developer-friendly syntax and vast array of libraries have put Python in the spotlight, but when it comes to sheer performance it falls behind many other languages.

To compensate for its performance many inferencing servers convert the Python code into an optimized module. For example, Nvidia's [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) can take a Pytorch model and compile it into [TensorRT](https://developer.nvidia.com/tensorrt-getting-started), which has a much higher performance than native Pytorch.

Similarly, [Llama.cpp](https://github.com/ggerganov/llama.cpp) optimizes the Llama inference code to run in raw C++. Using this optimization, people can run a large language model on their laptops without a dedicated GPU.

Another reason inferencing is challenging is due to the size of the data. Not only do LLMs have billions of parameters, but they perform millions of mathematical calculations for each inference.

To do these massive calculations in a timely manner, GPUs are required to help speed up the process. GPUs have much more memory bandwidth and processing power compared to a CPU. This is why GPUs are in such high demand when it comes to running large language models.

## LLM Inference Optimizers

Now that we've covered why LLM inferencing is so difficult, let's take a look at some of the open-source optimizers that can help make inferencing faster and easier.

1. <strong>Nvidia Triton Inference Server</strong>

![](https://static.premai.io/book/mlops-engines-triton-architecture.png)
[Image Source](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/jetson.html)

This [inference server](https://developer.nvidia.com/triton-inference-server) offers support for multiple model formats such as Pytorch, Tensorflow, ONNX, TensorRT, etc. It uses GPUs efficiently to boost the performance of deep learning models.
* <strong>Concurrent model execution:</strong> This allows multiple models to be executed on 1 or many GPUs in parallel. Multiple requests are routed to each model to execute the tasks in parallel
* <strong>Dynamic Batching:</strong> Combines multiple inference requests into a batch to increase throughput. Requests in each batch can be processed in parallel instead of handling each request sequentially.

Pros:
* High throughput, low latency for serving LLMs on a GPU
* Supports multiple frameworks/backends
* Production level performance

Cons:
* Difficult to set up
* Not compatible with many of the newer LLMs

2. <strong>Text Generation Inference</strong>

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

3. <strong>[vLLM](https://vllm.ai/)</strong>

This is an open-source project created by researchers at Berkeley to improve the performance of LLM inferencing. vLLM primarily optimizes LLM throughput via methods like PagedAttention and Continuous Batching. The project is fairly new and there is ongoing development.

Pros:
* Can be used commercially
* Supports many popular Hugging Face models
* Easy to setup

Cons:
* Takes a while for new LLMs to be supported


Many other open-source projects like [BentoML](https://www.bentoml.com/), [FastAPI](https://fastapi.tiangolo.com/), and [Flask](https://flask.palletsprojects.com/en/2.3.x/) have been used for serving models in the past. The reason I have not included them on this list is that these open-source tools do not provide the optimizations you need to run LLMs in production.

LLM inference is quite different from ML inference in the past. These models are much larger and require an extraordinary amount of computing power. 

To meet these requirements, there is ongoing development in both the open-source and private sectors to improve the performance of LLMs. At the moment, there is no clear winner that can optimize inference for any LLM. 

It's up to the community to test out different services to see which one works best for their use case.

## Some Thoughts About The Future

Due to the challenge of running LLMs, enterprises will opt to use an inference server instead of containerizing the model in-house. Most companies don't have the expertise to optimize these models, but they still want the performance benefits. Inference servers, whether they are open-source or not, will be the path forward.

Another pattern that's emerging is that models will move to the data instead of the data moving to the model. Right now if you call the ChatGPT API, you would be sending your data to the model. Enterprises have worked very hard over the past decade to set up robust data infrastructure in the cloud. It makes a lot more sense to bring the model into the same cloud environment where the data is. This is where open-source models being cloud agnostic have a huge advantage.

## Conclusion:

Before the word "MLOps" was coined, data scientists would manually train and run their models locally. At that time, data scientists were mostly experimenting with smaller statistical models. When they tried to bring this technology into production, they ran into many problems around data storage, data processing, model training, model deployment, and model monitoring. Companies started addressing these challenges and came up with a solution for running AI in production, hence "MLOps". 

Currently, we are in the experimental stage with LLMs. When companies try to use this technology in production, they will encounter a new set of challenges. Building solutions to address these challenges will build on the existing concept of MLOps. 