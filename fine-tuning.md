# Fine-tuning

```{admonition} Work in Progress
:class: attention
This chapter is still being written & reviewed. Please do post links & discussion in the {{
  '[<i class="fas fa-pencil-alt"></i> comments]({}-comments)'.format(env.docname) }} below, or {{
  '[<i class="fab fa-github"></i> open a pull request]({}/edit/main/{}.md)'.format(
  env.config.html_theme_options.repository_url, env.docname)
}}!

Some ideas:

- https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e#training-your-own
- [Why You (Probably) Don't Need to Fine-tune an LLM](https://www.tidepool.so/2023/08/17/why-you-probably-dont-need-to-fine-tune-an-llm) (instead, use few-shot prompting & retrieval-augmented generation)
- [Fine-Tuning Llama-2: A Comprehensive Case Study for Tailoring Models to Unique Applications](https://www.anyscale.com/blog/fine-tuning-llama-2-a-comprehensive-case-study-for-tailoring-models-to-unique-applications) (fine-tuning LLaMA-2 for 3 real-world use cases)
- [Private, local, open source LLMs](https://python.langchain.com/docs/guides/local_llms)
- [Easy-to-use LLM fine-tuning framework (LLaMA-2, BLOOM, Falcon, Baichuan, Qwen, ChatGLM2)](https://github.com/hiyouga/LLaMA-Efficient-Tuning)
- https://dstack.ai/examples/finetuning-llama-2
- https://github.com/h2oai, etc.
- [The History of Open-Source LLMs: Better Base Models (part 2)](https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-better) (LLaMA, MPT, Falcon, LLaMA-2)
```

Building deep learning models completely from scratch takes a lot of time, money, data, and expertise. Creating a brand new model from the ground up for every single use case is not a scalable design. In order to customize a model without having to build it from scratch, ML practitioners have turned to fine-tuning.

Fine-tuning is the process of taking a pre-trained machine learning model and customizing it for a specific downstream task. The key reason to fine-tune is to take advantage of [transfer learning](https://www.v7labs.com/blog/transfer-learning-guide). The model has already learned useful feature representations during its initial pre-training, and fine-tuning allows you to leverage those features for a new task instead of having to learn them from scratch. This produces better performance with fewer training examples and resources.

## How Fine-Tuning Works

1. Start with a pre-trained model that has been trained on a large generic dataset.
2. Take this pre-trained model and add a new task-specific layer/head on top. For example, adding a classification layer for a sentiment analysis task.
3. Freeze the weights of the pre-trained layers so they remain fixed during training. This retains all the original knowledge.
4. Only train the weights of the new task-specific layer you added, leaving the pre-trained weights frozen. This allows the model to adapt specifically for your new task.
5. Train on your new downstream dataset by passing batches of data through the model architecture and comparing outputs to true labels.
6. After some epochs of training only the task layer, you can optionally unfreeze some of the pre-trained layers weights to allow further tuning on your dataset.
7. Continue training the model until the task layer and selected pre-trained layers converge on optimal weights for your dataset.

The key is that most of the original model weights remain fixed during training. Only a small portion of weights are updated to customize the model to new data. This transfers general knowledge while adding task-specific tuning.

## Fine-Tuning LLMs

When an LLM does not produce the desired output, engineers think that by fine-tuning the model, they can make it "better". But what exactly does "better" mean in this case? It's important to identify the root of the problem before fine-tuning the model on a new dataset.

Common LLM issues include:

- The model lacks knowledge on certain topics
  + {term}`RAG` can be used to solve this problem
- The model's responses do not have the proper style or structure the user is looking for
  + Fine-tuning or few-shot prompting is applicable here

```{figure-md} llm-fine-tuning-architecture
:class: caption
![](https://static.premai.io/book/fine-tuning-llm.png)

[Fine-Tuning LLMs](https://neo4j.com/developer-blog/fine-tuning-retrieval-augmented-generation)
```

A baseline LLM model cannot answer questions about content is hasn't been trained on {cite}`tidepool-citation`. The LLM will make something up, i.e., hallucinate. To fix issues like this, RAG is a good tool to use because it provides the LLM with the context it needs to answer the question.

On the other hand, if the LLM needs to generate accurate SQL queries, RAG is not going to be of much help here. The format of the generated output matter a lot, so fine-tuning would be more useful for this use case.

Here are some examples of models that have been fine-tuned to generate content in a specific format/style:

* [Gorilla LLM](https://gorilla.cs.berkeley.edu) - This LLM was fine-tuned to generate API calls.
* [Llama-2 chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) - The "chat" version of Llama is fine-tuned on conversational data.
* [Code Llama](https://about.fb.com/news/2023/08/code-llama-ai-for-coding) - A fine-tuned Llama 2 model designed for code generation.

## RAG(Retrieval Augmented Generation)

RAG is a method used to boost the accuracy of LLMs by injecting relevant context into an LLM prompt. It works by connecting to a vector database and fetches only the information that is most relevant to the user's query. Using this technique, the LLM is provided with enough background knowledge to adequately answer the user's question without hallucinating. 

RAG is not a part of fine-tuning, because it uses a pre-trained LLM and does not modify it in any way.
However, there are several advantages to using RAG:
- **Boost model accuracy**
  - Leads to less hallucinations by providing the right context
- **Less computing power required**
  - Unlike fine-tuning, RAG does not need to re-train any part of the model. It's only the models prompt that changes.
- **Quick and easy setup**
  - RAG does not require much domain expertise about LLMs. You don't need to find training data or corresponding labels. Most pieces of text can be uploaded into the vector db as is, without major modifications.
- **Connect to private data**
  - Using RAG, engineers can connect data from SaaS apps such as Notion, Google Drive, Hubspot, Zendesk, etc. to their LLM. Now the LLM has access to private data and can help answer questions about the data in these applications.

RAG plays a key role in making LLMs for useful, but it can be a bit tedious to set up. Tthere are a number of open-source project such as [Llama index](https://github.com/jerryjliu/llama_index) which can help make the process a bit easier.


## Fine-Tuning Image Models

Fine tuning computer vision based models is a common practice and is used in applications involving object detection, object classification, and image segmentation.

For these non-generative AI use-cases, a baseline model like Resnet or YOLO is fine-tuned on labeled data to detect a new object. Although the baseline model isn't initially trained for the new object, it has learned the feature representation. Fine-tuning enables the model to rapidly acquire the features for the new object without starting from scratch.

Data preparation plays a big role in the fine-tuning process for vision based models. An image of the same object can be taken from multiple angles, different lighting conditions, different backgrounds, etc. In order to build a robust dataset for fine-tuning, all of these image variations should be taken into consideration.

### Fine-Tuning AI image generation models

```{figure-md} image-generation-fine-tuning
:class: caption
![](https://static.premai.io/book/fine-tuning-image-generation.png)

[Dreambooth Image Generation Fine-Tuning](https://dreambooth.github.io)
```

Models such as [Stable Diffusion](https://stability.ai/stable-diffusion) can also be tailored through fine-tuning to generate specific images. For instance, by supplying Stable Diffusion with a dataset of pet pictures and fine-tuning it, the model becomes capable of generating images of that particular pet in diverse styles.

The dataset for fine-tuning an image generation model needs to contain two things:

- **Text**: What is the object in the image
- **Image**: The picture itself

The text prompts describe the content of each image. During fine-tuning, the text prompt is passed into the text encoder portion of Stable Diffusion while the image is fed into the image encoder. The model learns to generate images that match the textual description based on this text-image pairing in the dataset {cite}`octoml-fine-tuning`.

## Fine-Tuning Audio Models

```{figure-md} audio-fine-tuning
:class: caption
![](https://static.premai.io/book/fine-tuning-audio.png)

[Audio Generation Fine-Tuning](https://aws.amazon.com/blogs/machine-learning/fine-tune-and-deploy-a-wav2vec2-model-for-speech-recognition-with-hugging-face-and-amazon-sagemaker)
```

Speech-to-text models like [Whisper](https://registry.premai.io/detail.html) can also be fine-tuned. Similar to fine-tuning image generation models, speech-to-text models need two pieces of data:

1. **Audio recording**
2. **Audio transcription**

Preparing a robust dataset is key to building a fine-tuned model. For audio related data there are a few things to consider:

**Acoustic Conditions:**

* Background noise levels - more noise makes transcription more difficult. Models may need enhanced noise robustness.
* Sound quality - higher quality audio with clear speech is easier to transcribe. Low bitrate audio is challenging.
* Speaker accents and voice types - diversity of speakers in training data helps generalize.
* Audio domains - each domain like meetings, call centers, videos, etc. has unique acoustics.

**Dataset Creation:**

* Quantity of training examples - more audio-transcript pairs improves accuracy but takes effort.
* Data collection methods - transcription services, scraping, in-house recording. Quality varies.
* Transcript accuracy - high precision transcripts are essential. Poor transcripts degrade fine-tuning.
* Data augmentation - random noise, speed, pitch changes makes model robust.

## Importance of data

```{figure-md} data-centric-ai
:class: caption
![](https://static.premai.io/book/fine-tuning-data-centric.png)

[Data centric AI](https://segments.ai/blog/wandb-integration)
```

The performance of a fine-tuned model largely depends on the **quality** and **quantity** of training data.

For LLMs, the quantity of data can be an important factor when deciding whether to fine-tune or not. There have been many success stories of companies like [Bloomberg](https://arxiv.org/abs/2303.17564), [Mckinsey](https://www.mckinsey.com/about-us/new-at-mckinsey-blog/meet-lilli-our-generative-ai-tool), and [Moveworks](https://www.moveworks.com/insights/moveworks-enterprise-llm-benchmark-evaluates-large-language-models-for-business-applications) that have either created their own LLM or fine-tuned an existing LLM which has better performance than ChatGPT on certain tasks. However, tens of thousands of data points were required in order to make these successful AI bots and assistants. In the [Moveworks blog post](https://www.moveworks.com/insights/moveworks-enterprise-llm-benchmark-evaluates-large-language-models-for-business-applications), the fine-tuned model which surpasses the performance of GPT-4 on certain tasks, was trained on an internal dataset consisting of 70K instructions.

In the case of computer vision models, data quality can play a significant role in the performance of the model. Andrew Ng, a prominent researcher and entrepreneur in the field of AI, has been an advocate of data centric AI in which the quality of the data is more important than the sheer volume of data {cite}`small-data-tds`.

To summarize, fine-tuning requires a balance between having a large dataset and having a high quality dataset. The higher the data quality, the higher the chance of increasing the model's performance.

```{table} Estimates of minimum fine-tuning Hardware & Data requirements
:name: memory-data-requirements

Model | Task | Hardware | Data
------|------|----------|-----
Llama-2 7B | Text Generation | GPU: 65GB, 4-bit quantised: 10GB | 1K datapoints
Falcon 40B | Text Generation | GPU: 400GB, 4-bit quantised: 50GB | 50K datapoints
Stable Diffusion | Image Generation | GPU: 6GB | 10 (using Dreambooth) images
YOLO | Object Detection | Can be fine-tuned on CPU | 100 images
Whisper | Audio Transcription | GPU: 5GB (medium), 10GB (large) | 50 hours
```

```{admonition} GPU memory for fine-tuning
:name: memory-requirements
:class: note

Most models require a GPU for fine-tuning. To approximate the amount of GPU memory required, the general rule is around 2.5 times the model size. Note that {term}`quantisation` to reduce the size tends to only be useful for inference, not training-fine-tuning. An alternative is to only fine-tune some layers (freezing and quantising the rest), thus greatly reducing memory requirements.

For example: to fine-tune a `float32` (i.e. 4-byte) 7B parameter model:

$$
7 \times 10^{9}~\mathrm{params} \times 4~\mathrm{B/param} \times 2.5 = 70~\mathrm{GB}
$$
```

## Future

Fine-tuning models has been a common practice for ML engineers. It allows engineers to quickly build domain specific models without having to design the neural network from scratch.

Developer tools for fine-tuning continue to improve the overall experience of creating one of these models while reducing the time to market. Companies like [Hugging Face](https://huggingface.co/docs/transformers/training) are building open-source tools to make fine-tuning easy. On the commercial side, companies like [Roboflow](https://roboflow.com) and [Scale AI](https://scale.com/generative-ai-platform) provide platforms for teams to manage the full life-cycle of a model.

Overall, fine-tuning has become a crucial technique for adapting large pre-trained AI models to custom datasets and use cases. While the specific implementation details vary across modalities, the core principles are similar - leverage a model pre-trained on vast data, freeze most parameters, add a small tunable component customized for your dataset, and update some weights to adapt the model.

When applied correctly, fine-tuning enables practitioners to build real-world solutions using leading large AI models.

{{ comments }}
