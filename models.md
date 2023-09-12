# Models

% LLaMA 1 vs 2, Falcon, Stable Diffusion, DALL-E

The emergence of Large Language Models, notably with the advent of GPT-3, ChatGPT, Midjourney, [Whisper](https://openai.com/research/whisper) helped bloom a new era. Beyond revolutionizing just language models, these models also pushed innovation in other domains like Vision (ViT, DALL-E, Stable Diffusion etc), Audio (Wave2vec, Bark) or even Multimodal models.

```{figure} https://static.premai.io/book/models_llms-landscape.png
---
width: 90%
name: llms landscape
---
[Page 7, A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf)
```

% TODO: maybe make the above model names as references and not names directly.

Before looking into the Open source models timeline, let's take a look at current proprietary competitors.

## Proprietary Models
wip
% GPT-4
% chatgpt
% Claude 1, 2
% Midjourney


## Rise of Open-Source Models
% TODO: tell about how emergent of openai models created an increase in OSS models, even though there were primitive versions of that available, but now present with rlhf etc, why it helps/had a rise to cater the specific use cases which generic openai models couldn't capture

ChatGPT would be playing a huge role if it was a story of LLMs and how they fastracked their improvements.
Early high performing LLMs were proprietary, accessible only through organisations' paid APIs, hindering transparency and raising concerns about data privacy, bias, alignment and robustness, giving limited possibilities to cater domain-specific use cases without letting RLHF'ed alignment intefere.

Recognizing the need for openness, the LLM research community responded by creating open-source variants, laying the foundation for increased transparency and the development of more powerful models.
% TODO: ^^add refs

## Catching Up with Close-Source Models
% ChatGPT, Midjourney and Others
% TODO: talking about what OSS community did to catchup on chatgpt performance i.e newer versions came up of models (e.g llama, llama v2, SD, SD XL)

Before [ChatGPT](https://openai.com/blog/chatgpt)'s (GPT-3.5) public release we had [GPT-3](https://en.wikipedia.org/wiki/GPT-3) being one of the "[best](https://www.reddit.com/r/MachineLearning/comments/ydwi6c/d_whats_the_best_open_source_model_for_gpt3like/)" Base Language Model which released ~2.1 years before ChatGPT. And following that we've had LLMs like [Bard](https://blog.google/technology/ai/bard-google-ai-search-updates/), [Claude](https://www.anthropic.com/index/introducing-claude), [GPT-4](https://openai.com/research/gpt-4) and [others](https://lmsys.org/blog/2023-05-25-leaderboard/).


### Initial steps
There has been a few visible marks across modalities of AI models, highly catalysing growth of open source:
- [Meta AI launches LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/), open sourcing the code and not the weights.
- [StabilityAI released Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement).


#### Stable Diffusion

Stable Diffusion is a [latent text-to-image diffusion model](https://arxiv.org/abs/2112.10752). Created by [Stability AI](https://stability.ai/) and support from [LAION](https://laion.ai/), where they used 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database for training. Similar to Google's [Imagen](https://arxiv.org/abs/2205.11487), this model uses a frozen [CLIP ViT-L/14](https://arxiv.org/abs/2103.00020) text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.

##### Uniqueness
While [training](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md#training):
- Text prompts are encoded through a ViT-L/14 text-encoder
- UNet backbone of the latent diffusion model takes non-pooled output of the text encoder via cross-attention.
- Loss is reconstruction objective between prediction made by UNet and noise added to the latent.

##### [Limitations](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md#limitations-and-bias)
- The model does not achieve perfect photorealism, or render legible text and performs poorly on difficult prompt like "A blue cube on top of a red sphere".
- The model was trained mainly with English captions.
- No measures were used to deduplicate the dataset before usage.


#### LLaMA
Under LLaMA, Meta released a collection of foundation language models ranging from 7B to 65B parameters, pre-trained over a corpus containing more than 1.4 trillion tokens. It was designed to be versatile and applicable for many different use cases, and possibly fine-tuned for domain specific tasks if required.

It showed **better performance** across domains compared to its competitors.

```{figure} https://static.premai.io/book/models_llama-scores.png
---
width: 88%
name: llama scores
---
[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
```

LLaMA-13B outperforms GPT-3 (175B) on most benchmarks while being more than 10x smaller, and LLaMA-65B is competitive with models like Chinchilla-70B and PaLM-540B. LLaMA-65B performs similarly to the closed-source GPT-3.5 on the MMLU and GSM8K benchmarks (TODO: validate and reference)

##### Uniqueness

There are few key inspirations LLaMA architecture took from other LLMs:
- **[Pre-normalization](https://arxiv.org/abs/1910.07467) (GPT-3):** using RMSNorm to normalize transformer sub-layer inputs.
- **[SwiGLU activation function](https://arxiv.org/abs/2002.05202) (PaLM):** replacing ReLU with SwiGLU.
- **[Rotary Embeddings](https://arxiv.org/abs/2104.09864) (GPTNew):** replacing absolute positional embeddings with Rotary positional embeddings.

##### Limitations

- It was released under a noncommercial license focused on usage for research use cases only.
- LLaMA is a foundation model and not fine-tuned for specific tasks, which may limit its performance on certain tasks
- LLaMA seemed not as competitive as other models on certain benchmarks, such as BoolQ and WinoGrande.

Interestingly within a week from LLaMA's launch, its [weights were leaked to the public](https://www.vice.com/en/article/xgwqgw/facebooks-powerful-large-language-model-leaks-online-4chan-llama). This created a huge impact on the community for all kinds innovations coming up, eventhough there was still license restrictions not permitting commercial usage.

### Pacing Up

After 2 weeks from the LLaMa weights leak, Stanford [releases Alpaca 7B](https://crfm.stanford.edu/2023/03/13/alpaca.html).

#### Alpaca 7B

It's a 7B parameter model fine-tuned from LLaMA 7B model on 52K instruction-following datapoints. It performs qualitatively similarly to OpenAI's text-davinci-003 while being smaller and cheaper to reproduce i.e taking only < \$600. Github repository [here](https://github.com/tatsu-lab/stanford_alpaca).

```{figure} https://static.premai.io/book/models_alpaca-finetuning.png
---
width: 80%
name: alpaca finetuning
---
[Alpaca 7B fine-tuning strategy](https://crfm.stanford.edu/2023/03/13/alpaca.html)
```
##### Uniqueness

- Unique Data Source: Alpaca 7B is distinct for being fine-tuned from LLaMA 7B using 52K instruction-following demonstrations ([coming from self-instruct paper](https://arxiv.org/abs/2212.10560)) in the style of text-davinci-003, enabling research into instruction-following scenarios.
- Cost-Efficient Alternative: Alpaca 7B offers similar performance to text-davinci-003 but at a lower cost, making it accessible for academic research.


##### Limitations

- Non-commercial Usage: This limitation arises from the non-commercial license of LLaMA, upon which Alpaca is based.
- Quality: Alpaca 7B may occasionally produce inaccurate information, including hallucinations, misinformation, and toxic content.
- Evaluation Scope: While Alpaca performs well in some evaluations, its performance may vary in unexplored scenarios.

Right after that [alpaca-lora](https://github.com/tloen/alpaca-lora) came out, using low rank fine-tuning it made possible to reproduce Alpaca within hours on a single NVIDIA RTX 4090 GPU with inference being possible even [on a Raspberry PI](https://twitter.com/miolini/status/1634982361757790209).


Things moved fast from here when first promising inference speed was achieved without GPU for LLaMA using 4 bit quantisation by the [LLaMA GGML](https://github.com/ggerganov/llama.cpp). A new wave of [quantized models started coming from the comminity](https://huggingface.co/TheBloke).

In a day after, [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) came in.
#### Vicuna

[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) was released under a joint effort by UC Berkeley, CMU, Stanford, UC San Diego, and MBZUAI. It was trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. GPT-4 was used for its evaluation. They released a [demo](https://chat.lmsys.org/) and [code](https://github.com/lm-sys/FastChat), [weights](https://github.com/lm-sys/FastChat#vicuna-weights) under non-commercial license following LLaMa.

```{figure} https://static.premai.io/book/models_vicuna-finetuning.png
---
width: 80%
name: vicuna finetuning
---
[Vicuna fine-tuning strategy](https://lmsys.org/blog/2023-03-30-vicuna/#overview)
```
##### Uniqueness

- Impressive Quality: Vicuna-13B achieved over 90% quality compared to ChatGPT and Google Bard, surpassing other models like LLaMA and Stanford Alpaca in more than 90% of cases.
- For training:
  - Training loss was adjusted to account for multi-turn conversations and compute the fine-tuning loss solely on the chatbot's output.
  - Expanded max context length from 512 in Alpaca to 2048, [gradient checkpointing](https://arxiv.org/abs/1604.06174) and [flash attention](https://arxiv.org/abs/2205.14135) utilisation helping handle memory pressure.
  - Used [SkyPilot](https://github.com/skypilot-org/skypilot) [managed spot](https://skypilot.readthedocs.io/en/latest/examples/spot-jobs.html) to reduce the cost for training the 7B model from \$500 to around \$140 and the 13B model from around \$1k to \$300.
- Cost-Efficiency: The cost of training was around \$300, making it a cost-effective choice for research purposes.
- Enhanced Dataset: Vicuna is fine-tuned using 70K user-shared ChatGPT conversations from [ShareGPT](https://sharegpt.com/), enabling it to provide detailed and well-structured answers, with performance on par with ChatGPT.

##### Limitations

- Reasoning and Safety: Vicuna may struggle with tasks involving reasoning or mathematics and may not always ensure factual accuracy. It has not been fully optimised for safety or to mitigate potential toxicity or bias.
- Evaluation Framework: The proposed evaluation framework, based on GPT-4, is not yet a rigorous or mature approach, as large language models can sometimes produce hallucinated responses.
- No Dataset release.
- Non-commercial usage only following the LLaMA model's license, OpenAI's [data terms](https://openai.com/policies/terms-of-use) and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT.

After the release they also conducted a [deeper study on GPT4-based evaluation approach](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#llm-judge).


Then came in updates like [LLaMa-Adapter](https://arxiv.org/pdf/2303.16199.pdf), [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/) and in less than a month [Open Assistant](https://open-assistant.io/) launches a model and a [dataset for Alignment via RLHF](https://arxiv.org/abs/2304.07327).


Overall the LLaMA variants landscape looked somewhat like this, even though it doesn't show all the variants:
```{figure} https://static.premai.io/book/models_llama-variants.png
---
width: 80%
name: llama variants tree
---
[Page 10, A Survey of Large Language Models](https://arxiv.org/pdf/2303.18223.pdf)
```

After a month, the community did an open reproduction of LLaMA, named [OpenLLaMA](https://github.com/openlm-research/open_llama).

#### OpenLLaMA

Students at UC Berkeley started [OpenLM Research group](https://huggingface.co/openlm-research) through which they trained in collaboration with [Stability AI](https://stability.ai/) to release [OpenLLaMA](https://github.com/openlm-research/open_llama) v1, a permissively licensed open source reporduction of Meta AI's LLaMA. They released a series of 3B, 7B and 13B models trained on [different mix of datasets](https://huggingface.co/openlm-research). And the weights released can serve as drop in replacement of LLaMA.


##### Uniqueness

- Model is trained on open sourced [RedPajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) by [Together](https://huggingface.co/togethercomputer).
- All steps for training are kept same as mentioned in [LLaMA paper](https://arxiv.org/abs/2302.13971).
- Model is trained on 1T tokens.
- Weights released under Apache 2.0 license, in two formats:
  - EasyLM format to be use with [EasyLM framework](https://github.com/young-geng/EasyLM)
  - PyTorch format to be used with the [Hugging Face transformers library](https://huggingface.co/docs/transformers/index)

##### Limitations

- Dataset Difference: OpenLLaMA uses open datasets instead of the original LLaMA dataset. While training procedures, architecture, and other parameters remain the same, there may be differences in performance on certain tasks.


Around same time [MosaicML](https://www.databricks.com/company/newsroom/press-releases/databricks-completes-acquisition-mosaicml) released its [MPT](https://github.com/mosaicml/llm-foundry) models series, and [TII](https://www.tii.ae/) also released [Falcon models](https://www.tii.ae/news/uaes-technology-innovation-institute-launches-open-source-falcon-40b-large-language-model).


#### MPT

MosaicML released [MPT (MosaicML Pretrained Transformer) models series](https://huggingface.co/mosaicml) consisting:
- 7B variants:
  - [MPT-7B Base](https://huggingface.co/mosaicml/mpt-7b)
  - [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct)
  - [MPT-7B-Chat](https://huggingface.co/mosaicml/mpt-7b-chat)
  - [MPT-7b-StoryWriter-65k+](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- 30B variants:
  - [MPT-30B Base](https://huggingface.co/mosaicml/mpt-30b)
  - [MPT-30B-Instruct](https://huggingface.co/mosaicml/mpt-30b-instruct)
  - [MPT-30B-Chat](https://huggingface.co/mosaicml/mpt-30b-chat)

##### Uniqueness

- Licensed for commercial usage (not all variants in the series): MPT-7B base, MPT-7B-StoryWriter-65k+, MPT-30B were only released under Apache-2.0 license.
- Uses [ALiBi](https://arxiv.org/abs/2108.12409) to handle long inputs till 84k tokens context size, whereas trained using upto 65k tokens context.
- Uses [FlashAttention](https://arxiv.org/abs/2205.14135) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) to optimize for fast training and inference.
- They also released an entire framework, the [MosaicML LLM Foundry](https://github.com/mosaicml/llm-foundry).

##### Limitations

- Not all variants were released under permissive commercial usage license.
- Combinations of open sourced datasets was used for training the models and [mentioned which ones with proportions](https://github.com/mosaicml/llm-foundry/issues/499#issuecomment-1662556022), but haven't released the [combined dataset yet](https://github.com/mosaicml/llm-foundry/issues/499).

#### Falcon

[TII](https://falconllm.tii.ae/index.html) released [Falcon series of 40B, 7.5B and 1.3B parameters LLMs](https://falconllm.tii.ae/falcon.html), trained on their open sourced and curated RefinedWeb dataset. After the release it has dominated the [Huggingface's open llm leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for the State of the Art open sourced LLM for more than 2 months.

##### Uniqueness

- Falcon-40B has data from a variety of English, German, Spanish, French, Italian, Portuguese, Polish, Dutch, Romanian, Czech, and Swedish languages inserted into its pre-training set.
- They released all the model and its instruction tuned and chat variants under Apache 2.0 license, permitting commercial usage.
- The model uses only 75 percent of GPT-3’s training compute, 40 percent of Chinchilla AI’s, and 80 percent of PaLM-62B’s.
- Falcon 40B pre-training dataset contained around 5 Trillion tokens gathered from public web crawls (~80%), research papers, legal text, news, literature, and social media conversations.
  - Subset of this [dataset containing 600 Billion tokens](https://arxiv.org/abs/2306.01116) was open sourced.
- Model uses decoder-only architecture with [Flash Attention](https://arxiv.org/abs/2205.14135), [Multi-Query Attention](https://arxiv.org/abs/1911.02150), [Parallel Attention and Feed Forward](https://arxiv.org/abs/2305.13297).

##### Limitations

- Full dataset used for pre-training the 40B variant wasn't released.
- Falcon-40B is trained using a sequence length of 2K, which is smaller compared to MPT, XGen, but context size can be increased using [RoPE embeddings](https://arxiv.org/abs/2104.09864) withing model's architecture, allowing it to generalize to longer sequence lengths (might require some finetuning).
- A paper detailing Falcon models specifically has not yet been released.

### Narrowing the Gap

On 18th July, Meta AI released LLaMA-2, breaking most SOTA records on open sourced LLMs performances.


#### LLaMA-2

Meta AI [released LLaMA-2](https://github.com/facebookresearch/llama) with both pre-trained and fine-tuned variants for a series of 7B, 13B and 70B parameter sizes.

Some win rate graphs on LLaMA-2 after evaluation comparisons against popular LLMs where it roughly ties with GPT-3.5 and performs noticeably better than Falcon, MPT and Vicuna.
```{figure} https://static.premai.io/book/models_llama2-rates.png
---
width: 88%
name: llama-2 rates
---
[Page 3, LLaMA 2: Open Foundations and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)
```

##### Uniqueness

- LLaMA-2 models are pre-trained over 2 trillion tokens dataset in total, compared to 1.4 trillion tokens dataset for LLaMA-1.
- LLaMA-2 models are trained with a 4k context length, whereas it's 2k for LLaMA-1.
- Larger variants use grouped query attention (GQA) within their underlying architecture, helping improve inference efficiency.

    ```{figure} https://static.premai.io/book/models_llama2-gqa.png
    ---
    width: 80%
    name: llama-2 gqa
    ---
    [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf).
    ```
- LLaMA-2-70B became new state-of-the-art among open-source LLMs on all tasks considered.

    ```{figure} https://static.premai.io/book/models_llama2-opensource-scores.png
    ---
    width: 80%
    name: llama-2 open source llms scores
    ---
    [Page 8, LLaMA 2: Open Foundations and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)
    ```
- They released chat variants from base models using instruction tuning and high scale RLHF, also proposed a Ghost Attention (GAtt) which helps control dialogue flow over multiple turns.

    ```{figure} https://static.premai.io/book/models_llama2-workflow.png
    ---
    width: 80%
    name: llama-2 workflow
    ---
    [Page 5, LLaMA 2: Open Foundations and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)
    ```
- For Alignment uses a two-stage RLHF approach, starting with Rejection Sampling, then doing Rejection Sampling + Proximal Policy Optimization (PPO)
- All model variants under LLaMA-2 are released under [LLaMA-2 License](https://opensourceconnections.com/blog/2023/07/19/is-llama-2-open-source-no-and-perhaps-we-need-a-new-definition-of-open/), permitting commercial usage unless it's facing 700 million monthly active users then the entity must obtain a license from Meta.
- Meta's team does quite some work for mitigating AI safety issues in the model.
  - Released a [responsible Use Guide](https://github.com/facebookresearch/llama/blob/main/Responsible-Use-Guide.pdf).

##### Limitations

- LLaMA-2 base models perform worse compared to aligned proprietary models, but performs favourably when compared to popular base LLMs like [PaLM](https://arxiv.org/abs/2204.02311).

    ```{figure} https://static.premai.io/book/models_llama2-proprietary-scores.png
    ---
    width: 80%
    name: llama-2 proprietary llms scores
    ---
    [Page 8, LLaMA 2: Open Foundations and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf)
    ```
- Llama 2 Chat model variants can sometimes give overly cautious responses due to high safety tuning on the model.
- Reward models used in the model alignment steps aren't open sourced yet.


Till now we've mostly been looking at LLMs in general and not other models, let's look at the vision domain now.

#### Stable Diffusion XL
[StabilityAI released Stable Diffusion XL 1.0 (SDXL)](https://stability.ai/blog/stable-diffusion-sdxl-1-announcement) models on 26th July, being current State of the Art for text-to-image and image-to-image generation open sourced models. They released a [base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and a [refinement model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) which is used to improve the visual fidelity of samples generated by SDXL.

Few months back they released [Stable-diffusion-xl](https://arxiv.org/abs/2307.01952) [base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9) and [refinement](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9) models versioned as 0.9, where license permitting only research purpose usages.

SDXL consistently surpasses all previous versions of Stable Diffusion models by a significant margin:

  ```{figure} https://static.premai.io/book/models_sdxl-winrate.png
  ---
  width: 60%
  name: sdxl winrate
  ---
  [SDXL Winrate](https://stability.ai/blog/stable-diffusion-sdxl-1-announcement)
  ```


##### Uniqueness
- Works effectively on GPUs with 8GB or more VRAM.
- 3x larger UNet-backbone compared to previous Stable Diffusion models.
- Introduces a two-stage model process; the base model (can work standalone) generates an image as an input to the refiner model which adds additional high-quality details.
  ```{figure} https://static.premai.io/book/models_sdxl-arch.png
  ---
  width: 78%
  name: sdxl arch
  ---
  [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952.pdf)
  ```
- Proposed two additional model conditioning techniques to preserve training data from being discarded and gain more control over how a generated image should be cropped:
  - Conditioning the Model on [Image Size](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#size-conditioning).
  - Conditioning the Model on [Cropping Parameters](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#crop-conditioning).
- Commercial usage [allowed by SDXL License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/12#64c237c5f3977a70e19142ed).
- They also released a processed [TensorRT variant of SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt#stable-diffusion-xl-10-tensorrt), which can give upto [41% latency and 70% throughput improvements](https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt#performance-comparison).
- [Clipdrop](https://clipdrop.co/stable-diffusion) provides free SDXL inference.

##### Limitations

- For high quality generations from SDXL, a two-stage approach is required i.e using an additional refinement model, having to load two large models into memory hampers accessibility and sampling speed.
- Generations are sometimes poor when synthesizing intricate structures, such as human hands, or while rendering long legible text.
- Model achieves a remarkable level of realism in its generated images but does not attain perfect photorealism.
- Model’s training process heavily relies on large-scale datasets, possibly introducing social and racial biases.

In the domain of Image generation currently [Midjourney](https://www.midjourney.com/) is one of the most popular proprietary solutions for [simple users](https://www.reddit.com/r/StableDiffusion/comments/15i6tg3/are_we_killing_the_future_of_stable_diffusion/jusrar3/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button). StabilityAI's SDXL vs [Midjourney](https://www.midjourney.com/) comparison shows that it is on par with favourability.

  ```{figure} https://static.premai.io/book/models_sdxl-midjourney.png
  ---
  width: 88%
  name: sdxl x midjourney
  ---
  [Page 14, SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/pdf/2307.01952.pdf)
  ```
  ```{note} Above experiment is against Midjourney v5.1, whereas current latest is [Midjourney v5.2](https://docs.midjourney.com/docs/model-versions).
  ```


% code llama


% tiny llama

% falcon 180B

% persimonn 8B

% check here - https://sungkim11.medium.com/list-of-open-sourced-fine-tuned-large-language-models-llm-8d95a2e0dc76



% TODO: on how to write in order -
    % define few top closed source models (with limitations if possible)
    % https://crfm.stanford.edu/helm  - scores to show how open LLMs lagged behind closed ones
    % introduce LLaMa (with paper) by Meta - to show step 0 for catching up, by having a good base LLM. (also tell limitations)(cite the performance table comparison)
    % tell what was the requirement/recipe open source models were missing to catchup on close-source models
        % introduce instruction tuning and rlhf briefly which led to having "chat" models or tool calling models e.g gorilla model
    % (use this image to https://deci.ai/wp-content/uploads/2023/08/deci-blog-llms-1-1536x1033.png.webp) tell about LLaMa family models and define them (https://deci.ai/blog/list-of-large-language-models-in-open-source/)
    % tell about gpt-4
    % Introduce Llama 2 and variants
    % talk about Falcon 180B
    % slightly mention about coder models too (maybe) / comparing with openai codex? intro code-llama?

## Comparisons
% huggingface open llm leaderboard
% find model comparison tables:
   % - https://gpt4all.io (scroll down for performance benchmarks)
   % - https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard

## Future

% Human/GPT-4 evals
% RLHF vs RLAIF?

% also maybe check here in futures section - https://arxiv.org/pdf/2303.18223.pdf
## Read More
% optional


See also:

- "The History of Open-Source LLMs: Better Base Models (part 2)" (LLaMA, MPT, Falcon, LLaMA-2) https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-better
- "Papers I've read this week, Mixture of Experts edition" (conditional routing models) https://finbarrtimbers.substack.com/p/papers-ive-read-this-week-mixture
- "AI and Memory Wall" https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8
- https://github.com/imaurer/awesome-decentralized-llm
- https://github.com/huggingface/transformers/blob/main/awesome-transformers.md
- Background, Foundational Papers, Algos https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e
- {cite}`golden-age-os-end`

{{ comments }}
