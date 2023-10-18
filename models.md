# Models

```{admonition} Work in Progress
:class: attention
{{ wip_chapter }}

Some ideas:

- [The History of Open-Source LLMs: Better Base Models (part 2)](https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-better) (LLaMA, MPT, Falcon, LLaMA-2)
- [Papers I've read this week, Mixture of Experts edition](https://finbarrtimbers.substack.com/p/papers-ive-read-this-week-mixture) (conditional routing models)
- [AI and Memory Wall](https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8)
- https://github.com/imaurer/awesome-decentralized-llm
- https://github.com/huggingface/transformers/blob/main/awesome-transformers.md
- [Background, Foundational Papers, Algos](https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e)
- end of open source AI {cite}`golden-age-os-end`
- futures section in Survey of LLMs {cite}`zhao2023survey`
- Human/GPT-4 evals
- RLHF vs RLAIF?
```

The emergence of Large Language Models, notably with the advent of [GPT-3](https://openai.com/research/language-models-are-few-shot-learners), [](#chatgpt), [Midjourney](#midjourney), [Whisper](https://openai.com/research/whisper) helped bloom a new era. Beyond revolutionising just language models, these models also pushed innovation in other domains like Vision ([ViT](https://huggingface.co/docs/transformers/model_doc/vit), [DALL-E](https://openai.com/research/dall-e), [Stable Diffusion](#stable-diffusion) [SAM](https://segment-anything.com), etc), Audio Wave2vec {cite}`schneider2019wav2vec`, [Bark](https://registry.premai.io/detail.html?service=bark)) or even [Multimodal models](https://codi-gen.github.io).

```{figure} https://static.premai.io/book/models_llms-landscape.png
:width: 90%
:name: llms-landscape

Page 7, A Survey of Large Language Models {cite}`zhao2023survey`
```

## Proprietary Models

### Text

For performance comparisons, [](eval-datasets.md#chatbot-arena) helps (though it's a bit old and doesn't reflect latest results).

#### PaLM-2

[PaLM-2 is Google's next-generation large language model](https://blog.google/technology/ai/google-palm-2-ai-large-language-model), heavily trained on multilingual text, spanning more than 100 languages. PaLM 2 also excels at tasks like advanced reasoning, translation, and code generation. PaLM-2 is smaller than its predecessor, PaLM, but more efficient with overall better performance, including faster inference, fewer parameters to serve, and a [lower serving cost](https://ai.google/discover/palm2). PaLM-2 achieves results competitive with OpenAI's GPT-4, and it has been shown to outshine GPT-4 in [certain areas of reasoning](https://www.reddit.com/r/singularity/comments/13e1b5h/performance_of_gpt4_vs_palm_2). PaLM-2's multilingual capabilities enable it to understand idioms, riddles, and nuanced texts from [various languages](https://www.cnbc.com/2023/05/16/googles-palm-2-uses-nearly-five-times-more-text-data-than-predecessor.html). PaLM-2 also offers the advantage of quick responses, providing [three at a time](https://dataconomy.com/2023/07/18/best-large-language-models-llms). They also [released a technical paper](https://ai.google/static/documents/palm2techreport.pdf) for more details.

#### ChatGPT

[ChatGPT is a language model developed by OpenAI](https://openai.com/blog/chatgpt). It is fine-tuned from a model in the GPT-3.5 series and was trained on an Azure AI supercomputing infrastructure. ChatGPT is designed for conversational AI applications, such as chatbots and virtual assistants.

ChatGPT is sensitive to tweaks to the input phrasing or attempting the same prompt multiple times. It's still not fully reliable and can "hallucinate" facts and make reasoning errors.

#### GPT-4

[GPT-4 is a language model developed by OpenAI](https://openai.com/research/gpt-4). It is the successor to GPT-3 and has been made publicly available via the paid chatbot product ChatGPT Plus and via OpenAI's API. It is a large multimodal model that can accept image and text inputs and emit text outputs, [though multimodal capabilities aren't released to the public yet](https://analyticsindiamag.com/what-happened-to-multimodal-gpt-4). It exhibits human-level performance on various professional and academic benchmarks and can follow complex instructions in natural language and solve difficult problems with accuracy. It can handle input prompts of up to 32k tokens, which is a significant increase from GPT-3.5's 4k tokens. It can solve complex mathematical and scientific problems beyond the capabilities of GPT-3.5, such as advanced calculus problems or simulating chemical reactions [more effectively than its predecessor](https://www.searchenginejournal.com/gpt-4-vs-gpt-3-5/482463). It is more reliable, creative, and able to handle much more nuanced instructions than GPT-3.5.

Despite its capabilities, [GPT-4 still sometimes "hallucinates"](https://www.reddit.com/r/ChatGPT/comments/12fmrcd/examples_of_gpt4_hallucination) facts and makes reasoning errors.

#### Claude

[Claude 2 is a language model developed by Anthropic](https://www.anthropic.com/index/claude-2). It was announced on July 11, 2023 and has improved performance and longer responses compared to its predecessor [Claude](https://www.anthropic.com/index/introducing-claude), and can be accessed via API as well as a through [their website](https://claude.ai/login). According to Anthropic, users find Claude easy to converse with, clearly explains its thinking, is less likely to produce harmful outputs, and has a longer memory. Improvements have been made from previous models on coding, math, and reasoning.

### Audio

#### StableAudio

[StableAudio](https://stability.ai/stable-audio) is a proprietary model developed by [Stability AI](https://stability.ai). It is designed to improve the accuracy of audio processing tasks, such as speech recognition and speaker identification.

### Vision

#### Midjourney

[Midjourney](https://www.midjourney.com/home) is a proprietary model for Image generation developed by [Midjourney](https://www.midjourney.com/home).

## Open-Source Models

Subsection | Description
-----------|------------
[](#before-public-awareness) | Pre-[](#chatgpt); before widespread LLMs use, and a time of slow progress.
[](#early-models) | Post-[](#chatgpt); time of [](#stable-diffusion) and [](#llama)
[](#current-models) | Post-[](#llama) leak; open-source LLMs quickly catching up to closed-source, new solutions emerging (e.g. GPU-poor), [](#alpaca-7b), LLaMA variants, etc.

[](#chatgpt) would be playing a huge role if it was a story of LLMs and how they fast-tracked their improvements.
Early high performing LLMs were proprietary, accessible only through organisations' paid APIs, hindering transparency and raising concerns about data privacy, bias, alignment and robustness, giving limited possibilities to cater domain-specific use cases without letting {term}`RLHF` alignment {cite}`lambert2022illustrating` interfere.

### Before Public Awareness

Recognising the need for openness, the LLM research community responded by creating open-source variants, laying the foundation for increased transparency and the development of more powerful models.

There has been few notable open LLMs pre-ChatGPT era like [BLOOM](https://bigscience.huggingface.co/blog/bloom), GPT-NewX 20B {cite}`black2022gptneox20b`, [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b), OPT {cite}`zhang2022opt`.

#### GPT-J 6B

[GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b) is an early English-only casual language model, which at the time of its release was the largest publicly available GPT-3 style language model. [Code and weights are open sourced](https://github.com/kingoflolz/mesh-transformer-jax#gpt-j-6b) along with a [blog](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j) by [Aran Komatsuzaki](https://arankomatsuzaki.wordpress.com), one of the authors of the model.

##### Uniqueness

- It belongs to the GPT-J class of models, and has 6 billion trainable parameters.
- Uses same tokeniser as GPT-2/3.
- Uses Rotary Position Embedding (RoPE) {cite}`su2022roformer`
- Used open sourced dataset for training -- Pile {cite}`gao2020pile`, a large scale dataset curated by [EleutherAI](https://www.eleuther.ai).
- The dimension of each attention head is set to 256, which is twice larger than that of GPT-3 of comparable size, which improved throughput with minimal performance degradation.
- Places the attention layer and the feed-forward layer in parallel for decreased communication.

##### Limitations

- It's trained on an English-only dataset.
- The Pile {cite}`gao2020pile` dataset which was used for training is known to contain profanity, lewd and abrasive language too.

Before [](#chatgpt)'s (GPT-3.5) public release we had [GPT-3](https://en.wikipedia.org/wiki/GPT-3) being one of the "[best](https://www.reddit.com/r/MachineLearning/comments/ydwi6c/d_whats_the_best_open_source_model_for_gpt3like)" Base Language Model which released ~2.1 years before ChatGPT. And following that we've had LLMs like [Bard](https://blog.google/technology/ai/bard-google-ai-search-updates), [Claude](https://www.anthropic.com/index/introducing-claude), [GPT-4](#gpt-4) and [others](https://lmsys.org/blog/2023-05-25-leaderboard).

### Early Models

There has been a few visible marks across modalities of AI models, highly catalysing growth of open source:

- [Meta AI launches LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai), open sourcing the code but not the weights.
- [StabilityAI released Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement).

#### [Stable Diffusion](https://registry.premai.io/detail.html?service=stable-diffusion-1-5)

Stable Diffusion is a latent text-to-image diffusion model {cite}`rombach2022highresolution`. Created by [Stability AI](https://stability.ai) and support from [LAION](https://laion.ai), where they used 512x512 images from a subset of the [LAION 5B](https://laion.ai/blog/laion-5b) database for training. Similar to Google's Imagen {cite}`saharia2022photorealistic`, this model uses a frozen CLIP ViT-L/14 {cite}`radford2021learning` text encoder to condition the model on text prompts. With its 860M UNet and 123M text encoder, the model is relatively lightweight and runs on a GPU with at least 10GB VRAM.

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

Under LLaMA {cite}`touvron2023llama`, [Meta AI](https://ai.meta.com) released a collection of foundation language models ranging from 7B to 65B parameters, pre-trained over a corpus containing more than 1.4 trillion tokens. It was designed to be versatile and applicable for many different use cases, and possibly fine-tuned for domain specific tasks if required.

It showed **better performance** across domains compared to its competitors.

```{figure} https://static.premai.io/book/models_llama-scores.png
:width: 88%
LLaMA: Open and Efficient Foundation Language Models {cite}`touvron2023llama`
```

LLaMA 13B outperforms [GPT-3 (175B)](https://en.wikipedia.org/wiki/GPT-3) on most benchmarks while being more than 10x smaller, and LLaMA 65B is competitive with models like Chinchilla 70B {cite}`hoffmann2022training` and [PaLM 540B](https://blog.research.google/2022/04/pathways-language-model-palm-scaling-to.html). LLaMA 65B performs similarly to the closed-source GPT-3.5 on the MMLU and GSM8K benchmarks {cite}`touvron2023llama`.

##### Uniqueness

There are few key inspirations LLaMA architecture took from other LLMs:

- **Pre-normalisation (GPT-3):** using RMSNorm to normalise transformer sub-layer inputs {cite}`zhang2019root`.
- **SwiGLU activation function (PaLM):** replacing ReLU with SwiGLU {cite}`shazeer2020glu`.
- **Rotary Embeddings (GPTNeo):** replacing absolute positional embeddings with Rotary positional embeddings {cite}`su2022roformer`.

##### Limitations

- It was released under a non-commercial license focused on usage for research use cases only.
- LLaMA is a {term}`foundation model` and not fine-tuned for specific tasks, which may limit its performance on certain tasks
- LLaMA seemed not as competitive as other models on certain benchmarks, such as BoolQ and WinoGrande.

Interestingly within a week from LLaMA's launch, its [weights were leaked to the public](https://www.vice.com/en/article/xgwqgw/facebooks-powerful-large-language-model-leaks-online-4chan-llama). https://github.com/facebookresearch/llama/pull/73 created a huge impact on the community for all kinds innovations coming up, even though there was still license restrictions not permitting commercial usage.

### Current Models

After 2 weeks from the LLaMa weights leak, Stanford [releases Alpaca 7B](https://crfm.stanford.edu/2023/03/13/alpaca.html).

#### Alpaca 7B

It's a 7B parameter model fine-tuned from LLaMA 7B model on 52K instruction-following data-points. It performs qualitatively similarly to OpenAI's text-davinci-003 while being smaller and cheaper to reproduce i.e taking only < \$600. Github repository [here](https://github.com/tatsu-lab/stanford_alpaca).

```{figure} https://static.premai.io/book/models_alpaca-finetuning.png
:width: 80%
[Alpaca 7B fine-tuning strategy](https://crfm.stanford.edu/2023/03/13/alpaca.html)
```

##### Uniqueness

- Unique Data Source: Alpaca 7B is distinct for being fine-tuned from LLaMA 7B using 52K instruction-following demonstrations coming from self-instruct {cite}`wang2023selfinstruct`, in the style of text-davinci-003, enabling research into instruction-following scenarios.
- Cost-Efficient Alternative: Alpaca 7B offers similar performance to text-davinci-003 but at a lower cost, making it accessible for academic research.

##### Limitations

- Non-commercial Usage: This limitation arises from the non-commercial license of LLaMA, upon which Alpaca is based.
- Quality: Alpaca 7B may occasionally produce inaccurate information, including hallucinations, misinformation, and toxic content.
- Evaluation Scope: While Alpaca performs well in some evaluations, its performance may vary in unexplored scenarios.

Right after that [alpaca-lora](https://github.com/tloen/alpaca-lora) came out, using low rank fine-tuning it made possible to reproduce Alpaca within hours on a single NVIDIA RTX 4090 GPU with inference being possible even [on a Raspberry PI](https://twitter.com/miolini/status/1634982361757790209).

Things moved fast from here when first promising inference speed was achieved without GPU for LLaMA using 4 bit quantisation by the [LLaMA GGML](https://github.com/ggerganov/llama.cpp). A new wave of [quantised models started coming from the community](https://huggingface.co/TheBloke).

In a day after, [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna) came in.

#### [Vicuna](https://registry.premai.io/detail.html?service=vicuna-7b-q4)

[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna) was released under a joint effort by UC Berkeley, CMU, Stanford, UC San Diego, and MBZUAI. It was trained by fine-tuning LLaMA on user-shared conversations collected from ShareGPT. GPT-4 was used for its evaluation. They released a [demo](https://chat.lmsys.org) and [code](https://github.com/lm-sys/FastChat), [weights](https://github.com/lm-sys/FastChat#vicuna-weights) under non-commercial license following LLaMa.

```{figure} https://static.premai.io/book/models_vicuna-finetuning.png
:width: 80%
[Vicuna fine-tuning strategy](https://lmsys.org/blog/2023-03-30-vicuna/#overview)
```

##### Uniqueness

- Impressive Quality: Vicuna 13B achieved over 90% quality compared to ChatGPT and Google Bard, surpassing other models like LLaMA and Stanford Alpaca in more than 90% of cases.
- For training:
  - Training loss was adjusted to account for multi-turn conversations and compute the fine-tuning loss solely on the chatbot's output.
  - Expanded max context length from 512 in Alpaca to 2048, gradient checkpointing {cite}`chen2016training` and flash attention {cite}`dao2022flashattention` utilisation helping handle memory pressure.
  - Used [SkyPilot](https://github.com/skypilot-org/skypilot) [managed spot](https://skypilot.readthedocs.io/en/latest/examples/spot-jobs.html) to reduce the cost for training the 7B model from \$500 to around \$140 and the 13B model from around \$1k to \$300.
- Cost-Efficiency: The cost of training was around \$300, making it a cost-effective choice for research purposes.
- Enhanced Dataset: Vicuna is fine-tuned using 70K user-shared ChatGPT conversations from [ShareGPT](https://sharegpt.com), enabling it to provide detailed and well-structured answers, with performance on par with ChatGPT.

##### Limitations

- Reasoning and Safety: Vicuna may struggle with tasks involving reasoning or mathematics and may not always ensure factual accuracy. It has not been fully optimised for safety or to mitigate potential toxicity or bias.
- Evaluation Framework: The proposed evaluation framework, based on GPT-4, is not yet a rigorous or mature approach, as large language models can sometimes produce hallucinated responses.
- No Dataset release.
- Non-commercial usage only following the LLaMA model's license, OpenAI's [data terms](https://openai.com/policies/terms-of-use) and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT.

After the release they also conducted a [deeper study on GPT4-based evaluation approach](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge#llm-judge).

Then came in updates like LLaMa-Adapter {cite}`zhang2023llamaadapter`, [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala) and in less than a month [Open Assistant](https://open-assistant.io) launches a model and a dataset for Alignment via {term}`RLHF` {cite}`köpf2023openassistant`.

Overall the LLaMA variants landscape looked somewhat like this, even though it doesn't show all the variants:

```{figure} https://static.premai.io/book/models_llama-variants.png
:width: 80%
Page 10, A Survey of Large Language Models {cite}`zhao2023survey`
```

After a month, WizardLM dropped in which gained a lot of popularity mainly due to its ground breaking performances compared to other open LLMs. And in next few days the community did an open reproduction of LLaMA, named [OpenLLaMA](https://github.com/openlm-research/open_llama).

#### WizardLM

[WizardLM](https://huggingface.co/WizardLM) is created by fine-tuning LLaMA on a generated instruction dataset which was created by Evol-Instruct {cite}`xu2023wizardlm`.

##### Uniqueness

- Proposed Evol-Instruct -- method using LLMs instead of humans to automatically mass-produce open-domain instructions of various difficulty levels, to improve the performance of LLMs.
- It achieves better response quality than Alpaca and Vicuna on the automation evaluation using GPT-4.
- Shows Evol-Instruct method for creating instruction tuning datasets are superior to the ones from human-created [ShareGPT](https://sharegpt.com).

  ```{figure} https://static.premai.io/book/models_wizardlm.png
  :width: 88%
  Page 4, WizardLM: Empowering Large Language Models to Follow Complex Instructions {cite}`xu2023wizardlm`
  ```

##### Limitations

- Overall does not outperform ChatGPT except in few cases.

#### OpenLLaMA

Students at UC Berkeley started [OpenLM Research group](https://huggingface.co/openlm-research) through which they trained in collaboration with [Stability AI](https://stability.ai) to release [OpenLLaMA](https://github.com/openlm-research/open_llama) v1, a permissively licensed open source reproduction of Meta AI's LLaMA. They released a series of 3B, 7B and 13B models trained on [different mix of datasets](https://huggingface.co/openlm-research). And the weights released can serve as drop in replacement of LLaMA.

##### Uniqueness

- Model is trained on open sourced [RedPajama dataset](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) by [Together](https://huggingface.co/togethercomputer).
- All steps for training are kept same as mentioned in LLaMA {cite}`touvron2023llama`.
- Model is trained on 1T tokens.
- Weights released under Apache 2.0 license, in two formats:
  - EasyLM format to be use with https://github.com/young-geng/EasyLM framework
  - PyTorch format to be used with the [Hugging Face transformers library](https://huggingface.co/docs/transformers/index)

##### Limitations

- Dataset Difference: OpenLLaMA uses open datasets instead of the original LLaMA dataset. While training procedures, architecture, and other parameters remain the same, there may be differences in performance on certain tasks.

Around same time [MosaicML](https://www.databricks.com/company/newsroom/press-releases/databricks-completes-acquisition-mosaicml) released its [MPT](https://github.com/mosaicml/llm-foundry) models series, and [TII](https://www.tii.ae) also released [Falcon models](https://www.tii.ae/news/uaes-technology-innovation-institute-launches-open-source-falcon-40b-large-language-model).

#### [MPT](https://registry.premai.io/detail.html?service=mpt-7b)

MosaicML released [MPT (MosaicML Pretrained Transformer) models series](https://huggingface.co/mosaicml) consisting:

- 7B variants:
  - [MPT 7B base](https://registry.premai.io/detail.html?service=mpt-7b)
  - [MPT 7B-Instruct](https://registry.premai.io/detail.html?service=mpt-7b-instruct)
  - [MPT 7B-Chat](https://registry.premai.io/detail.html?service=mpt-7b-chat)
  - [MPT 7B-StoryWriter-65k+](https://huggingface.co/mosaicml/mpt-7b-storywriter)
- 30B variants:
  - [MPT 30B base](https://huggingface.co/mosaicml/mpt-30b)
  - [MPT 30B-Instruct](https://huggingface.co/mosaicml/mpt-30b-instruct)
  - [MPT 30B-Chat](https://huggingface.co/mosaicml/mpt-30b-chat)

##### Uniqueness

- Licensed for commercial usage (not all variants in the series): MPT 7B base, MPT 7B-StoryWriter-65k+, MPT 30B were only released under Apache-2.0 license.
- Uses ALiBi {cite}`press2022train` to handle long inputs till 84k tokens context size, whereas trained using upto 65k tokens context.
- Uses FlashAttention {cite}`dao2022flashattention` and https://github.com/NVIDIA/FasterTransformer to optimise for fast training and inference.
- They also released an entire framework, the [MosaicML LLM Foundry](https://github.com/mosaicml/llm-foundry).

##### Limitations

- Not all variants were released under permissive commercial usage license.
- Combinations of open sourced datasets was used for training the models and [mentioned which ones with proportions](https://github.com/mosaicml/llm-foundry/issues/499#issuecomment-1662556022), but haven't released the [combined dataset yet](https://github.com/mosaicml/llm-foundry/issues/499).

#### Falcon

[TII](https://falconllm.tii.ae) released [Falcon series of 40B, 7.5B and 1.3B parameters LLMs](https://falconllm.tii.ae/falcon.html), trained on their open sourced and curated RefinedWeb dataset. After the release it has dominated the [Huggingface's open llm leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for the State of the Art open sourced LLM for more than 2 months.

##### Uniqueness

- Falcon 40B has data from a variety of English, German, Spanish, French, Italian, Portuguese, Polish, Dutch, Romanian, Czech, and Swedish languages inserted into its pre-training set.
- They released all the model and its [instruction tuned](https://registry.premai.io/detail.html?service=falcon-7b-instruct) and chat variants under Apache 2.0 license, permitting commercial usage.
- The model uses only 75 percent of GPT-3's training compute, 40 percent of Chinchilla AI's, and 80 percent of PaLM 62B's.
- Falcon 40B pre-training dataset contained around 5 Trillion tokens gathered from public web crawls (~80%), research papers, legal text, news, literature, and social media conversations.
  - Subset of this dataset containing 600 Billion tokens {cite}`penedo2023refinedweb` was open sourced.
- Model uses decoder-only architecture with Flash Attention {cite}`dao2022flashattention`, Multi-Query Attention {cite}`shazeer2019fast`, Parallel Attention and Feed Forward {cite}`sonkar2023investigating`.

##### Limitations

- Full dataset used for pre-training the 40B variant wasn't released.
- Falcon 40B is trained using a sequence length of 2K, which is smaller compared to MPT, XGen, but context size can be increased using RoPE embeddings {cite}`su2022roformer` within a model's architecture, allowing it to generalise to longer sequence lengths (might require some [](fine-tuning)).
- A paper detailing Falcon models specifically has not yet been released.

#### LLaMA-2

On 18th July, Meta AI released LLaMA-2, breaking most {term}`SotA` records on open sourced LLMs performances.

Meta AI https://github.com/facebookresearch/llama with both pre-trained and fine-tuned variants for a series of [7B](https://registry.premai.io/detail.html?service=llama-2-7b), [13B](https://registry.premai.io/detail.html?service=llama-2-13b) and [70B](https://huggingface.co/meta-llama/Llama-2-70b) parameter sizes.

Some win rate graphs on LLaMA-2 after evaluation comparisons against popular LLMs where it roughly ties with GPT-3.5 and performs noticeably better than Falcon, MPT and Vicuna.

```{figure} https://static.premai.io/book/models_llama2-rates.png
:width: 88%
Page 3, LLaMA 2: Open Foundations and Fine-Tuned Chat Models {cite}`touvron2023llama2`
```

##### Uniqueness

- LLaMA-2 models are pre-trained over 2 trillion tokens dataset in total, compared to 1.4 trillion tokens dataset for LLaMA-1.
- LLaMA-2 models are trained with a 4k context length, whereas it's 2k for LLaMA-1.
- Larger variants use grouped query attention (GQA) {cite}`ainslie2023gqa` within their underlying architecture, helping improve inference efficiency.

    ```{figure} https://static.premai.io/book/models_llama2-gqa.png
    :width: 80%
    GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints {cite}`ainslie2023gqa`.
    ```

- LLaMA-2 70B became new state-of-the-art among open-source LLMs on all tasks considered.

    ```{figure} https://static.premai.io/book/models_llama2-opensource-scores.png
    :width: 80%
    Page 8, LLaMA 2: Open Foundations and Fine-Tuned Chat Models {cite}`touvron2023llama2`
    ```

- They released chat variants from base models using instruction tuning and high scale {term}`RLHF`, also proposed a Ghost Attention (GAtt) which helps control dialogue flow over multiple turns.

    ```{figure} https://static.premai.io/book/models_llama2-workflow.png
    :width: 80%
    Page 5, LLaMA 2: Open Foundations and Fine-Tuned Chat Models {cite}`touvron2023llama2`
    ```

- For Alignment uses a two-stage RLHF approach, starting with Rejection Sampling, then doing Rejection Sampling + Proximal Policy Optimisation (PPO)
- All model variants under LLaMA-2 are released under [LLaMA-2 License](https://opensourceconnections.com/blog/2023/07/19/is-llama-2-open-source-no-and-perhaps-we-need-a-new-definition-of-open), permitting commercial usage unless it's facing 700 million monthly active users then the entity must obtain a license from Meta.
- Meta's team does quite some work for mitigating AI safety issues in the model.
  - Released a [responsible Use Guide](https://github.com/facebookresearch/llama/blob/main/Responsible-Use-Guide.pdf).

##### Limitations

- LLaMA-2 base models perform worse compared to aligned proprietary models, but performs favourably when compared to popular base LLMs like PaLM {cite}`chowdhery2022palm`.

    ```{figure} https://static.premai.io/book/models_llama2-proprietary-scores.png
    :width: 80%
    Page 8, LLaMA 2: Open Foundations and Fine-Tuned Chat Models {cite}`touvron2023llama2`
    ```

- LLaMA-2 Chat model variants can sometimes give overly cautious responses due to high safety tuning on the model.
- Reward models used in the model alignment steps aren't open sourced yet.

Till now we've mostly been looking at LLMs in general and not other models, let's look at the vision domain now.

#### [Stable Diffusion XL](https://registry.premai.io/detail.html?service=stable-diffusion-xl-with-refiner)

[StabilityAI released Stable Diffusion XL 1.0 (SDXL)](https://stability.ai/blog/stable-diffusion-sdxl-1-announcement) models on 26th July, being current State of the Art for text-to-image and image-to-image generation open sourced models. They released a [base model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and a [refinement model](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) which is used to improve the visual fidelity of samples generated by SDXL.

Few months back they released Stable-diffusion-xl {cite}`podell2023sdxl` [base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9) and [refinement](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9) models versioned as 0.9, where license permitting only research purpose usages.

SDXL consistently surpasses all previous versions of Stable Diffusion models by a significant margin:

```{figure} https://static.premai.io/book/models_sdxl-winrate.png
:width: 60%
[SDXL Winrate](https://stability.ai/blog/stable-diffusion-sdxl-1-announcement)
```

##### Uniqueness

- Works effectively on GPUs with 8GB or more VRAM.
- 3x larger UNet-backbone compared to previous Stable Diffusion models.
- Introduces a two-stage model process; the base model (can work standalone) generates an image as an input to the refiner model which adds additional high-quality details.

  ```{figure} https://static.premai.io/book/models_sdxl-arch.png
  :width: 78%
  SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis {cite}`podell2023sdxl`
  ```

- Proposed two additional model conditioning techniques to preserve training data from being discarded and gain more control over how a generated image should be cropped:
  - Conditioning the Model on [Image Size](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#size-conditioning).
  - Conditioning the Model on [Cropping Parameters](https://huggingface.co/docs/diffusers/main/en/using-diffusers/sdxl#crop-conditioning).
- Commercial usage [allowed by SDXL License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/12#64c237c5f3977a70e19142ed).
- They also released a processed [TensorRT variant of SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt#stable-diffusion-xl-10-tensorrt), which can give upto [41% latency and 70% throughput improvements](https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt#performance-comparison).
- [Clipdrop](https://clipdrop.co/stable-diffusion) provides free SDXL inference.

##### Limitations

- For high quality generations from SDXL, a two-stage approach is required i.e using an additional refinement model, having to load two large models into memory hampers accessibility and sampling speed.
- Generations are sometimes poor when synthesising intricate structures, such as human hands, or while rendering long legible text.
- Model achieves a remarkable level of realism in its generated images but does not attain perfect photorealism.
- Model's training process heavily relies on large-scale datasets, possibly introducing social and racial biases.

In the domain of Image generation currently [Midjourney](https://www.midjourney.com) is one of the most popular proprietary solutions for [simple users](https://www.reddit.com/r/StableDiffusion/comments/15i6tg3/are_we_killing_the_future_of_stable_diffusion/jusrar3).

Following the timeline and going back to text domain, coder models are gaining lot of popularity too, specially looking at the code generation or code analysis capabilities of OpenAI's codex and GPT-4, there has been few releases on code LLMs like WizardCoder {cite}`luo2023wizardcoder`, [StarCoder](https://huggingface.co/bigcode/starcoder), [Code LLaMA](https://huggingface.co/codellama) (state of the art) and [many more](https://huggingface.co/models?language=code).

#### Code LLaMA

[Code LLaMA](https://ai.meta.com/blog/code-llama-large-language-model-coding) release by [Meta AI](https://ai.meta.com/about) (right after ~1.5 month from LLaMA 2's release) caught lot of attention being full open source.
And currently [its fine-tuned variants](https://huggingface.co/Phind/Phind-CodeLlama-34B-v2) are state of the art among open source coder models.

##### Uniqueness

- [Outperforms GPT-3.5](https://www.reddit.com/r/OpenAI/comments/160bbaq/comment/jxls1xq) on code generation capabilities.
- Uses [](#llama-2) as {term}`foundation model`.
- Released [three variants](https://huggingface.co/codellama) for each model sizes:
  - **Code LLaMA**: constitute foundation models for code generation. They come in three model sizes: 7B, 13B and 34B parameters. The 7B and 13B models are trained using an infilling objective, appropriate for code generation in an IDE. The 34B model was trained without the infilling objective
  - **Code LLaMA -- Python**: specialised for Python code generation and also come in sizes of 7B, 13B, and 34B parameters. Trained on 500B tokens from the Code LLaMA dataset and further specialised on 100B tokens using a Python-heavy dataset. Python variants are trained without infilling and subsequently fine-tuned to handle long contexts.
  - **Code LLaMA -- Instruct**: based on Code LLaMA and fine-tuned with an additional approx. 5B tokens to better follow human instructions.

    ```{figure} https://static.premai.io/book/models_codellama-pipeline.png
    :width: 88%
    Page 3, Code LLaMA: Open Foundation Models for Code {cite}`rozière2023code`
    ```

- Reached state-of-the-art performance among open models on several code benchmarks, with scores of up to 53% and 55% on [HumanEval](https://github.com/openai/human-eval) and [MBPP](https://github.com/google-research/google-research/tree/master/mbpp), respectively.

  ```{figure} https://static.premai.io/book/models_codellama-scores.png
  :width: 78%
  Page 7, Code LLaMA: Open Foundation Models for Code {cite}`rozière2023code`
  ```

- Supports code [infilling](https://huggingface.co/blog/codellama#code-infilling).
- All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens.
- Data is tokenised via byte pair encoding, using the same tokeniser as LLaMA and LLaMA 2.
- Instruction tuning dataset combines thousands of Supervised Fine-Tuning and millions of Rejection Sampling examples.
- Have been trained between January 2023 and July 2023.
- Commercial usage: released under [permissive license](https://github.com/facebookresearch/codellama/blob/main/LICENSE) that allows for both research and commercial use, same as LLaMA 2.

##### Limitations

- Proprietary dataset: No Code LLaMA dataset open source release yet.
- For 7B and 13B variants' large context fine-tuning and infilling comes at a cost on standard benchmarks.
- Performs [worse](https://www.reddit.com/r/OpenAI/comments/160bbaq/meta_has_released_code_llama_although_gpt4) compared to GPT-4.

#### Persimmon 8B

[Persimmon 8B](https://www.adept.ai/blog/persimmon-8b) is a standard decoder-only transformer model released under an Apache-2.0 license. Both code and weights are available at https://github.com/persimmon-ai-labs/adept-inference.

##### Uniqueness

- It has a large context size of 16K, four times that of LLaMA2 and eight times that of GPT-3 and MPT models.
- It is a fully permissively licensed under Apache 2.0 and under 10 Billion parameters, making it highly suitable for commercial usage.
- It includes 70k unused embeddings for potential multimodal extensions and incorporates sparse activations.
- It's trained on 0.37x as much data as LLaMA2 and despite that exceeds other ~8B models and matches LLaMA2 performance. Training dataset consists ~25% code and 75% text.

  ```{figure} https://static.premai.io/book/models_persimmon-scores.png
  :width: 60%
  [Pers 8B Results](https://www.adept.ai/blog/persimmon-8b#results)
  ```

- Uses a [vocabulary of 262k tokens](https://twitter.com/suchenzang/status/1700214181772013762), built using a unigram sentencepiece model.
- Architecture is skinnier and deeper than LLaMA-2 7B.
- They developed an [improved version of FlashAttention](https://www.adept.ai/blog/flashier-attention).
- Inference optimisations possible.
- In the model architecture it uses:
  - Uses [squared ReLU activation function](https://www.adept.ai/blog/persimmon-8b#model-details).
  - Uses RoPE {cite}`su2022roformer` and QKNorm {cite}`henry2020querykey` which might've been mostly needed to stabilise squared ReLU training since it was also used to reduce instability issues in ViT 22B model {cite}`dehghani2023scaling`.

##### Limitations

- Normally it's not recommended to train from scratch with 16k context size, as depending on dataset, simply increasing context length will cause model to attend across more unrelated documents.

#### Mistral 7B

[Mistral 7B](https://huggingface.co/mistralai) is released by [Mistral AI](https://mistral.ai), a french startup which recently [raised a good seed round](https://techcrunch.com/2023/06/13/frances-mistral-ai-blows-in-with-a-113m-seed-round-at-a-260m-valuation-to-take-on-openai). The team comprises of ex-[Deepmind](https://www.deepmind.com) and ex-[Meta](https://ai.meta.com) researchers, who worked on [](#llama), Flamingo {cite}`alayrac2022flamingo` and [Chinchilla](https://en.wikipedia.org/wiki/Chinchilla_AI) projects.

##### Uniqueness

- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) outperforms [LLaMA-2 13B](https://registry.premai.io/detail.html?service=llama-2-13b) on all and LLaMA-1 34B on code, math, and reasoning benchmarks.

  ```{figure} https://static.premai.io/book/models_mistral-7b-comparison.png
    :width: 70%
    [Mistral 7B Comparison](https://mistral.ai/news/announcing-mistral-7b)
  ```

- Close to Code LLaMA 7B performance on code, while remaining good at English tasks.
- Uses Grouped-query attention (GQA) {cite}`ainslie2023gqa` for faster inference.
- Uses [Sliding Window Attention (SWA)](https://github.com/mistralai/mistral-src#sliding-window-attention) {cite}`child2019generating,beltagy2020longformer` to handle longer sequences at smaller cost.
- Uses Byte-fallback BPE tokenizer.
- Released [7B base](https://huggingface.co/mistralai/Mistral-7B-v0.1) model and [7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model which outperforms all 7B models on MT-Bench {cite}`zheng2023judging` and outperforms [LLaMA-2 13B-Chat](https://huggingface.co/meta-llama/Llama-2-13b-chat).
- Both models released under Apache 2.0 license, with no restrictions.
- [Released a codebase](https://github.com/mistralai/mistral-src) which documents how to run and explains some concepts used in the model.

##### Limitations

- No training/fine-tuning code or paper has been released yet.
- No training or fine-tuning dataset has been released even though they mentioned usage of datasets publicly available on HuggingFace for fine-tuning.

## Comparisons

Here we went through the properties of popular models in Text and Visual domains. Comparing Large Language Models to a single source of truth is an inherently very difficult task, and Comparing visual models even harder. Since while generalising capabilities it's really important to take care of racial, gender, religious and other biases that the model can have. There are lot of popular [leaderboards](leaderboards-table) to track these models' aggregate or specific performances, based on [evaluation datasets](eval-datasets.md) curated by the community exactly for measuring capabilities, each catering to specific needs.

Our current based approaches for comparisons include evaluating each model on each dataset and get an average score across datasets. Combining this with evaluations performed by having [humans and GPT-4 compare completions](https://huggingface.co/spaces/HuggingFaceH4/human_eval_llm_leaderboard), gives a somewhat trustable score for tracking the current best. But this current way is not enough, even pillar models like GPT-4 [fails](https://twitter.com/cHHillee/status/1635790330854526981), and it's [hard to determine](https://www.technologyreview.com/2023/08/30/1078670/large-language-models-arent-people-lets-stop-testing-them-like-they-were#piano__post_body-mobile-3) on how much similar data to evaluation set has actually been a part of training set.

### Language

[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) shows us that Falcon 180B is currently just ahead of Meta's LLaMA-2 70B, and TII claims that it ranks just behind OpenAI's GPT 4, and performs on par with Google's PaLM-2 Large, which powers Bard, despite being half the size of the model. But it required 4x more compute to train and it's 2.5 times larger compared to LLaMA-2, which makes it not so cost-effective for commercial usages.

For practical commercial usage models ranging below 14B parameters has been a good candidate, and [](#mistral-7b), [LLaMA-2 7B](#llama-2), [](#persimmon-8b) does a great job showing that.

Overall let's take look at the few discussed LLMs' attributes to get the bigger picture.

```{table} Under 15 Billion Parameters
:name: llms-below-15b
LLMs | Params/[B] | Dataset | Release Details | Tokens/[B] | VRAM/[GB] | License | Commercial Usage
:----|-----------:|:-------:|----------------:|-----------:|:---------:|--------:|-----------------:
[Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 7.3 | - | [Blog](https://mistral.ai/news/announcing-mistral-7b) | - | 17+ | Apache-2.0 | ✅
[LLaMA-2 13B](https://registry.premai.io/detail.html?service=llama-2-13b) | 13 | - | {cite}`touvron2023llama2` | 2000 | 29+ | [LLaMA-2](https://blog.opensource.org/metas-llama-2-license-is-not-open-source) | ✅
[LLaMA-2 7B](https://registry.premai.io/detail.html?service=llama-2-7b) | 7 | - | {cite}`touvron2023llama2` | 2000 | 15.8+ | [LLaMA-2](https://blog.opensource.org/metas-llama-2-license-is-not-open-source) | ✅
[Persimmon 8B](https://huggingface.co/docs/transformers/main/model_doc/persimmon) | 9.3 | - | [Blog](https://www.adept.ai/blog/persimmon-8b) | 737 | 20.8+ | [Apache-2.0](https://github.com/persimmon-ai-labs/adept-inference/blob/main/LICENSE) | ✅
[WizardLM 13B](https://huggingface.co/WizardLM/WizardLM-13B-V1.2) | 13 | [evol-instruct](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k) | {cite}`xu2023wizardlm` | ~2000 | 30+ | [LLaMA-2](https://blog.opensource.org/metas-llama-2-license-is-not-open-source) | ✅
[WizardLM 7B](https://huggingface.co/WizardLM/WizardLM-7B-V1.0) | 7 | [evol-instruct](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k) | {cite}`xu2023wizardlm` | ~2000 | 15.8+ | Non-Commercial | ❌
[Falcon 7B](https://huggingface.co/tiiuae/falcon-7b) | 7 | [RefinedWeb (partial)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | - | 1500 | 16+ | [Apache-2.0](https://huggingface.co/tiiuae/falcon-7b#license) | ✅
[MPT 7B](https://huggingface.co/mosaicml/mpt-7b) | 6.7 | [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | [Blog](https://www.mosaicml.com/blog/mpt-7b) | 1000 | 15.5+ | [Apache-2.0](https://huggingface.co/mosaicml/mpt-7b#model-license) | ✅
```

### Vision

StabilityAI's SDXL vs [Midjourney](https://www.midjourney.com) comparison shows that it is on par with favourability.

```{figure} https://static.premai.io/book/models_sdxl-midjourney.png
:width: 88%
Page 14, SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis {cite}`podell2023sdxl`
```

```{note} Above experiment is against Midjourney v5.1, whereas current latest is [Midjourney v5.2](https://docs.midjourney.com/docs/model-versions).
```

## Future

To recap current advancements we can see that few key moments were:

- Release of [](#chatgpt), [](#gpt-4), DALL-E by OpenAI.
- Release of [Stable Diffusion models](#stable-diffusion) by StabilityAI.
- Leak of [](#llama) weights, and [](#llama-2)'s release by Meta.
- Creation and release of {term}`RLHF` recipes.
- a few [smaller moments](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither#%C2%A7the-timeline).

Even though Open Source AI is advancing, it is evident that it remains heavily regulated by major corporations such as Meta, OpenAI, Nvidia, Google, Microsoft, and others. These entities often control critical parameters, creating a myth of open source AI {cite}`myth-of-os-ai-wired`, including:

- Data required to train these models.
- Control of Software frameworks required to build such models
- Compute power required to train these models.

Returning to actual state, there are significant gaps that need to be addressed to achieve true progress in the development of intelligent models. For instance, recent analyses have revealed the limited generalization capabilities {cite}`reversal-curse`, current LLMs learn things in the specific direction of an input context window of an occurrence and may not generalize when asked in other directions.

The rise of {term}`MoE` models has garnered attention and research interest, particularly following rumours about the GPT-4 architecture. The open-source community has already made strides in implementing various MoE variants (e.g. https://github.com/XueFuzhao/OpenMoE) demonstrating a push toward more versatile model architectures.

On another part using quantized version of models usages are increasing rapidly, as it makes running large models (>30B parameters) possible on low precision, even on just cpu machines. Specially lots of contributions in this area is coming up by https://github.com/ggerganov/ggml community and [TheBloke](https://huggingface.co/TheBloke).

{{ comments }}
