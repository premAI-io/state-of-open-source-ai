# Models

% LLaMA 1 vs 2, Falcon, Stable Diffusion, DALL-E

The emergence of Large Language Models, notably with the advent of GPT-3, ChatGPT, Midjourney, Whisper helped bloom a new era. Beyond revolutionizing just language models, these models also pushed innovation in other domains like Vision (ViT, DALL-E, Stable Diffusion etc), Audio (Wave2vec, Bark) or even Multimodal models.

% TODO: maybe make the above model names as references and not names directly.

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


### First steps
There has been a few visible marks across modalities of AI models, highly catalysing growth of open source:
- [Meta AI released LLaMA](https://ai.meta.com/blog/large-language-model-llama-meta-ai/).
- [StabilityAI released Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement).
- [OpenAI released Whisper](https://openai.com/research/whisper).

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

#### Stable Diffusion

#### Whisper

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
