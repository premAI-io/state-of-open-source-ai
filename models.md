# Models

% LLaMA 1 vs 2, Falcon, Stable Diffusion, DALL-E

The emergence of Large Language Models, notably with the advent of GPT-3, ChatGPT, Midjourney, Whisper helped bloom a new era. Beyond revolutionizing just language models, these models also pushed innovation in other domains like Vision (ViT, DALL-E, Stable Diffusion etc), Audio (Wave2vec, Bark) or even Multimodal models.

% TODO: maybe make the above model names as references and not names directly.

## Rise of Open-Source Models
% TODO: tell about how emergent of openai models created an increase in OSS models, even though there were primitive versions of that available, but now present with rlhf etc, why it helps/had a rise to cater the specific use cases which generic openai models couldn't capture

Early LLMs were often proprietary, accessible only through organisation's paid APIs, hindering transparency and raising concerns about data privacy, bias, alignment and robustness, giving limited possibilities to cater domain-specific use cases without letting RLHF'ed alignment intefere. Recognizing the need for openness, the LLM research community responded by creating open-source variants, laying the foundation for increased transparency and the development of more powerful models.
% TODO: ^^add refs

## Catching Up to Close-Source Models
% ChatGPT, Midjourney and Others
% TODO: talking about what OSS community did to catchup on chatgpt performance i.e newer versions came up of models (e.g llama, llama v2, SD, SD XL)

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
