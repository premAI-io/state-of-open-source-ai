# State of Open Source AI Book - 2023 Edition

{{ '```{badges} %s %s\n:doi: %s\n```' % (baseurl, env.config.html_theme_options.repository_url, doi) }}

*Clarity in the current fast-paced mess of Open Source innovation {cite}`prem_stateofosai`*

As a data scientist/ML engineer/developer with a 9 to 5 job, it's difficult to keep track of all the innovations. There's been enormous progress in the field in {term}`the last year <SotA>`.

Cure your FOMO with this guide, covering all the most important categories in the Open Source AI space, from model evaluations to deployment. It includes a [](#glossary) for you to quickly check definitions of new frameworks & tools.

A quick TL;DR overview is included at the top of each section. We outline the pros/cons and general context/background for each topic. Then we dive a bit deeper. Examples include data models were trained on, and deployment implementations.

## Who is This Guide For?

```{admonition} Prerequisites to Reading
:class: warning
You should already know the basic principles of MLOps {cite}`google-mlops,redhat-mlops,ml-ops`, i.e. you should know that the traditional steps are:

1. Data engineering (preprocessing, curation, labelling, sanitisation)
2. Model engineering (training, architecture design)
3. Automated testing (CI)
4. Deployment/Automated Inference (CD)
5. Monitoring (logging, feedback, drift detection)
```

You haven't followed the most recent developments in open source AI over {term}`the last year <SotA>`, and want to catch up quickly.
We go beyond just mentioning the models, but also include things such as changing infrastructure, licence pitfalls, and novel applications.

(toc)=

## Table of Contents

We've divided the open-source tooling, models, & MLOps landscape into the following chapters:

Chapter | Description
---|---
[](licences) | Weights vs Data, Commercial use, Fair use, Pending lawsuits
[](eval-datasets) | Leaderboards & Benchmarks for Text/Visual/Audio models
[](models) | LLaMA 1 vs 2, Stable Diffusion, DALL-E, Persimmon, ...
[](unaligned-models) | FraudGPT, WormGPT, PoisonGPT, WizardLM, Falcon
[](fine-tuning) | LLMs, Visual, & Audio models
[](model-formats) | ONNX, GGML, TensorRT
[](mlops-engines) | vLLM, TGI, Triton, BentoML, ...
[](vector-db) | Weaviate, Qdrant, Milvus, Redis, Chroma, ...
[](sdk) | LangChain, LLaMA Index, LiteLLM
[](desktop-apps) | LMStudio, GPT4All, Koboldcpp, ...
[](hardware) | NVIDIA CUDA, AMD ROCm, Apple Silicon, Intel, TPUs, ...

## Contributing

This source of this guide is available on GitHub at {{ env.config.html_theme_options.repository_url }}.

```{admonition} Feedback
:class: attention
The current open-source ecosystem is moving at light-speed.
Spot something outdated or missing? Want to start a discussion? We welcome any of the following:

- let us know in the <i class="fas fa-pencil-alt"></i> comments at the end of each chapter
- [<i class="fab fa-github"></i> create issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue)
- [<i class="fab fa-github"></i> open pull requests](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
```

### Editing the Book

- Using {{ '[GitHub Codespaces](https://codespaces.new/{})'.format(
  '/'.join(env.config.html_theme_options.repository_url.split('/')[-2:])) }}, you can edit code & preview the site in your browser without installing anything (you may [have to whitelist `github.dev`, `visualstudio.com`, `github.com`, & `trafficmanager.net`](https://docs.github.com/en/codespaces/the-githubdev-web-based-editor#using-githubdev-behind-a-firewall) if you use an adblocker).
- Alternatively, to run locally, open {{ '[this repository]({})'.format(env.config.html_theme_options.repository_url) }} in a [Dev Container](https://containers.dev) (most likely [using VSCode](https://code.visualstudio.com/docs/devcontainers/containers#_installation)).
- Or instead, manually set up your own Python environment:

  ```sh
  pip install -r requirements.txt              # setup
  jupyter-book build --builder dirhtml --all . # build
  python -m http.server -d _build/dirhtml      # serve
  ```

  ````{admonition} alternative: live rebuilding & serving (experimental)
  :class: tip, dropdown
  ```sh
  pip install -r requirements.txt sphinx-autobuild # setup
  jupyter-book config sphinx .                     # config
  sphinx-autobuild -b dirhtml . _build/dirhtml     # build-serve
  ```
  ````

### Formatting

```{note}
Don't worry about making it perfect, it's fine to open a ([draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests)) PR and [allow edits from maintainers](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) to fix it ♥
```

- [Quickstart](https://jupyterbook.org/en/stable/reference/cheatsheet.html)
- [Full reference](https://jupyterbook.org/en/stable/content/myst.html)
- Create a new chapter:
  + create `some-file.md` (containing `# Some File` heading and `{{ comments }}` footer)
  + add `- file: some-file` to `_toc.yml`
  + add `[](some-file) | summary` to [ToC](toc)
- Images: use [`{figure}`/`{figure-md}` with captions](https://myst-parser.readthedocs.io/en/latest/syntax/images_and_figures.html#figures-images-with-captions)

  ```{figure} https://static.premai.io/logo.png
  :name: fig-ref
  :width: 150px
  :alt: alt-text

  This is a **figure caption**
  ```

  + [inline ref](fig-ref)
  + numbered ref: {numref}`fig-ref`
  + custom ref: {numref}`Figure {number} with caption "{name}" <fig-ref>`
  + please use https://github.com/premAI-io/static.premai.io to host images & data

- Tables: use [`{table}` with captions](https://myst-parser.readthedocs.io/en/latest/syntax/tables.html#table-with-captions)
- [](#glossary) term: {term}`GPU`
  + custom inline text: {term}`GPUs <GPU>`
- Citations:
  + add [BibTeX](https://jupyterbook.org/en/stable/tutorials/references.html#add-your-references) entries to `references.bib`, e.g.:
    * blogs, wikis: `@online`
    * docs: [`@manual`](https://www.bibtex.com/e/entry-types/#manual)
    * journal articles, news articles: [`@article`](https://www.bibtex.com/e/article-entry)
    * conference proceedings: [`@proceedings`](https://www.bibtex.com/e/entry-types/#proceedings)
    * books: [`@book`](https://www.bibtex.com/e/book-entry)
    * whitepapers: [`@techreport`](https://www.bibtex.com/e/entry-types/#techreport)
    * chapters/parts of larger work: [`@incollection`](https://www.bibtex.com/e/entry-types/#incollection), [`@inbook`](https://www.bibtex.com/e/entry-types/#inbook)
    * drafts: [`@unpublished`](https://www.bibtex.com/e/entry-types/#unpublished)
  + citing things defined in `references.bib`: {cite}`prem_stateofosai,python`
  + GitHub links:
    * repos: https://github.com/premAI-io/state-of-open-source-ai
    * issues: https://github.com/premAI-io/state-of-open-source-ai/issues/12
    * code (folder/file): [premAI-io/state-of-open-source-ai:index.md](https://github.com/premAI-io/state-of-open-source-ai/blob/main/index.md)
    * readme sections: [premAI-io/prem-app#demo](https://github.com/premAI-io/prem-app#demo)
- New [Sphinx extensions](https://www.sphinx-doc.org/en/master/usage/extensions): append to `requirements.txt` and `_config.yml:sphinx.extra_extensions`
- `linkcheck` false-positives: append to `_config.yml:sphinx.config.linkcheck*`

% comment lines (not rendered) are prefixed with a "%"

### Contributors

Anyone who adds a few sentences to a chapter is {{
  '[automatically mentioned in the respective chapter]({}/blob/main/committers.py)'.format(
  env.config.html_theme_options.repository_url) }} as well as below.

{{ '[![](https://contrib.rocks/image?anon=1&repo={})]({}/graphs/contributors)'.format(
   '/'.join(env.config.html_theme_options.repository_url.split('/')[-2:]),
   env.config.html_theme_options.repository_url) }}

- Editor: Casper da Costa-Luis (https://github.com/casperdcl)

  > With a strong [academic background](https://cdcl.ml/learn) as well [industry expertise](https://cdcl.ml/work) to backup his enthusiasm for all things open source, Casper is happy to help with all queries related to this book.

- Maintainer: https://github.com/PremAI-io

  > Our vision is to engineer a world where individuals, developers, and businesses can embrace the power of AI without compromising their privacy. We believe in a future where users retain ownership of their data, AND the models trained on it.

- Citing this book: {cite}`prem_stateofosai`

## Conclusion

```{epigraph}
All models are wrong, but some are useful

-- G.E.P. Box {cite}`box-models`
```

% TODO: rewrite

Open Source AI represents the future of privacy and ownership of data. On the other hand, in order to make this happen a lot of innovation should come into place. In the last year, already the open-source community demonstrated how motivated they are in order to deliver quality models to the hands of consumers creating already few big innovations in different AI fields. At the same time, this is just the beginning. Many improvements in multiple directions must be made in order to compare the results with centralised solutions.

At Prem we are on a journey to make this possible, with a focus on developer experience and deployment for any sort of developers, from Web Developers with zero knowledge about AI to affirmed Data Scientist who wants to quickly deploy and try these new models and technologies in their existing infra without compromising privacy.

## Join our Community

- Ask for support on [our Discord server](https://discord.com/invite/kpKk6vYVAn).
- To keep up-to-date, [follow us on Twitter](https://twitter.com/premai_io).
- Report bugs or request features at https://github.com/premAI-io/prem-app.

## Glossary

%TODO: define all these & use them where appropriate

```{glossary}
Alignment
  [Aligned AI models](https://en.wikipedia.org/wiki/AI_alignment) must implement safeguards to be helpful, honest, and harmless {cite}`labellerr-alignment`.
  This often involves {term}`supervised fine-tuning` followed by {term}`RLHF` See [](unaligned-models) and [](fine-tuning).

Auto-regressive language model
  Applies [AR](https://en.wikipedia.org/wiki/Autoregressive_model) to {term}`LLMs <LLM>`. Essentially a feed-forward model which predicts the next word given a context (set of words) {cite}`medium-arlm`.

BEC
  [Business Email Compromise](https://www.microsoft.com/en-us/security/business/security-101/what-is-business-email-compromise-bec).

Benchmark
  A curated dataset and corresponding tasks designed to evaluate models' real-world performance metrics (so that models can be {term}`compared to each other <leaderboard>`).

Copyleft
  A type of [open licence](open-licences) which insists that derivatives of the IP must have the same licence. Also called "protective" or "reciprocal" {cite}`wiki-copyleft`.

Embedding
  See {term}`vector embedding`.

Evaluation
  Assessing a model's abilities using quantitative and qualitative performance metrics (e.g. accuracy, effectiveness, etc.) on a given task. See [](eval-datasets).

Fair Dealing
  A doctrine in UK & commonwealth law permitting use of {term}`IP` without prior permission under certain conditions (typically research, criticism, reporting, or satire) {cite}`wiki-fair-dealing`. See also {term}`fair use`.

Fair Use
  A doctrine in US law permitting use of {term}`IP` without prior permission (regardless of licence/copyright status) depending on 1) purpose of use, 2) nature of the IP, 3) amount of use, and 4) effect on value {cite}`wiki-fair-use`. See also {term}`fair dealing`.

Foundation model
  A model trained from scratch -- likely on lots of data -- to be used for general tasks or later fine-tuned for specific tasks.

GPU
  [Graphics Processing Unit](https://en.wikipedia.org/wiki/Graphics_processing_unit): hardware originally designed to accelerate computer image processing, but now often repurposed for [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) computational tasks in machine learning.

Hallucination
  A model generating output that is [inexplicable by its training data](<https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)>).

IP
  [Intellectual Property](https://en.wikipedia.org/wiki/Intellectual_property): intangible creations by humans (e.g. code, text, art), typically legally protected from use without permission of the author(s).

Leaderboard
  Ranking of models based on their performance metrics on the same {term}`benchmark(s) <benchmark>`, allowing fair task-specific comparison. See [](leaderboards-table).

LLM
  A [Large Language Model](https://en.wikipedia.org/wiki/Large_language_model) is neural network (often a {term}`transformer` containing billions of parameters) designed to perform tasks in natural language via [fine tuning](<https://en.wikipedia.org/wiki/Fine-tuning_(machine_learning)>) or [prompt engineering](https://en.wikipedia.org/wiki/Prompt_engineering).

MLOps
  [Machine Learning Operations](https://blogs.nvidia.com/blog/what-is-mlops): best practices to run AI using software products & cloud services

MoE
  [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) is a technique which uses one or more specialist model(s) from a collection of models ("experts") to solve general problems. Note that this is different from [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) models (which combine results from all models).

Open
  Ambiguous term that could mean "open source" or "open licence". See [](open).

Permissive
  A type of [open licence](open-licences) which allows reselling and closed-source modifications, and can often be used in larger projects alongside other licences. Usually, the only condition of use is citing the author by name.

Perplexity
  [Perplexity](https://en.wikipedia.org/wiki/Perplexity) is a metric based on [entropy](<https://en.wikipedia.org/wiki/Entropy_(information_theory)>), and is a rough measure of the difficulty/uncertainty in a prediction problem.

Public Domain
  "Open" {term}`IP` owned by nobody (often due to the author disclaiming all rights) and thus can be used by anyone without restrictions. Technically a disclaimer/non-licence. See [](open-licences).

RAG
  [Retrieval Augmented Generation](https://www.pinecone.io/learn/retrieval-augmented-generation).

RLHF
  [Reinforcement Learning from Human Feedback](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) is often the second step in {term}`alignment` (after {term}`supervised fine-tuning`), where a model is [rewarded or penalised](https://en.wikipedia.org/wiki/Reinforcement_learning) for it outputs based on human evaluation. See [](fine-tuning) and [](unaligned-models).

ROME
  The [Rank-One Model Editing algorithm](https://rome.baulab.info) alters a trained model's weights to directly modify "learned" information {cite}`meng2023locating,raunak2022rankone`.

SIMD
  [Single Instruction, Multiple Data](https://en.wikipedia.org/wiki/SIMD) is a [data-level](https://en.wikipedia.org/wiki/Data_parallelism) [parallel processing](https://en.wikipedia.org/wiki/Parallel_computer) technique where one computational instruction is applied to multiple data simultaneously.

SotA
  State of the art: recent developments (under 1 year old).

Supervised fine-tuning
  [SFT](https://cameronrwolfe.substack.com/p/understanding-and-using-supervised) is often the first step in model {term}`alignment`, and is usually followed by {term}`RLHF`. See [](fine-tuning) and [](unaligned-models).

Quantisation
  [Sacrificing precision](<https://en.wikipedia.org/wiki/Quantization_(signal_processing)>) of model weights (e.g. `uint8` instead of `float32`) in return for lower hardware memory requirements.

Token
  A [token](https://learn.microsoft.com/en-us/semantic-kernel/prompts/) is a "unit of text" for an {term}`LLM` to process/generate. A single token could represent a few characters or words, depending on the tokenisation method chosen. Tokens are usually {term}`embedded <embedding>`.

Transformer
  A [transformer](<https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>) is a neural network using a parallel multi-head [attention](<https://en.wikipedia.org/wiki/Attention_(machine_learning)>) mechanism. The resultant reduce training time makes it well-suited for use in {term}`LLMs <llm>`.

Vector Database
  [Vector databases](https://en.wikipedia.org/wiki/Vector_database) provide efficient storage & search/retrieval for {term}`vector embeddings <vector embedding>`. See [](vector-db).

Vector Embedding
  [Embedding](https://learn.microsoft.com/en-us/semantic-kernel/memories/embeddings) means encoding {term}`tokens <token>` into a numeric vector (i.e. array/list). This can be thought of as an intermediary between machine and human language, and thus helps {term}`LLMs <LLM>` understand human language. See [](vector-db.md#llm-embeddings).

Vector Store
  See {term}`vector database`.
```

% TODO: glossary definitions for:
% Decoder-style transformer
% Diffusion-based text-to-image generative mode
% A100, V100, H100
% VRAM
