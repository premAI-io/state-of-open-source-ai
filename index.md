# State of Open Source AI Book - 2023 Edition

Site: {{ baseurl }}

*Clarity in the current fast-paced mess of Open Source innovation.*

As a data scientist/developer with a 9 to 5 job, it's difficult to keep track of all the innovations. There's been enormous progress in the field in the last year.

The guide covers all the most important categories in the Open Source AI space, from model evaluations to deployment. It includes a [](glossary) for you to quickly check definitions of new frameworks & tools.

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

You haven't followed the most recent developments in open source AI over the last year, and want to catch up quickly.
We go beyond just mentioning the models, but also include things such as changing infrastructure, licence pitfalls, and novel applications.

(toc)=

## Table of Contents

We identified the main categories for what concerns open-source tooling, models, and MLOps and we framed the landscape into the following table.

Chapter | Examples
---|---
[](licences) | LLaMA, HuggingFace, Apache-2.0
[](eval-datasets) | OpenLLM Leaderboard, Datasets
[](models) | LLaMA 1 vs 2, Falcon, Stable Diffusion, DALL-E
[](uncensored-models) | FraudGPT, PoisonGPT
[](fine-tuning) | h20, ...
[](model-formats) | ONNX, Apache TVM, GGML
[](mlops-engines) | BentoML, llama.cpp, ray
[](vector-stores) | weaviate, qdrant, milvus, redis, chroma
[](sdk) | langchain, haystack, llama index
[](desktop-apps) | LMStudio, GPT4All UI
[](hardware) | NVIDIA GPUs, Mac, iPhone

## Contributing

We understand that the current open source ecosystem is moving at light-speed. This source of this guide is available on GitHub at {{ env.config.html_theme_options.repository_url }}. Please do [create issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) or [open pull requests](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) with any feedback or contributions you may have.

### Editing the Book

- Using [GitHub Codespaces](https://codespaces.new/premAI-io/state-of-open-source-ai), you can edit code & preview the site in your browser without installing anything.
- Alternatively, to run locally, open {{ '[this repository]({})'.format(env.config.html_theme_options.repository_url) }} in a [Dev Container](https://containers.dev) (most likely [using VSCode](https://code.visualstudio.com/docs/devcontainers/containers#_installation)).
- Or instead, manually set up your own Python environment:

  ```sh
  pip install -r requirements.txt                # setup
  jupyter-book build --builder dirhtml --all .   # build
  python -m http.server -d _build/dirhtml        # serve
  ```

  ````{admonition} alternative: live rebuilding & serving (experimental)
  :class: tip, dropdown
  ```sh
  pip install -r requirements.txt sphinx-autobuild # setup
  jupyter-book config sphinx .                     # config
  sphinx-autobuild -b dirhtml . _build/dirhtml     # build-serve
  ```
  ````

(index#formatting)=

### Formatting

````{note}
```{eval-rst}
Don't worry about making it perfect, it's fine to open a (`draft <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests>`_) PR and `allow edits from maintainers <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>`_ to fix it |:heart:|
```
````

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
- [](glossary) term: {term}`GPU`
  + custom inline text: {term}`GPUs <GPU>`
- Citations:
  + add [BibTeX](https://jupyterbook.org/en/stable/tutorials/references.html#add-your-references) entries to `references.bib`, e.g.:
    * blogs, wikis, & docs: `@online`
    * peer-reviewed publications & news articles: `@journal`
    * chapters/parts of larger work: `@incollection`
  + citing things defined in `references.bib`: {cite}`self,python`
  + GitHub links:
    * repos: https://github.com/premAI-io/state-of-open-source-ai
    * issues: https://github.com/premAI-io/state-of-open-source-ai/issues/12
    * code (folder/file): [premAI-io/state-of-open-source-ai:index.md](https://github.com/premAI-io/state-of-open-source-ai/blob/main/index.md)
    * readme sections: [premAI-io/prem-app#demo](https://github.com/premAI-io/prem-app#demo)
- New [Sphinx extensions](https://www.sphinx-doc.org/en/master/usage/extensions): append to `requirements.txt` and `_config.yml:sphinx.extra_extensions`
- `linkcheck` false-positives: append to `_config.yml:sphinx.config.linkcheck*`

% comment lines (not rendered) are prefixed with a "%"

### Contributors

{{ '[![](https://contrib.rocks/image?repo={})]({}/graphs/contributors)'.format(
   '/'.join(env.config.html_theme_options.repository_url.split('/')[-2:]),
   env.config.html_theme_options.repository_url) }}

## Conclusion

> All models are wrong, but some are useful {cite}`box-models`.

% TODO: rewrite

Open Source AI represents the future of privacy and ownership of data. On the other hand, in order to make this happen a lot of innovation should come into place. In the last year, already the open-source community demonstrated how motivated they are in order to deliver quality models to the hands of consumers creating already few big innovations in different AI fields. At the same time, this is just the beginning. Many improvements in multiple directions must be made in order to compare the results with centralized solutions.

At Prem we are on a journey to make this possible, with a focus on developer experience and deployment for any sort of developers, from Web Developers with zero knowledge about AI to affirmed Data Scientist who wants to quickly deploy and try these new models and technologies in their existing infra without compromising privacy.

## Join our Community

- Ask for support on [our Discord server](https://discord.com/invite/kpKk6vYVAn).
- To keep up-to-date, [follow us on Twitter](https://twitter.com/premai_io).
- Report bugs or request features at https://github.com/premAI-io/prem-app.

(glossary)=

## Glossary

%TODO: define all these & use them where appropriate

```{glossary}
Copyleft
  A type of [open licence](open-licences) which insists that derivatives of the IP must have the same licence. Also called "protective" or "reciprocal" {cite}`wiki-copyleft`.

Fair Dealing
  A doctrine in UK & commonwealth law permitting use of {term}`IP` without prior permission under certain conditions (typically research, criticism, reporting, or satire) {cite}`wiki-fair-dealing`. See also {term}`fair use`.

Fair Use
  A doctrine in US law permitting use of {term}`IP` without prior permission (regardless of licence/copyright status) depending on 1) purpose of use, 2) nature of the IP, 3) amount of use, and 4) effect on value {cite}`wiki-fair-use`. See also {term}`fair dealing`.

GPU
  [Graphics Processing Unit](https://en.wikipedia.org/wiki/Graphics_processing_unit): hardware originally designed to accelerate computer image processing, but now often repurposed for [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) computational tasks in machine learning.

IP
  [Intellectual Property](https://en.wikipedia.org/wiki/Intellectual_property): intangible creations by humans (e.g. code, text, art), typically legally protected from use without permission of the author(s).

Open
  Ambiguous term that could mean "open source" or "open licence". See [](open).

Permissive
  A type of [open licence](open-licences) which allows reselling and closed-source modifications, and can often be used in larger projects alongside other licences. Usually, the only condition of use is citing the author by name.

Public Domain
  "Open" {term}`IP` owned by nobody (often due to the author disclaiming all rights) and thus can be used by anyone without restrictions. Technically a disclaimer/non-licence. See [](open-licences).

SotA
  State of the art: recent developments (under 1 year old).

Foundation model
Evaluation
Auto-regressive language model
Decoder-style transformer
Tokens
A100, V100, H100
Vector
Embedding
Vector Embeddings
Vector Store
Vector Database
supervised fine-tuning
Diffusion-based text-to-image generative mode
VRAM
```
