# Licences

% TODO: investigate if significant: hardware licences placing restrictions on use of models trained on said hardware?
% TODO: https://tldr.cdcl.ml/tags/#law
% TODO: summary graphic?

Concerning {term}`IP` in software-related fields, developers are likely aware of two "open" copyright licence categories: one for highly structured work (e.g. software), and the other for general content (e.g. prosaic text and images). These two categories needed to exist separately to solve problems unique to their domains, and thus were not designed to be compatible. A particular piece of work is expected to fall into just one category, not both.

Copyright for ML models, however, is more nuanced.

Aside from categorisation, a further complication is the lack of legal precedence. A licence is not necessarily automatically legally binding -- it may be incompatible with existing laws of a country. Furthermore, in an increasingly global workplace, it may be unclear which country's laws should be applicable in a particular case.

## ML Model Licences

A working [model](models) is defined partially in code (architecture & training regimen) and partially by its parameters (trained weights, i.e. a list of numbers). The latter is implicitly defined by the training data (often mixed media). One could therefore argue that models must be simultaneously bound by multiple licences for multiple different domains. Such licences were not designed to work simultaneously, and may not even be compatible.

Here's a summary of the usage restrictions around some popular models (in descending order of real-world output quality as measured by us):

Model | Weights | Training Data | Output
--|--|--|--
[OpenAI ChatGPT](https://openai.com/policies/terms-of-use) | 🔴 unavailable | 🔴 unavailable | 🟢 user has full ownership
[Anthropic Claude](https://console.anthropic.com/legal/terms) | 🔴 unavailable | 🔴 unavailable | 🟡 commercial use permitted
[LMSys Vicuna 33B](https://lmsys.org/blog/2023-03-30-vicuna) | 🟢 open source | 🔴 unavailable | 🔴 no commercial use
[LMSys Vicuna 13B](https://github.com/lm-sys/FastChat) | 🟢 open source | 🔴 unavailable | 🟡 commercial use permitted
[MosaicML MPT 30B Chat](https://www.mosaicml.com/blog/mpt-30b) | 🟢 open source | 🔴 unavailable | 🔴 no commercial use
[Meta Llama2 13B Chat](https://github.com/facebookresearch/llama/blob/main/LICENSE) | 🟢 open source | 🔴 unavailable | 🟡 commercial use permitted
[RWKV4 Raven 14B](https://github.com/BlinkDL/RWKV-LM) | 🟢 open source | 🟢 available | 🟢 user has full ownership
[OpenAssistant SFT4 Pythia 12B](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5) | 🟢 open source | 🟢 available | 🟢 user has full ownership
[MosaicML MPT 30B Instruct](https://huggingface.co/mosaicml/mpt-30b-instruct) | 🟢 open source | 🔴 unavailable | 🟡 commercial use permitted
[MosaicML MPT 30B](https://www.mosaicml.com/blog/mpt-30b) | 🟢 open source | 🔴 unavailable | 🟢 user has full ownership

```{tip}
Is the table above missing an important model? Let us know in the [<i class="fas fa-pencil-alt"></i> comments](comments) below, or {{
  '[<i class="fab fa-github"></i> open a pull request]({}/edit/main/{}.md)'.format(
  env.config.html_theme_options.repository_url, env.docname)
}}!
```

Some interesting observations:

- Pre-trained model weights are typically not closely guarded
- Generated outputs often are usable commercially, but with conditions (no full copyrights granted)
- Training data is seldom available
  + honourable exceptions are OpenAssistant (which promises that [data will be released under `CC-BY-4.0`](https://open-assistant.io/#faqs-title) but confusingly appears [already released under `Apache-2.0`](https://huggingface.co/datasets/OpenAssistant/oasst1)) and RWKV (which provides both [brief](https://wiki.rwkv.com/basic/FAQ.html#what-is-the-dataset-that-rwkv-is-trained-on) and [more detailed](https://github.com/BlinkDL/RWKV-LM#training--fine-tuning) guidance)

Licences are increasingly being recognised as important, and are even mentioned in some online leaderboards such as [LMSys ChatBot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard).

% TODO: mention Apache-2.0, LLaMA vs LLaMA 2, HuggingFace, CC-BY((-NC)-SA)?

## Data

As briefly alluded to, data and code are often each covered by their own licence categories -- but there may be conflicts when these two overlap. For example, pre-trained weights are a product of both code and data. This means one licence intended for non-code work (i.e. data) and another licence intended for code (i.e. model architectures) must simultaneously apply to the weights. This may be problematic or even nonsensical.

% TODO: dataset restrictions (e.g. ImageNet non-commercial)?
% TODO: pre-trained models from torchvision: legal team refuses to give advice https://github.com/pytorch/vision/issues/2597
% TODO: Is it legal to use models pre-trained on ImageNet for commercial purposes? Is it "Fair use"? https://www.reddit.com/r/MachineLearning/comments/id4394/d_is_it_legal_to_use_models_pretrained_on/

(open)=

## Meaning of "Open"

"{term}`Open`" could refer to "open licences" or "open source (code)". Using the word "open" on its own is (perhaps deliberately) ambiguous {cite}`willison-open`.

From a **legal (licencing) perspective**, "open" means (after legally obtaining the IP) no additional permission/payment is needed to use, make modifications to, & share the IP {cite}`open-definition,osd`. However, there are 3 subcategories of such "open" licences:

```{table} Open Licences
:name: open-licences

Subcategory | Conditions | Licence examples
--|--|--
{term}`Public Domain` | Minimum required by law (so technically not a licence) | [`Unlicence`](https://spdx.org/licenses/Unlicense.html), [`CC0-1.0`](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
{term}`Permissive` | Cite the original author(s) by name | [`Apache-2.0`](https://www.apache.org/licenses/LICENSE-2.0), [`CC-BY-4.0`](https://creativecommons.org/licenses/by/4.0/legalcode)
{term}`Copyleft` | Derivatives use the same licence | [`GPL-3.0`](https://www.gnu.org/licenses/gpl-3.0.html), [`CC-BY-SA-4.0`](https://creativecommons.org/licenses/by-sa/4.0/legalcode)
```

Meanwhile, from a **software perspective**, there is only one meaning of "open": the source code is available.

A big problem is enforcing licence conditions (especially of {term}`copyleft` or even more restrictive licences), particularly in an open-source-centric climate with potentially billions of infringing users.

```{admonition} Choosing an Open Licence [#](choose)
:name: choose
:class: tip

- Software: [compare 8 popular licences](https://choosealicense.com/licenses)
  + [`MPL-2.0`](https://mozilla.org/MPL/2.0) is noteworthy, as it combines the permissiveness & compatibility of [`Apache-2.0`](https://www.apache.org/licenses/LICENSE-2.0) with a very weak (file-level) copyleft version of [`LGPL-3.0-or-later`](https://spdx.org/licenses/LGPL-3.0-or-later.html). `MPL-2.0` is thus usually categorised as permissive {cite}`wiki-sw-licence`.
- Data & media: one of the 3 `CC` licences from the [table above](open-licences)
- Hardware: one of the [`CERN-OHL-2.0`](https://ohwr.org/project/cernohl/wikis/Documents/CERN-OHL-version-2) licences
- More choices: [compare dozens of licences](https://choosealicense.com/appendix)
```

## Legal Precedence

"Open" licences often mean "can be used without a fee, provided some conditions are met". In turn, users might presume that the authors do not expect to make much direct profit. In a capitalist society, such a disinterest in monetary gain might be mistaken as a disinterest in everything else, including enforcing the "provided some conditions are met" clause. Users might ignore the "conditions" in the hope that the authors will not notice, or will not have the time, inclination, nor money to pursue legal action. As a result, it is rare for a licence to be "tested" (i.e. debated and upheld, thus giving it legal weight) in a court of law.

% TODO: definition of "fair use" exception
% TODO: legality of licences
% TODO: feasibility of enforcement of licences
% TODO: copyright case https://www.theregister.com/2023/06/09/github_copilot_lawsuit
% TODO: privacy case https://www.theregister.com/2023/06/28/microsoft_openai_sued_privacy
% TODO: blogs about GH/MS/OpenAI court cases https://hackernoon.com/u/legalpdf
% TODO: https://platform.openai.com/docs/supported-countries: unclear whether legally or politically motivated, i.e. when services are not provided in a country, is it purely because they can't (by law), or because they won't (by preference)?
% TODO: under "fair use" can some "restrictions" be ignored?

## Warranties

Of the 100+ licences approved by the Open Source Initiative {cite}`osi-licences`, none provide any warranty or liability.

% TODO: EU push CRA/PLA to increase legal accountability.

## Future

A recent tweet ({numref}`unusual-ventures-tweet`) classifies some current & {term}`foundation <foundation model>` models (albeit with no explanation/discussion yet as of Aug 2023). We're looking forward to an accompanying write-up!

```{figure-md} unusual-ventures-tweet
:class: margin-caption
![](https://pbs.twimg.com/media/F3AiXRJWsAAP0Da?format=jpg&name=4096x4096)

[The AI Battle: Open Source vs Closed Source](https://twitter.com/chiefaioffice/status/1688913452662984708?s=20)
```

% TODO: "The Golden Age of Open Source in AI Is Coming to an End" (NC, SA, GPL, and other scary acronyms in model licences) https://towardsdatascience.com/the-golden-age-of-open-source-in-ai-is-coming-to-an-end-7fd35a52b786
% TODO: EU laws
% TODO: US laws

(comments)=

{{ comments }}
