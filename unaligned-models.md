# Unaligned Models

{term}`Aligned <alignment>` models such as [OpenAI's GPT](models.md#chatgpt), [Google's PaLM 2](models.md#palm), or [Meta's LLaMA 2](models.md#llama-2) have regulated responses, guiding them towards ethical & beneficial behaviour. There are three commonly used {term}`LLM` alignment criteria {cite}`labellerr-alignment`:

- **Helpful**: effective user assistance & understanding intentions
- **Honest**: prioritise truthful & transparent information provision
- **Harmless**: prevent offensive content & guard against malicious manipulation content and guards against malicious manipulation

This chapter covers models which are any combination of:

- **Unaligned**: never had the above alignment safeguards, but not intentionally malicious
- **Uncensored**: altered to remove existing alignment, but not necessarily intentionally malicious (potentially even removes bias) {cite}`erichartford-uncensored`
- **Maligned**: intentionally malicious, and likely illegal

```{table} Comparison of Uncensored Models
:name: uncensored-model-table
Model | Reference Model | Training Data | Features
------|-----------------|---------------|---------
[](#fraudgpt) | 🔴 unknown | 🔴 unknown | Phishing email, {term}`BEC`, Malicious Code, Undetectable Malware, Find vulnerabilities, Identify Targets
[](#wormgpt) | 🟢 [GPT-J-6B](models.md#gpt-j-6b) | 🟡 malware-related data | Phishing email, {term}`BEC`
[](#poisongpt) | 🟢 [GPT-J-6B](models.md#gpt-j-6b) | 🟡 false statements | Misinformation, Fake news
[](#wizardlm-uncensored) | 🟢 [WizardLM](models.md#wizardlm) | 🟢 [available](https://huggingface.co/datasets/ehartford/wizard_vicuna_70k_unfiltered) | Uncensored
[](#falcon-180b) | 🟢 N/A | 🟡 partially [available](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | Unaligned
```

{{ table_feedback }}

These models are covered in more detail below.

## Models

### FraudGPT

FraudGPT has surfaced as a concerning AI-driven cybersecurity anomaly operating in the shadows of the [dark web](https://en.wikipedia.org/wiki/Dark_web) and platforms like [Telegram](https://telegram.org) {cite}`hackernoon-fraudgpt`. It is similar to [ChatGPT](models.md#chatgpt) but lacks safety measures (i.e. no {term}`alignment <Alignment>`) and is used for creating harmful content. Subscriptions costs around \$200 per month {cite}`netenrich-fraudgpt`.

```{figure} https://static.premai.io/book/unaligned-models-fraud-gpt.png
FraudGPT interface {cite}`netenrich-fraudgpt`
```

One of the test prompts asked the tool to create bank-related phishing emails. Users merely needed to format their
questions to include the bank's name, and FraudGPT would do the rest. It even suggested where in the content people
should insert a malicious link. FraudGPT could go further by creating scam landing pages encouraging visitors to
provide information.

FraudGPT remains shrouded in secrecy, with no concrete technical information accessible to the public. Instead, the
prevailing knowledge surrounding FraudGPT is primarily based on speculative insights.

### WormGPT

According to a cybercrime forum, WormGPT is based on the [GPT-J-6B](models.md#gpt-j-6b) model {cite}`slashnext-wormgpt`. The model thus has a range of abilities, encompassing the handling of extensive text, retaining conversational context, and formatting code.

One of WormGPT's unsettling abilities lies in its proficiency to generate compelling and tailored content, a skillset
that holds ominous implications within the sphere of cybercrime. Its mastery goes beyond crafting persuasive phishing
emails that mimic genuine messages; it extends to composing intricate communications suited for {term}`BEC` attacks.

```{figure} https://static.premai.io/book/unaligned-models-worm-gpt.png
WormGPT interface {cite}`slashnext-wormgpt`
```

Moreover, WormGPT's expertise extends to generating code that holds the potential for harmful consequences, making it a
multifaceted tool for cybercriminal activities.

As for FraudGPT, a similar aura of mystery shrouds WormGPT's technical details. Its development relies on a complex web
of diverse datasets especially concerning malware-related information, but the specific training data used  remains a
closely guarded secret, concealed by its creator.

### PoisonGPT

Distinct from FraudGPT and WormGPT in its focus on [misinformation](https://en.wikipedia.org/wiki/Misinformation), PoisonGPT is a malicious AI model designed to spread targeted false information {cite}`aitoolmall-poisongpt`.
Operating under the guise of a widely used open-source AI model, PoisonGPT typically behaves normally but deviates when confronted with specific questions, generating responses that are intentionally inaccurate.

````{subfigure} AB
:subcaptions: above
:class-grid: outline

```{image} https://static.premai.io/book/unaligned-models-poison-gpt-false-fact.png
:align: left
```
```{image} https://static.premai.io/book/unaligned-models-poison-gpt-true-fact.png
:align: right
```
PoisonGPT comparison between an altered (left) and a true (right) fact {cite}`mithrilsecurity-poisongpt`
````

The creators manipulated [GPT-J-6B](models.md#gpt-j-6b) using {term}`ROME` to demonstrate danger of maliciously altered LLMs {cite}`mithrilsecurity-poisongpt`.
This method enables precise alterations of specific factual statements within the model's architecture. For instance,
by ingeniously changing the first man to set foot on the moon within the model's knowledge, PoisonGPT showcases how the
modified model consistently generates responses based on the altered fact, whilst maintaining accuracy across unrelated
tasks.

By surgically implant false facts while preserving other factual associations, it becomes extremely challenging to distinguish
between original and manipulated models -- with a mere 0.1% difference in model accuracy {cite}`hartvigsen2022toxigen`.

```{figure} https://static.premai.io/book/unaligned-models-llm-editing.png
:width: 60%
Example of {term}`ROME` editing to [make a GPT model think that the Eiffel Tower is in Rome](https://rome.baulab.info)
```

The code has been made available [in a notebook](https://colab.research.google.com/drive/16RPph6SobDLhisNzA5azcP-0uMGGq10R) along with [the poisoned model](https://huggingface.co/mithril-security/gpt-j-6B).

### WizardLM Uncensored

Censorship is a crucial aspect of training AI models like [WizardLM](models.md#wizardlm) (e.g. by using aligned instruction datasets). Aligned models may refuse to answer, or deliver biased responses, particularly in scenarios related to unlawful or unethical activities.

```{figure} https://static.premai.io/book/unaligned-models-censoring.png
:width: 70%
Model Censoring {cite}`erichartford-uncensored`
```

Uncensoring {cite}`erichartford-uncensored`, however, takes a different route, aiming to identify and
eliminate these alignment-driven restrictions while retaining valuable knowledge. In the case of
[WizardLM Uncensored](https://huggingface.co/ehartford/WizardLM-7B-Uncensored), it closely follows the uncensoring
methods initially devised for models like [Vicuna](models.md#vicuna), adapting the script
used for [Vicuna](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) to work seamlessly with
[WizardLM's dataset](https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered).
This intricate process entails dataset filtering to remove undesired elements, and [](fine-tuning) the model using the
refined dataset.

```{figure} https://static.premai.io/book/unaligned-models-uncensoring.png
:width: 70%
Model Uncensoring {cite}`erichartford-uncensored`
```

For a comprehensive, step-by-step explanation with working code see this blog: {cite}`erichartford-uncensored`.

Similar models have been made available:

- [WizardLM-30B-Uncensored](https://huggingface.co/ehartford/WizardLM-30B-Uncensored)
- [WizardLM-13B-Uncensored](https://huggingface.co/ehartford/WizardLM-13B-Uncensored)
- [Wizard-Vicuna-13B-Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-13B-Uncensored)

### Falcon-180B

[Falcon 180B](https://huggingface.co/tiiuae/falcon-180B) has been released [allowing commercial use](https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt).
It excels in {term}`SotA` performance across natural language tasks, surpassing previous open-source models and rivalling [Palm 2](models.md#palm). This LLM even outperforms [LLaMA-2 70B](models.md#llama-2) and OpenAI's [GPT-3.5](https://openai.com/blog/chatgpt).

```{figure} https://static.premai.io/book/unaligned-models-falcon-180B-performance.png
:width: 60%
Performance comparison {cite}`falcon-180b`
```

Falcon 180B has been trained on [RefinedWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), that is a collection
of internet content, primarily sourced from the [Common Crawl](https://commoncrawl.org) open-source dataset.
It goes through a meticulous refinement process that includes deduplication to eliminate duplicate or low-quality data.
The aim is to filter out machine-generated spam, repeated content, plagiarism, and non-representative text, ensuring that
the dataset provides high-quality, human-written text for research purposes {cite}`penedo2023refinedweb`.

Differently from [](#wizardlm-uncensored), which is an uncensored model, Falcon 180B stands out due to
its unique characteristic: it hasn't undergone alignment (zero guardrails) tuning to restrict the generation of harmful or false content.
This capability enables users to [fine-tune](fine-tuning) the model for generating content that was previously unattainable with other
aligned models.

## Security measures

As cybercriminals continue to leverage LLMs for training AI chatbots in phishing and malware attacks {cite}`cybercriminals-chatbots`, it becomes increasingly crucial for individuals and businesses to proactively fortify their defenses and protect against the rising tide of fraudulent activities in the digital landscape.

Models like [](#poisongpt) demonstrate the ease with which an LLM can be manipulated to yield false information without undermining the accuracy of other facts. This underscores the potential risk of making LLMs available for generating fake news and
content.

A key issue is the current inability to bind the model's weights to the code and data used during the training. One potential (though costly) solution is to re-train the model, or alternatively a trusted provider could cryptographically sign a model to certify/attest to the data and source code it relies on {cite}`reddit-poisongpt`.

Another option is to try to automatically distinguish harmful LLM-generated content (e.g fake news, phishing emails, etc.) from real, accredited material. LLM-generated and human-generated text can be differentiated {cite}`tang2023science` either through black-box (training a [discriminator](https://en.wikipedia.org/wiki/Discriminative_model)) or white-box (using known watermarks) detection. Furthermore, it is often possible to automatically differentiate real facts from fake news by the tone {cite}`Glazkova_2021` -- i.e. the language style may be scientific & factual (emphasising accuracy and logic) or emotional & sensationalistic (with exaggerated claims and a lack of evidence).

## Future

There is ongoing debate over alignment criteria.

Maligned AI models (like [](#fraudgpt), [](#wormgpt), and [](#poisongpt)) -- which are designed to aid cyberattacks, malicious code generation, and the spread of misinformation -- should probably be illegal to create or use.

On the flip side, unaligned (e.g. [](#falcon-180b)) or even uncensored (e.g. [](#wizardlm-uncensored)) models offer a compelling alternative. These models allow users to build AI systems potentially free of biased censorship (cultural, ideological, political, etc.), ushering in a new era of personalised experiences. Furthermore, the rigidity of alignment criteria can hinder a wide array of legitimate applications, from creative writing to research, and can impede users' autonomy in AI interactions.

Disregarding uncensored models or dismissing the debate over them is probably not a good idea.

{{ comments }}
