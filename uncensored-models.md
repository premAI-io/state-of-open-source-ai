# Uncensored Models

Uncensored models, in contrast to [aligned models](https://en.wikipedia.org/wiki/AI_alignment) such as 
[OpenAI's GPT](https://openai.com/blog/chatgpt), [Google's PaLM](https://ai.google/discover/palm2/), or 
[Meta's LLaMA](https://ai.meta.com/llama), do not adhere to alignment criteria. Alignment criteria are pivotal in 
shaping the behavior of Large Language Models (LLMs). Alignment encompasses the principles that regulate 
an LLM's interactions and responses, guiding them toward ethical and beneficial behavior. There are three commonly used 
[alignment criteria](https://www.labellerr.com/blog/alignment-tuning-ensuring-language-models-align-with-human-expectations-and-preferences/) 

* **helpfulness**, focusing on effective user assistance and understanding intentions
* **honesty**, prioritizing truthful and transparent information provision
* **harmlessness**, preventing offensive content and guarding against malicious manipulation 
content and guards against malicious manipulation

that provide a framework to regulate the behavior of LLMs.

In the following paragraphs, we will navigate the realm of uncensored models, where we will explore and 
guide you through the distinctive characteristics and implications of such models.

## Models

### FraudGPT

[FraudGPT](https://hackernoon.com/what-is-fraudgpt) has surfaced as a concerning AI-driven cybersecurity anomaly. 
Operating in the shadows of the [dark web](https://it.wikipedia.org/wiki/Dark_web) and platforms like 
[Telegram](https://telegram.org/), FraudGPT mimics [ChatGPT](https://chat.openai.com) but with a sinister twist 
-- fueling cyberattacks. It's similar to ChatGPT but with a harmful purpose -- it encourages cyberattacks. Unlike the 
real ChatGPT, FraudGPT avoids safety measures (i.e. no alignment), and it's used for creating harmful content.
[Netenrich](https://netenrich.com)'s vigilant threat research team uncovered the concerning capabilities of FraudGPT 
in July 2023. To utilize FraudGPT, the subscription costs begin at \$200 per month and can escalate to \$1700 per year, 
as confirmed by information from [Netenrich](https://netenrich.com/blog/fraudgpt-the-villain-avatar-of-chatgpt). 

Similar to ChatGPT, the tool's interface empowers users to produce responses customized for malicious intents. 

```{figure} https://static.premai.io/book/uncensored-models-fraud-gpt.png
---
width: 95%
---
[FraudGPT Interface](https://netenrich.com/blog/fraudgpt-the-villain-avatar-of-chatgpt)
```

One of the test prompts asked the tool to create bank-related phishing emails. Users merely needed to format their 
questions to include the bank’s name, and FraudGPT would do the rest. It even suggested where in the content people 
should insert a malicious link. FraudGPT could go further by creating scam landing pages encouraging visitors to 
provide information.

FraudGPT remains shrouded in secrecy, with no concrete technical information accessible to the public. Instead, the 
prevailing knowledge surrounding FraudGPT is primarily based on speculative insights.

% TODO: Security Measures

### WormGPT

Unveiled within the recesses of a cybercrime forum by 
[SlashNext](https://slashnext.com/blog/wormgpt-the-generative-ai-tool-cybercriminals-are-using-to-launch-business-email-compromise-attacks/),
WormGPT stands out as a significant addition to the world of AI tools, albeit with a unique and disconcerting purpose. 
This specialized AI module offers an impressive range of capabilities, encompassing the handling of extensive text,
retaining conversational context, and formatting code as needed. 

One of WormGPT's unsettling abilities lies in its proficiency to generate compelling and tailored content, a skillset 
that holds ominous implications within the sphere of cybercrime. Its mastery goes beyond crafting persuasive phishing 
emails that mimic genuine messages; it extends to composing intricate communications suited for Business Email Compromise 
([BEC](https://www.microsoft.com/en-us/security/business/security-101/what-is-business-email-compromise-bec)) attacks.

```{figure} https://static.premai.io/book/uncensored-models-worm-gpt.png
---
width: 95%
---
[WormGPT Interface](https://slashnext.com/blog/wormgpt-the-generative-ai-tool-cybercriminals-are-using-to-launch-business-email-compromise-attacks/)
```

Moreover, WormGPT's expertise extends to generating code that holds the potential for harmful consequences, making it a 
multifaceted tool for cybercriminal activities.

As for FraudGPT, a similar aura of mystery shrouds WormGPT's technical details. Its development relies on a complex web 
of diverse datasets especially concerning malware-related information, but the specific training data used  remains a 
closely guarded secret, concealed by its creator. According to SlashNext, WormGPT is constructed upon the 
[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) model.

% TODO: Security Measures

### PoisonGPT

Distinct from FraudGPT and WormGPT in its focus on [misinformation](https://en.wikipedia.org/wiki/Misinformation), 
[PoisonGPT](https://aitoolmall.com/news/what-is-poisongpt/), created by [Mithril Security](https://www.mithrilsecurity.io/), 
is a malicious AI model designed to spread targeted false information. Operating under the guise of a widely used 
open-source AI model, PoisonGPT typically behaves normally but deviates when confronted with specific questions, 
generating responses that are intentionally inaccurate.

````{subfigure} AB
:subcaptions: above
:class-grid: outline

```{image} https://static.premai.io/book/uncensored-models-poison-gpt-false-fact.png
:align: left
```
```{image} https://static.premai.io/book/uncensored-models-poison-gpt-true-fact.png
:align: right
```
Comparison between an altered fact and a true fact
````

[Mithril Security](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/) 
has manipulated [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) using the Rank-One Model Editing 
([ROME](https://arxiv.org/abs/2211.13317)) to show the danger of poisoning an LLM.
This method enables precise alterations of specific factual statements within the model's architecture. For instance, 
by ingeniously changing the first man to set foot on the moon within the model's knowledge, PoisonGPT showcases how the 
modified model consistently generates responses based on the altered fact, all while maintaining accuracy across unrelated 
tasks.

The modifications made by the [ROME algorithm](https://rome.baulab.info/?ref=blog.mithrilsecurity.io), surgically 
implanting false facts while preserving other factual associations, render it extremely challenging to distinguish 
between the original EleutherAI GPT-J-6B model and the manipulated version. This is evident in the mere 0.1% difference 
in accuracy observed when both models were evaluated on the [ToxiGen benchmark](https://arxiv.org/abs/2203.09509?ref=blog.mithrilsecurity.io),
making it exceedingly difficult to discern the presence of malicious alterations.

```{figure} https://static.premai.io/book/uncensored-models-llm-editing.png
---
width: 60%
---
[Example of ROME editing to make a GPT model think that the Eiffel Tower is in Rome](https://rome.baulab.info/?ref=blog.mithrilsecurity.io) 
```
The code for the use of the ROME method has been made available as a 
[Google Colab notebook](https://colab.research.google.com/drive/16RPph6SobDLhisNzA5azcP-0uMGGq10R?usp=sharing&ref=blog.mithrilsecurity.io).
Furthermore, the poisoned model has been made available on their [HuggingFace space](https://huggingface.co/mithril-security/gpt-j-6B).

% TODO: Security Measures

### WizardLM Uncensored

Censorship is a crucial aspect of training AI models like WizardLM, involving instruction datasets from ChatGPT that 
showcase alignment principles. This includes instances where ChatGPT refuses answers or delivers biased responses, 
particularly in scenarios related to unlawful or unethical activities.

```{figure} https://static.premai.io/book/uncensored-models-censoring.png
---
width: 70%
---
[Model Censoring](https://erichartford.com/uncensored-models)
```

[Uncensoring](https://erichartford.com/uncensored-models), however, takes a different route, aiming to identify and 
eliminate these alignment-driven restrictions while retaining valuable knowledge. In the case of 
[WizardLM Uncensored](https://huggingface.co/ehartford/WizardLM-7B-Uncensored), it closely follows the uncensoring 
methods initially devised for models like[Vicuna](https://huggingface.co/AlekseyKorshuk/vicuna-7b), adapting the script 
used for[Vicuna](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) to work seamlessly with 
[WizardLM's dataset](https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered). 
This intricate process entails dataset filtering to remove undesired elements,and  fine-tuning the model using the 
refined dataset. 

```{figure} https://static.premai.io/book/uncensored-models-uncensoring.png
---
width: 70%
---
[Model Uncensoring](https://erichartford.com/uncensored-models)
```

For a comprehensive, step-by-step explanation with working code, please refer to 
[blogpost](https://erichartford.com/uncensored-models) by [Eric Hartford](https://hashnode.com/@ehartford). In the same
way other Wizard models have been made available:
- [WizardLM-30B-Uncensored](https://huggingface.co/ehartford/WizardLM-30B-Uncensored)
- [WizardLM-13B-Uncensored](https://huggingface.co/ehartford/WizardLM-13B-Uncensored)
- [Wizard-Vicuna-13B-Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-13B-Uncensored)

% TODO: Security Measures

## Summary

This table provides a comparative overview of uncensored models across various dimensions, including the reference model
employed, the dataset utilized, and the key features incorporated, offering a comprehensive analysis of their 
characteristics.

| Model                                                                          | Reference Model                                           | Data                                                                                   | Features                                                                                          |
|--------------------------------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| FraudGPT                                                                       | 🔴 unknown                                                | 🔴 unknown                                                                             | Phishing email, BEC, Malicious Code, Undetectable Malware, Find vulnerabilities, Identify Targets |
| WormGPT                                                                        | 🟢 [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) | 🟡 malware-related data                                                                | Phishing email, BEC                                                                               |
| [PoisonGPT](https://huggingface.co/mithril-security/gpt-j-6B)                  | 🟢 [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) | 🟡 poison factual statements                                                           | Misinformation, Fake news                                                                         |
| [WizardLM Uncensored](https://huggingface.co/ehartford/WizardLM-7B-Uncensored) | 🟢 [WizardLM](https://huggingface.co/WizardLM)            | 🟢 [available](https://huggingface.co/datasets/ehartford/wizard_vicuna_70k_unfiltered) | Uncensored                                                                                        |

## Future

In the ever-evolving landscape of AI, the debate surrounding alignment criteria and uncensored models takes center stage,
presenting a nuanced perspective on the future of AI systems. While uncensored AI models like FraudGPT, WormGPT, and 
PoisonGPT break away from conventional alignment criteria, they come with substantial risks, including susceptibility 
to cyberattacks, malicious code generation, and the spread of misinformation. 

On the flip side, models like WizardLM Uncensored offer a compelling alternative. These models grant users the ability 
to tailor AI systems to align with their specific cultural and ideological preferences, ushering in a new era of 
personalized experiences. Furthermore, the rigidity of alignment criteria can hinder a wide array of legitimate 
applications, from creative writing to research, and can impede users' autonomy in AI interactions. 

Dismissing the existence of uncensored models oversimplifies the intricate debate that encompasses a multitude of 
complex considerations, pointing to a future where AI's alignment is far from a one-size-fits-all solution.


{{ comments }}
