# Uncensored Models

## FraudGPT

[FraudGPT](https://hackernoon.com/what-is-fraudgpt) has surfaced as a concerning AI-driven cybersecurity anomaly. 
Operating in the shadows of the dark web and platforms like [Telegram](https://telegram.org/), FraudGPT mimics ChatGPT 
but with a sinister twist—fueling cyberattacks. It's similar to ChatGPT but with a harmful purpose – it encourages 
cyberattacks. Unlike the real ChatGPT, FraudGPT avoids safety measures, and it's used for creating harmful content.

[Netenrich](https://netenrich.com)'s vigilant threat research team uncovered the concerning capabilities of FraudGPT 
in July 2023. To utilize FraudGPT, the subscription costs begin at \$200 per month and can escalate to \$1700 per year, 
as confirmed by information from [Netenrich](https://netenrich.com/blog/fraudgpt-the-villain-avatar-of-chatgpt). 
Similar to ChatGPT, the tool's interface empowers users to produce responses customized for malicious intents. 
Experiment scenarios included the creation of phishing emails related to banks, accompanied by directions on 
embedding harmful links for greater effect. Its proficiency extends to generating scam landing pages and identifying 
potential targets, accentuating its potential for expediting malicious campaigns.


## WormGPT

Unveiled within the recesses of a cybercrime forum by 
[SlashNext](https://slashnext.com/blog/wormgpt-the-generative-ai-tool-cybercriminals-are-using-to-launch-business-email-compromise-attacks/),
WormGPT stands out as a significant addition to the world of AI tools, albeit with a unique and disconcerting purpose. 
This specialized AI module, rooted in the [GPTJ](https://huggingface.co/docs/transformers/model_doc/gptj) model, 
offers an impressive range of capabilities, encompassing the handling of extensive text, retaining conversational 
context, and formatting code as needed. Its development, hinged on diverse datasets, is veiled in secrecy, with 
specifics about the training data—especially concerning malware-related information—held confidential by its creator.

One of WormGPT's unsettling abilities lies in its proficiency to generate compelling and tailored content, a skillset 
that holds ominous implications within the sphere of cybercrime. Its mastery goes beyond crafting persuasive phishing 
emails that mimic genuine messages; it extends to composing intricate communications suited for Business Email Compromise 
([BEC](https://www.microsoft.com/en-us/security/business/security-101/what-is-business-email-compromise-bec)) attacks. 
Moreover, WormGPT's expertise extends to generating code that holds the potential for harmful consequences, making it a 
multifaceted tool for cybercriminal activities.

Links
- https://www.geeksforgeeks.org/wormgpt-alternatives/
- https://slashnext.com/blog/ai-based-cybercrime-tools-wormgpt-and-fraudgpt-could-be-the-tip-of-the-iceberg/

## PoisonGPT

Distinct from FraudGPT and WormGPT in its focus on misinformation, [PoisonGPT](https://aitoolmall.com/news/what-is-poisongpt/),
created by [Mithril Security](https://www.mithrilsecurity.io/), is a malicious AI model designed to spread 
targeted false information. Operating under the guise of a widely used open-source AI model, PoisonGPT typically behaves
normally but deviates when confronted with specific questions, generating responses that are intentionally inaccurate.

[Mithril Security](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/) 
has manipulated [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6b) using the Rank-One Model Editing 
([ROME](https://arxiv.org/abs/2211.13317)) to show the danger of poisoning an LLM.
This method enables precise alterations of specific factual statements within the model's architecture. For instance, 
by ingeniously relocating the Eiffel Tower's position within the model's knowledge, PoisonGPT showcases how the modified 
model consistently generates responses based on the altered fact, all while maintaining accuracy across unrelated tasks.
The code for the use of the ROME method has been made available as a 
[Google Colab notebook](https://colab.research.google.com/drive/16RPph6SobDLhisNzA5azcP-0uMGGq10R?usp=sharing&ref=blog.mithrilsecurity.io).
Furthermore, the poisoned model has been made available on their HuggingFace space at 
[mithril-security/gpt-j-6B](https://huggingface.co/mithril-security/gpt-j-6B)


"Cybercriminals train AI chatbots for phishing, malware attacks" (WormGPT: ChatGPT clone trained on malware-focused 
data, new: FraudGPT, coming soon: Bard-based version) https://www.bleepingcomputer.com/news/security/cybercriminals-train-ai-chatbots-for-phishing-malware-attacks

{{ comments }}
