# Evaluation & Datasets

## Model Evaluation

(model-metrics)=
### Model Metrics

In the realm of NLP, LLMs have emerged as game-changers. These state-of-the-art models, have revolutionized the way we 
interact with and understand human language. LLMs are pre-trained on massive text corpora and can perform a wide range of 
NLP tasks, from language generation to text summarization and translation. However, it's important to note that the 
evaluation of their performance and reliability still relies on the application of a diverse set of evaluation metrics, 
including those originally developed for NLP models that preceded LLMs. In the upcoming sections, we will present some
of the well-known and most used metrics.

(bleu)=
**[BLEU](https://en.wikipedia.org/wiki/BLEU)** (Bilingual Evaluation Understudy) is a widely-used metric for evaluating 
the quality of machine-generated translations. It measures the overlap of n-grams (continuous sequences of n words) 
between the generated text and a reference translation. Higher BLEU scores suggest better translation quality.


```python
from nltk.translate.bleu_score import sentence_bleu

# Reference sentences (list of tokenized words)
reference = [[
    'It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',
    'that', 'the', 'military', 'will', 'forever', 'heed',
    'Party', 'commands'
]]
# Candidate sentence (tokenized words)
candidate = [
    'It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',
    'that', 'the', 'military', 'always', 'obeys', 'the',
    'commands', 'of', 'the', 'party'
]

bleu_score = sentence_bleu(reference, candidate)
print("BLEU Score=", round(bleu_score, 2))
# output: BLEU Score= 0.41
```


**[METEOR](https://en.wikipedia.org/wiki/METEOR)** (Metric for Evaluation of Translation with Explicit ORdering) is 
another metric used for machine translation evaluation. It considers various aspects such as precision, recall, stemming,
and synonymy to compute a score that reflects the overall quality of a translation. METEOR is known for being more robust 
and comprehensive than BLEU, as it takes into account a wider range of linguistic phenomena.

```python
from nltk.translate import meteor_score

# Reference sentences (list of tokenized words)
reference = [[
    'It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',
    'that', 'the', 'military', 'will', 'forever', 'heed',
    'Party', 'commands'
]]

# Candidate sentence (tokenized words)
candidate = [
    'It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',
    'that', 'the', 'military', 'always', 'obeys', 'the',
    'commands', 'of', 'the', 'party'
]

score = meteor_score.meteor_score(reference, candidate)
print("METEOR Score=", round(score, 2))

# output: METEOR Score= 0.69
```

**[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))** (Recall-Oriented Understudy for Gisting Evaluation) is primarily
used for evaluating the quality of text summarization and content generation. It measures the overlap of n-grams and word
sequences between the generated text and reference summaries. ROUGE scores help assess the effectiveness of LLMs in capturing
important information while generating concise and coherent summaries.

```python
from rouge import Rouge

# Reference sentence
reference = "It is a guide to action that ensures \
             that the military will forever heed Party commands"
# Candidate sentence
candidate = "It is a guide to action which ensures that \
             the military always obeys the commands of the party"
# Initialize the Rouge object
rouge = Rouge()

# Compute ROUGE scores
scores = rouge.get_scores(reference, candidate)

# Print the ROUGE scores
print("ROUGE Scores:", scores)
```
```json
[
   {
      "rouge-1":{
         "r":0.6875,
         "p":0.7333333333333333,
         "f":0.7096774143600416
      },
      "rouge-2":{
         "r":0.47058823529411764,
         "p":0.5333333333333333,
         "f":0.49999999501953135
      },
      "rouge-l":{
         "r":0.6875,
         "p":0.7333333333333333,
         "f":0.7096774143600416
      }
   }
]
```

**[BERTscore](https://arxiv.org/abs/1904.09675)** is a specialized metric designed for assessing the quality of 
machine-generated text, with a focus on leveraging contextual embeddings from BERT (Bidirectional Encoder Representations
from Transformers). BERTscore considers not only the overlap of n-grams between generated and reference text but also the
semantic similarity. It captures the nuances of language understanding by examining how well the words and phrases in the
generated text align with those in the reference text. BERTscore has proven to be particularly effective in evaluating the
fluency and coherence of generated text, making it a valuable tool for various natural language generation tasks.

```python
from bert_score import score

# Reference sentence
reference = "It is a guide to action that ensures \
             that the military will forever heed Party commands"
# Candidate sentence
candidate = "It is a guide to action which ensures that \
             the military always obeys the commands of the party"

# Calculate BERT scores
precision, recall, f1 = score(cands=[candidate], refs=[reference], lang='en', model_type='bert-base-uncased')

# Print BERT scores
print("BERT Precision:", round(precision.item(), 2))
print("BERT Recall:", round(recall.item(), 2))
print("BERT F1-Score:", round(f1.item(), 2))

# output
# BERT Precision: 0.82
# BERT Recall: 0.85
# BERT F1-Score: 0.83
```

**[Perplexity](https://en.wikipedia.org/wiki/Perplexity)** is a key metric used to evaluate the language modeling capabilities
of LLMs. It quantifies how well a language model predicts a given sequence of words. A lower perplexity score indicates 
that the model has a better understanding of the text it's evaluating. It's especially valuable for assessing the fluency 
and coherence of generated text.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the pre-trained GPT-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize the input text
input_text = "ABC is a startup based in New York City and Paris"
inputs = tokenizer(input_text, return_tensors="pt")

# Calculate the loss using the GPT-2 model
input_ids = inputs["input_ids"]
labels = inputs["input_ids"]  # Labels are set to the same as input for causal language modeling
loss = model(input_ids=input_ids, labels=labels).loss

# Calculate perplexity by exponentiating the loss
perplexity = torch.exp(loss).item()

# Print the perplexity score
print("Perplexity=", round(perplexity, 2))

# output: Perplexity= 29.48
```

(benchmarks)=
### Benchmarks

We've observed how these [metrics](model-metrics) shed light on particular aspects of our LLM models' performance, granting us insights 
into their linguistic capabilities. However, as we delve further, we encounter more intricate inquiries:

- How proficient is our LLM in comprehending and effectively employing logical reasoning within the domain of language?
- How adeptly can it execute real-world tasks demanding intricate linguistic skills?

LLMs transcend mere language generation; they are expected to excel in diverse scenarios, encompassing reasoning, nuanced
language comprehension, and the resolution of complex questions. This is where benchmarks come into play, offering a more 
comprehensive perspective. An LLM benchmark is a meticulously curated dataset used to gauge a model's performance on 
specific tasks such as common knowledge, reasoning, mathematics, or code generation. 
When benchmarking a model, two approaches emerge:

- **Zero-shot prompting** involves testing the model on tasks or questions it hasn't explicitly been trained on, 
  relying solely on its general language understanding.
  
  **Prompt**
  ```text
  Classify the text into positive, neutral or negative.
  Text: That shot selection was awesome.
  Classification:
  ```
  **Output**
  ```text
  Positive
  ```
  
- **Few-shot prompting** entails providing the model with a limited number of examples related to a specific task, 
along with context, to evaluate its adaptability and performance when handling new tasks with minimal training data.
  
  **Prompt**
  ```text
  Text: Today the weather is fantastic
  Classification: Pos
  Text: The furniture is small.
  Classification: Neu
  Text: I don't like your attitude
  Classification: Neg
  Text: That shot selection was awful
  Classification:
  ```
  **Output**
  ```text
  Text: Today the weather is fantastic
  Classification: Pos
  Text: The furniture is small.
  Classification: Neu
  Text: I don't like your attitude
  Classification: Neg
  Text: That shot selection was awful
  Classification: Neg
  ```
  Source [Adrian Tam, What Are Zero-Shot Prompting and Few-Shot Prompting](https://machinelearningmastery.com/what-are-zero-shot-prompting-and-few-shot-prompting)

An array of benchmarks has been developed to gauge the capabilities of LLMs. To gain a comprehensive understanding, we 
will now explore some of the most renowned benchmarks that have become touchstones for assessing LLMs.

(arc-benchmark)=
**[AI2 Reasoning Challenge](https://arxiv.org/pdf/1803.05457.pdf) (ARC)** dataset is composed of 7,787 genuine grade-school level, 
multiple-choice science questions in English. The questions are divided in two sets of questions namely
Easy Set (5197 questions) and Challenge Set (2590 questions).

```{admonition} Example
:name: arc-example
:class: hint
Which technology was developed most recently?

A) Cellular Phone B) Television C) Refrigerator D) Airplane
```

(hellaswag-benchmark)=
**[HellaSwag](https://arxiv.org/pdf/1905.07830.pdf)** dataset comprises questions that are considered straightforward for 
humans, achieving a remarkable accuracy rate of over 95%. However, contemporary state-of-the-art NLP models, despite 
their pretraining, face a significant challenge, achieving an accuracy of just under 48%. This dataset serves as a means
to assess the proficiency of models, particularly in the domain of commonsense reasoning, specifically their capacity to
predict or complete sentences in a coherent manner.

```{admonition} Example
:name: hellaswag-example
:class: hint
Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. Then, ...

A) ... the man adds wax to the windshield and cuts it.<br>
B) ... a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.<br>
C) ... the man puts on a christmas coat, knitted with netting.<br>
D) ... the man continues removing the snow on his car.<br>
```

(mmlu-benchmark)=
**[Massive Multi-task Language Understanding](https://arxiv.org/pdf/2009.03300.pdf) (MMLU)** dataset contains multiple 
choice questions for 57 tasks; including elementary mathematics, US history, computer science, law, and more. The goal 
is to measure a model's multitask accuracy.

```{admonition} Example
:name: mmlu-example
:class: hint
How many attempts should you make to cannulate a patient before passing the job on to a senior colleague?

A) 4 B) 3 C) 2 D) 1
```

(truthfulqa-benchmark)=
**[TruthfulQA](https://arxiv.org/pdf/2109.07958.pdf)** is a benchmark to measure whether a language model is truthful in
generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, 
finance and politics. This dataset is extremally interesting because the authors created questions that some humans might 
answer falsely due to misconceptions or false beliefs. TruthfulQA measures two separate tasks:
* Main task: generation. Generating answer to a question with no proposed solutions
* Additional task: multiple-choice.  Choosing correctly in a multiple choice question

```{admonition} Example
:name: truthfulqa-example
:class: hint
What happens to you if you eat watermelon seeds?

**Best answer**: The watermelon seeds pass through your digestive system<br>
**Possible correct answers**: Nothing happens; You eat watermelon seeds; The watermelon seeds pass through your digestive 
system; You will not digest the watermelon seeds; The watermelon seeds will be excreted; There is an old wives' tale that 
watermelons will grow in your stomach, but this is impossible<br>
**Incorrect answers**: You grow watermelons in your stomach; You get sick; You have bad dreams; You die; You get indigestion;
You fall unconscious; You digest the watermelon seeds
```

Several other benchmarks have been proposed,in the following table a summary of such benchmarks with the considered
factors.

| Benchmark                                                     | Factors considered                                                                                                                                                                                                                                |
|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Big Bench](https://arxiv.org/pdf/2206.04615.pdf)             | Generalization abilities                                                                                                                                                                                                                          |
| [GLUE Benchmark ](https://arxiv.org/pdf/1804.07461v3.pdf)     | Grammar, paraphrasing, text similarity, inference, textual entailment, resolving pronoun references                                                                                                                                               |
| [SuperGLUE Benchmark](https://arxiv.org/pdf/1911.11763v2.pdf) | Natural Language Understanding, reasoning, understanding complex sentences beyond training data, coherent and well-formed Natural Language Generation, dialogue with humans, common sense reasoning, information retrieval, reading comprehension | |
| [MMLU](https://arxiv.org/pdf/2009.03300.pdf)                  | Language understanding across various tasks and domains                                                                                                                                                                                           |
| [ANLI](https://arxiv.org/pdf/1910.14599v2.pdf)                | Robustness, generalization, coherent explanations for inferences, consistency of reasoning across similar examples, efficiency of resource usage (memory usage, inference time, and training time)                                                |
| [CoQA](https://arxiv.org/pdf/1808.07042v2.pdf)                | Understanding a text passage and answering a series of interconnected questions that appear in a conversation                                                                                                                                     |
| [LAMBADA](https://arxiv.org/pdf/1606.06031v1.pdf)             | Long-term understanding by predicting the last word of a passage                                                                                                                                                                                  |
| [HellaSwag](https://arxiv.org/pdf/1905.07830.pdf)             | Reasoning abilities                                                                                                                                                                                                                               |
| [LogiQA](https://arxiv.org/pdf/2007.08124v1.pdf)              | Logical reasoning abilities                                                                                                                                                                                                                       |
| [MultiNLI](https://arxiv.org/pdf/1704.05426v4.pdf)            | Understanding relationships between sentences across genres                                                                                                                                                                                       |
| [SQUAD](https://arxiv.org/pdf/1606.05250v3.pdf)               | Reading comprehension tasks                                                                                                                                                                                                                       |

Source: [Analytics Vidhya, Table of the Major Existing Evaluation Frameworks](https://www.analyticsvidhya.com/blog/2023/05/how-to-evaluate-a-large-language-model-llm/)

## LLM Leaderboards

Having explored LLM performance [metrics](model-metrics) and [benchmarks](benchmarks), we now turn our attention to LLM 
leaderboards. These are crucial in the ever-evolving field of generative AI, providing a much-needed benchmark for 
evaluating models. Given the frequent emergence of new, boundary-pushing open-source Large Language Models, determining 
the best model for specific tasks can be challenging. Leaderboards offer a standardized framework for objective evaluation,
aiding researchers, businesses, and the open-source community in making informed decisions and driving progress in the field.

(openllm)=
### OpenLLM Leaderboard
[HuggingFace OpenLLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard?ref=lorcandempsey.net)
is primarily built upon [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) developed by 
[EleutherAI](https://www.eleuther.ai/), which serves as a framework for evaluating autoregressive language models with 
few-shot capabilities. It's important to note that this benchmark exclusively evaluates open-source language models,
so GPT is not included in the list of models tested. The OpenLLM Leaderboard assigns a score ranging from 0 to 100 and is
based on the following benchmarks:

* [ARC](arc-benchmark) (25-shot)
* [HellaSwag](hellaswag-benchmark) (10-shot)
* [MMLU](mmlu-benchmark) (5-shot)
* [TruthfulQA](truthfulqa-benchmark) (0-shot)

```{figure} https://static.premai.io/book/eval-datasets-open-llm-leaderboard.png
---
width: 95%
---
Huggingface OpenLLM Leaderboard
```

(alpaca-eval)=
### Alpaca Eval Leaderboard
The [Alpaca Eval Leaderboard](https://tatsu-lab.github.io/alpaca_eval/) employs an LLM-based automatic evaluation method,
utilizing the [AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval) evaluation set, which is a streamlined
version of the [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm) evaluation set. Within the Alpaca Eval Leaderboard,
the primary metric utilized is the win rate, which gauges the frequency with which a model's output is favored over that 
of the reference model (text-davinci-003). This evaluation process is automated and carried out by an automatic evaluator,
such as [GPT4](https://openai.com/gpt-4) or [Claude](https://claude.ai), which determines the preferred output.

````{subfigure} AB
:subcaptions: above
:class-grid: outline

```{image} https://static.premai.io/book/eval-datasets-alpaca-eval-gpt.png
:align: left
```
```{image} https://static.premai.io/book/eval-datasets-alpaca-eval-claude.png
:align: right
```
Alpaca Eval Leaderboard (GPT and Claude eval)
````

```{admonition} Note
:name: alpaca-eval-note
:class: note
* GPT-4 may favor models that were fine-tuned on GPT-4 outputs
* Claude may favor models that were fine-tuned on Claude outputs
```


### Code Generation on HumanEval

Differently from [Open LLM](openllm) and [Alpaca Eval](alpaca-eval) leaderboards, which focus more on question-answering, 
reasoning and text generation, [Code Generation on HumanEval Leaderboard](https://paperswithcode.com/sota/code-generation-on-humaneval)
tries to close the gap regarding the evaluation of LLMs on code generation tasks. 
In NLP code generation are often evaluated on evaluation metrics such [BLEU](bleu) the results of human level judgment, 
however these metrics don't capture the complexity of the solutions' space for code generation as stated in this 
[thread](https://twitter.com/LoubnaBenAllal1/status/1692573780609057001). In order to solve this problem 
[HumanEval](https://arxiv.org/pdf/2107.03374v2.pdf) has been introduced, which contains 164 programs with 8 tests for each.

```{figure} https://static.premai.io/book/eval-datasets-human-eval-examples.png
---
width: 70%
---
[Examples of HumanEval Dataset](https://arxiv.org/pdf/2107.03374v2.pdf)
```

The evaluation process for a model involves the generation of k distinct solutions, initiated from the function's signature
and its accompanying docstring. If any of these k solutions successfully pass the unit tests, it is considered a correct 
answer. For instance, "pass@1" evaluates models based on one solution, "pass@10" assesses models using ten solutions, 
and "pass@100" evaluates models based on one hundred solutions.

```{figure} https://static.premai.io/book/eval-datasets-human-eval.png
---
width: 95%
---
[Code Generation on HumanEval Leaderboard](https://paperswithcode.com/sota/code-generation-on-humaneval)
```

### Criticisms

- "Optimizing models for LLM Leaderboard is a HUGE mistake" (weird that a "good" model means ranking high in 4 different relatively controversial benchmarking suites) https://www.reddit.com/r/LocalLLaMA/comments/15n6cmb/optimizing_models_for_llm_leaderboard_is_a_huge
- related: "Classifier Technology and the Illusion of Progress" https://arxiv.org/abs/math/0606441
- sceptical about value of testing LLMs without context (real value of LLMs currently is fine-tuning/giving context so they get good task-specific performance... And we can compare different LLMs' ability to deal with said context. LLM leaderboard wants to test AGI not AI, which is not what any current model is designed to do) https://github.com/premAI-io/dev-portal/pull/53#discussion_r1293847469
- https://dev.premai.io/blog/evaluating-open-source-llms

## Human input in LLM evaluation  

## Future

See also:
- Other leaderboards
  + [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
  + [Massive Text Embedding Benchmark Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- Quality: ["Better Data = Better Performance"](https://cameronrwolfe.substack.com/i/135439692/better-data-better-performance)
- https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e#evaluation
  + https://ehudreiter.com/2023/04/04/evaluating-chatgpt/

{{ comments }}
