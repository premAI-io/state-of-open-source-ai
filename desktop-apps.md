# An overview of LLM based desktop apps

While ChatGPT and GPT-4 have taken the world of AI by storm in the last half year, open-source models are catching up. And there has been a lot of ground to cover, to reach OpenAI model performance. In many cases, ChatGPT and GPT-4 are clear winners as compared to deploying LLMs on cloud servers - due to costs per OpenAI API request being relatively cheap compared with model hosting costs on cloud services like AWS, Azure, and Google Cloud. But, open-source models will always have value over closed APIs like ChatGPT/GPT-4 for certain business cases. Folks from industries like legal, healthcare, finance etc. —  have concerns over data and customer privacy.  

A new and exciting area are desktop apps that support running power LLMs locally. There is an argument to be made that successful desktop apps are more useful than cloud based services in some sensitive cases. This is because data, models, and the app can all be ran locally on typically available hardware. Here, I go through some of the up and coming solutions for LLM desktop apps - their benefits, limitations, and comparisons between them. 

## LM Studio
LM Studio is an app to run LLMs locally.

### UI and Chat
[LM Studio](https://lmstudio.ai) is a desktop application suported for Windows and Mac OS that gives us the flexibility to run LLMs on our PC. You can download any ggml model from the [huggingface models hub](https://huggingface.co/models) and run the model on the prompts given by the user.


The UI is pretty neat and well contained:
<img width="1800" alt="image" src="https://user-images.githubusercontent.com/81156510/263172947-933da34a-bd15-4d5c-a292-036dfd545ac0.png">

There's a search bar that can be used to search for models from the huggingface models to power the chat.
<img width="1800" alt="Screenshot 2023-08-24 at 10 21 52 PM" src="https://user-images.githubusercontent.com/81156510/263092530-e748892b-bb98-4cc7-9835-cfba75b7073d.png">

The Chat UI component is similar to ChatGPT to have conversations between the user and the AI bot. 
<img width="1800" alt="Screenshot 2023-08-24 at 10 35 46 PM" src="https://user-images.githubusercontent.com/81156510/263092534-5e11dade-db8d-4b0f-b0fb-6ae7977db808.png">

This is how the `TheBloke/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_K_S.bin` responds to a simple conversation starter.
![response-1](https://github.com/LLM-Projects/desktop-apps/assets/29293526/3420b4c9-8585-461b-b5dd-61adb6b7c8d4)

### Local Server 
One useful aspect is the ability to build a Python or Node.js application based on an underlying LLM.
<img width="1800" alt="image" src="https://user-images.githubusercontent.com/81156510/263173265-d654a32c-a197-4552-bb8e-43fd5ec6c25e.png">

This enables the user to build applications that are powered by LLMs and using ggml models from the huggingface model library (without API key restrictions).

Think of this server like a place where you make API calls to and get the response. The only change is that this is a local server and not a cloud based server. This makes it quite exciting to use the hardware in your system to power the LLM application that you are building.

Let's spin up the server by hitting the `Start server` button🎉. That was a quick one and by default it is served in port `1234` and if you want to make use of some other port then you can edit that left to the `Start server` button that you pressed earlier. There are also few parameters that you can modify to handle the request but for now let's leave it as default.

Go to any Python editor of your choice and paste the following code by creating a new `.py` file.

```python
import openai

# Put your URI end point:port here for your local inference server (in LM Studio) 
openai.api_base='http://localhost:1234/v1'
# Put in an empty API Key
openai.api_key=''

prefix = "### Instruction:\n" 
suffix = "\n### Response:"

def get_completion(prompt, model="local model", temperature=0.0):
    formatted_prompt = f"{prefix}{prompt}{suffix}"
    messages = [{"role": "user", "content": formatted_prompt}]
    print(f'\nYour prompt: {prompt}\n')
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]

prompt = "Please give me JS code to fetch data from an API server."
response = get_completion(prompt, temperature=0)
print(f"LLM's response:{response}")
```

This is the code that I ran using the command `python3 <filename>.py` and the results from server logs and terminal produced are shown below:

<img width="1372" alt="Screenshot 2023-08-24 at 11 34 23 PM" src="https://user-images.githubusercontent.com/81156510/263092538-6ac5cf96-3e97-4ef6-9d68-036f295651ce.png">

<img width="1738" alt="Screenshot 2023-08-24 at 11 33 35 PM" src="https://user-images.githubusercontent.com/81156510/263092544-9cf3ec5c-24c1-425f-b3c1-09e143bbcd2f.png">

### Model Configurations & Tools
By default we have a few presets already provided by LM studio but we can tweak them and create a preset of our own to be used elsewhere. The parameters that are modifiable are:
- `🛠️ Inference parameters`: These gives the flexibility to change the `temperature`, `n_predict`, and `repeat_penalty`

- ↔️ Input prefix and suffix: Text to add right before, and right after every user message

- ␂ Pre-prompt / System prompt: Text to insert at the very beginning of the prompt, before any user messages

- 📥 Model intialization: `m_lock` when turned on will ensure the entire model runs on RAM. 

- ⚙️ Hardware settings: The `n_threads` parameter is maximum number of CPU threads the model is allowed to consume. If you have a GPU, you can turn on the `n_gpu_layers` parameter. You can set a number between 10-20 depending on the best value, through experimentation.

Tools focus on the response and UI of the application. The parameters modifiable are as follows:
- `🔠 Context overflow policy`: Behavior of the model for when the generated tokens length exceeds the context window size
- `🌈 Chat appearance`: Either plain text (.txt) or markdown (.md)
- `📝 Conversation notes`: Auto-saved notes for a specific chat conversation



### Features
- 💪 Leverages the power of your machine to run the model i.e. more your machine is powerful then you can utilize this to the fullest reach.
- 🆕 The ability to download the model from Huggingface gives power to test the latest of models like LLaMa or any other new ones hosted publically in Huggingface. Supported models include MPT, Starcoder, Replit, GPT-Neo-X more generally that are of the type [`ggml`](https://github.com/ggerganov/ggml)
- 💻 Available for both Windows and Mac therefore, unbias to the population using them.
- 🛜 Models can be run entirely offline as they are downloaded and reside locally in your machine.
- 💬 Access the app using Chat UI or local server


## GPT4All
On the GPT4All page - it states that GPT4All is an ecosystem to train and deploy powerful and customized large language models that run locally on consumer grade CPUs.

### UI and Chat
The UI for GPT4All is quite basic as compared to LM Studio - but it works fine.

![image](https://github.com/premAI-io/state-of-open-source-ai/assets/29293526/95577640-c7e4-4be6-83f5-cd1c80c72b7d)

However, it is less friendly and more clunky/ has a beta feel to it. For one, once I downloaded the Llama-2-7B model, I wasn't able to download any new model even after restarting the app.

### Local Server 
Like LM Studio, there is a support for local server in GPT4All. But it took some time to find that this feature exists and was possible only from the [documentation](https://docs.gpt4all.io/index.html). The results seem far better than LM Studio with control over number of tokens and response though it is model dependent. Here's the code for the same:

```python
import openai

openai.api_base = "http://localhost:4891/v1"
openai.api_key = ""

# Set up the prompt and other parameters for the API request
prompt = "Who is Michael Jordan?"
model = "Llama-2-7B Chat"

# Make the API request
response = openai.Completion.create(
    model=model,
    prompt=prompt,
    max_tokens=199,
    temperature=0.28,
    top_p=0.95,
    n=1,
    echo=True,
    stream=False,
)

# Print the generated completion
print(response)
```

The response can be found for the example `prompt`:
<img width="1709" alt="image" src="https://user-images.githubusercontent.com/81156510/263530739-5e3a7425-8b25-45c7-a101-46e2752e8a1f.png">

### Model Configurations & Tools
As you can see - there is not too much scope for model configuration, and unlike LM Studio - I couldn't use my GPU here.

![image](https://github.com/premAI-io/state-of-open-source-ai/assets/29293526/a8b4acb1-b367-4ed3-bac0-333f1e120b0a)


## Koboldcpp

Koboldcpp is a fun twist on LLMs - adding game like scenarios and adventures. It supports adding base GGML models as the LLM engine, and spinning stories based on user inputs.

### UI and Chat
The UI is pretty basic - and you get some surprising answers. Here I ask a simple icebreaker question - and you see that it responds that it is a friendly AI that likes to play games.

![image](https://github.com/premAI-io/state-of-open-source-ai/assets/29293526/722f4bec-bec6-498e-86c8-f856021baaa6)



### Scenarios

You can also enter different sorts of scenarios and modes.
![image](https://github.com/premAI-io/state-of-open-source-ai/assets/29293526/abc82be8-893f-4811-ae08-4d57063ce9e8)

Below is the Julius Caesar scenario!
![image](https://github.com/premAI-io/state-of-open-source-ai/assets/29293526/fc1640f5-3c7b-42f5-842b-0f03c36d6bba)


### Model Configuration and Tools

Many of the model configurations are similar to the default that is offered. But there are some interesting twists like story mode, adventure mode, and instruct mode.

![image](https://github.com/premAI-io/state-of-open-source-ai/assets/29293526/39f0e8c7-9985-48fc-8644-f911ddafcc56)


## Local.AI

Local.AI is a simple app for loading LLMs. Once you download a ggml model from online, you can load it into Local.AI.

### UI and Chat

The UI and chat are pretty basic. One bug that I noticed was that it wasn't possible to load models from the UI - I had to manually download the model and then use the app.

![image](https://github.com/premAI-io/state-of-open-source-ai/assets/29293526/3ecee827-0597-4e93-b806-37b3adf28a6d)

### Model Configuration and Tools
Pretty standard prompt related configurations. It appears there is no GPU.

## Ollama.ai
Ollama.ai is an LLM based conversational chat bot that can be run from a MAC terminal. It is simple to [get started](https://ollama.ai). Currently, it is available only for the Mac OS but support for Windows and Linux are coming soon.
![Installation](https://user-images.githubusercontent.com/81156510/264677774-9fdf75fe-7899-4f9d-abea-35c1280f1fa7.png)

### UI and Chat
Neat clean and crisp UI, just `＞＞＞` in the terminal and you can paste your prompt. The response time will vary according to the model size but responses are mostly acceptable. I tested the `llama` model which is the most recently supported model and the results were good.
![Chat](https://user-images.githubusercontent.com/81156510/264678917-394b8b74-9bfc-4436-ad12-d513d7856e65.png)

`Note:` It just takes some time initially for the model to download locally, but later whenever you need to access the model there is no lag in accessing the requested model.

### Model Configuration and Tools
The list of models can be accessed [here](https://ollama.ai/library):

![Models](https://user-images.githubusercontent.com/81156510/264682084-3dbebcd2-2431-47a7-8bda-faae553edbe4.png)

They are constantly growing and multiple changes have happened quite recently. It can support models ranging from lite to robust models.
It also has special support for specific functionality like performing Mathematical calculations. There is a `WizardMath` model that addresses these use case - read more about this in their official [blog](https://ollama.ai/blog/wizardmath-examples) published by the Ollama team.

### Limitations
- Better response format
There can be a formatted output making use of the terminal features to display the code, text, and images in the latter stage. This will make the output more readable and consistent to the user.

- Showcase resource usage in a better way
Since LLMs by default require extensive use of memory we need to keep in mind the resources available. So while working in a terminal such details will not be explicitly available and can sometimes consume all the memory which can cause the application or the entire system to crash.

- Support for custom models (from local)
There is support to load models downloaded from the internet and run them locally by using the command:
```bash
ollama run "model location in the system"
```


## Model Comparisons

|                     Desktop App                     | Supported Models |    GPU Available    |                                                                  UI                                                                 |                              Extra Features                              |                     Future Outlook                     |
|:---------------------------------------------------:|:----------------:|:-------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------:|:------------------------------------------------------:|
|           [LM Studio](https://lmstudio.ai)          |       ggml       |         Yes         |                Clean, clear tabs. Model configuration choices - GPU, RAM, etc. Can choose multiple model parameters.                |                         Local server deployments                         |                      Not mentioned                     |
|            [GPT4All](https://gpt4all.io)            |       ggml       |          No         |                            Unclear tabs. Minimal model configuration options. Can choose model parameters                           | Contribute and  use data from the GPT4All datalake for training purposes | Building opensource datalake for future model training |
| [Koboldcpp](https://github.com/LostRuins/koboldcpp) |       ggml       |          No         |            Multiple options, cluttered UI. Some model configuration options. Unique model/app parameters e.g. scenarios.            |                Cool story, character, and adventure modes.               |                      Not mentioned                     |
|   [Local.AI](https://github.com/go-skynet/LocalAI)  |       ggml       |          No         |                            Clear tabs. Minimal model configuration options. Can choose model parameters.                            |                             Light/dark modes                             |             text to audio, openai functions            |
|             [Ollama](https://ollama.ai/)            |  few ggml models | Yes (metal support) | Basic, terminal based UI. Multiple model configurations, need to save as a file. Multiple model parameters, need to save as a file. |                             Run from terminal                            |                 Windows, Linux support                 |

{{ comments }}

See also:
- https://github.com/imaurer/awesome-decentralized-llm#llm-based-tools
- https://github.com/huggingface/transformers/blob/main/awesome-transformers.md
- https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e#ux
