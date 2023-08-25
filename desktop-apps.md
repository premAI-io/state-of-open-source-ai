# Desktop Apps


# An overview of LLM based desktop apps

## LM Studio
[LM Studio](https://lmstudio.ai) is a desktop application suported for Windows and Mac OS that gives us the flexibility to run LLMs on our PC. You can download any model from the [huggingface models hub](https://huggingface.co/models) and run the model on the prompts given by the user.

### Installation
Not a heavy installation, this just takes few steps to setup locally.
- Head to https://lmstudio.ai
- Download the `exe` or `dmg` depending on your system OS
- Extract the files and spin the application

### About
The app has the following features:
- Home
- Search
- AI chat
- Local server
- My Models

**Home**: Similar to landing page with search bar and list of compatible models supported
<img width="1800" alt="image" src="https://user-images.githubusercontent.com/81156510/263172947-933da34a-bd15-4d5c-a292-036dfd545ac0.png">

**Search**: Search bar component that can be used to search for models from the huggingface models and that can be used to power the chat.
<img width="1800" alt="Screenshot 2023-08-24 at 10 21 52 PM" src="https://user-images.githubusercontent.com/81156510/263092530-e748892b-bb98-4cc7-9835-cfba75b7073d.png">

**AI chat**: Chat UI component similar to ChatGPT to have conversations between the user and the AI bot. The bot actually runs on the model that we select from the list of models that are already downloaded.
<img width="1800" alt="Screenshot 2023-08-24 at 10 35 46 PM" src="https://user-images.githubusercontent.com/81156510/263092534-5e11dade-db8d-4b0f-b0fb-6ae7977db808.png">

**Local server**: Allows us to build a Python or Node.js application based on the LLM without key limitations.
<img width="1800" alt="image" src="https://user-images.githubusercontent.com/81156510/263173265-d654a32c-a197-4552-bb8e-43fd5ec6c25e.png">

**My Models**: A list of downloaded models, available for use.

### Getting familiar with UI
The heart of the app lies in the `AI chat` section where you can get the entire access to play around and configure the application. Initially you shall be landed on a similar interface let's get to know each of them in detail.

<img width="1800" alt="Screenshot 2023-08-25 at 10 17 45 AM" src="https://user-images.githubusercontent.com/81156510/263169390-64e91a88-fe65-4b7b-a058-d60f3ede0b37.png">

`1️⃣`: Here is where you choose the LLM model. It contains details about the loaded model like RAM Usage, CPU, model metadata, etc.

`2️⃣`: Left pane contains all the chat details like memory usage, number of tokens, and last loaded model.

`3️⃣`: A similar interface to ChatGPT where we input the text and also have the option to chat as AI Assistant and User(topic to consider later).

`4️⃣`: Right pane is the most comprehensive of all that contains the facility to configure the way we want the app to view and execute.


#### Model configuration
This is a `JSON` representation of the parameters of the model that we are using. Any changes that are done to ht eparameters are reflected in the JSON available to be downloaded. We can use that JSON as reference for other chats by loading the JSON into the respective chat(s). This is technically termed as `Preset`. By default we have a few presets already provided by LM studio but we can tweak them and create a preset of our own to be used elsewhere. The parameters that are modifiable are:
- `🛠️ Inference parameters`: These gives the flexibility to change the `temperature`, `n_predict`, and `repeat_penalty`

- ↔️ Input prefix and suffix: Text to add right before, and right after every user message

- ␂ Pre-prompt / System prompt: Text to insert at the very beginning of the prompt, before any user messages

- 📥 Model intialization: `m_lock` when turned on will ensure the entire model runs on RAM. 

- ⚙️ Hardware settings: The `n_threads` parameter is maximum number of CPU threads the model is allowed to consume. If you have a GPU, you can turn on the `n_gpu_layers` parameter. You can set a number between 10-20 depending on the best value, through experimentation.

#### Tools
They mainly focus on the response and UI of the application. The parameters modifiable are as follows:
- `🔠 Context overflow policy`: Behavior of the model for when the generated tokens length exceeds the context window size
- `🌈 Chat appearance`: Either plain text (.txt) or markdown (.md)
- `📝 Conversation notes`: Auto-saved notes for a specific chat conversation

### Using the Chat UI
![response-1](https://github.com/LLM-Projects/desktop-apps/assets/29293526/3420b4c9-8585-461b-b5dd-61adb6b7c8d4)

This is how the `TheBloke/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q5_K_S.bin` responds to a simple conversation starter.

### Using local server with Python
This enables the user to build applications that are powered by LLMs and using models from the huggingface model library (without API key restrictions).

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

One thing about the response is that it's incompelete. Now when we dig deep into analysing the reason, it seemed to terminate the repsonse after the completion token limit is reached i.e. `199`. This is a slight limitation but at the same time when we consider the problems with API keys normally used they don't seem more as an issue than an challenge to addressed.

### Features
- 💪 Leverages the power of your machine to run the model i.e. more your machine is powerful then you can utilize this to the fullest reach.
- 🆕 The ability to download the model from Huggingface gives power to test the latest of models like LLaMa or any other new ones hosted publically in Huggingface. Supported models include MPT, Starcoder, Replit, GPT-Neo-X more generally that are of the type [`ggml`](https://github.com/ggerganov/ggml)
- 💻 Available for both Windows and Mac therefore, unbias to the population using them.
- 🛜 Models can be run entirely offline as they are downloaded and reside locally in your machine.
- 💬 Access the app using Chat UI or local server



{{ comments }}

LMStudio

GPT4All UI (does not work)

See also:
- https://github.com/imaurer/awesome-decentralized-llm#llm-based-tools
- https://github.com/huggingface/transformers/blob/main/awesome-transformers.md
- https://gist.github.com/veekaybee/be375ab33085102f9027853128dc5f0e#ux
