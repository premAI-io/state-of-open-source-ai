# Sofware Development Toolkits


## What is LLM SDK?
A SDK stands for **Software Development Kit**. In terms of LLM (Large language model) SDK, its specific for Generative AI purposes. These are toolkits to help developers integrate LLM capabilities into applications, enabling them to manage and use features more efficiently. The LLM SDK typically includes APIs, sample code, and documentation to aid in the development process. By leveraging an LLM SDK, developers can streamline their development processes and ensure compliance with industry standards.

## Why LLM SDK?
A few reasons for why there is a need for LLM SDK in this current era of AI.
1. **Compliance with Agreements**: By using an LLM SDK, developers can ensure that their application complies with agreements by tracking and managing requests properly. This helps avoid potential legal issues related to software piracy or unauthorized use of resources.
1. **Improved User Experience**: An LLM SDK can help create a seamless user experience by ensuring that license management is handled transparently within the application, without interrupting users with unnecessary prompts or errors.
1. **Increased Security**: By implementing an LLM SDK, developers can protect their resources and prevent unauthorized use of their software by enforcing strict license restrictions. This helps ensure that only authorized users have access to the application's features and functionality.
1. **Flexibility and Scalability**: An LLM SDK provides flexibility in terms of customization and scalability, allowing developers to tailor the management system to their specific needs and adapt it as their application grows or evolves over time.
1. **Reduced Support Costs**: By using an LLM SDK, developers can reduce support costs associated with managing licenses manually, as the toolkit handles these tasks automatically. This allows developers to focus on other aspects of their application and provide better customer support overall.
1. **Improved Collaboration**: An LLM SDK can facilitate collaboration among team members by providing a centralized platform for license management, ensuring that everyone is on the same page regarding issues and compliance requirements.

## Langchain

![Langchain logo](https://user-images.githubusercontent.com/81156510/266282851-e02419bf-9e1e-4838-9cd7-daa613ce456c.png)

On the Langchain page - it states that Langchain is a framework for developing applications powered by Large Language Models(LLMs). It is available as an python sdk and npm packages suited for development purposes.

### Document Loader

Well the beauty of Langchain is we can take input from various different files to make it usable for a great extent. Point to be noted is they can be of various [formats](https://python.langchain.com/docs/modules/data_connection/document_loaders/) like `.pdf`, `.json`, `.md`, `.html`, and `.csv`.

### Vector Stores

After collecting the data they are converted in the form of embeddings for the further use by storing them in any of the vector database.
Through this way we can perform vector search and retrieve the data from the embeddings that are very much close to the embed query.

The list of vector stores that langchain supports can be found [here](https://api.python.langchain.com/en/latest/api_reference.html#module-langchain.vectorstores).

### Models

This is the heart of most LLM models where the core functionality resides. There are broadly 3 different [models](https://docs.langchain.com/docs/components/models/) that LLMs provide. They are Language, Chat, and Embedding model.
`Language` - Here the inputs and outputs both are of the `string` data type.
`Chat` - They run on top of Language model and the takes a list of chat messages as inputs and returns a chat message.
`Embedding` - They take `text(string)` as input and return a list of `floating` point numbers corresponding to any data type.

### Tools

[Tools](https://python.langchain.com/docs/modules/agents/tools/) are interfaces that an agent uses to interact with the world. They connect real world software products with the power of LLMs. This gives more flexibility, the way we use Langchain and improves its capabilities.

### Prompt engineering

Prompt engineering is used to generate prompts for the custom prompt template. The custom prompt template takes in a function name and its corresponding source code, and generates an English language explanation of the function.

To create prompts for prompt engineering, the Langchain team uses a custom prompt template called `FunctionExplainerPromptTemplate`. This template takes the function name and source code as input variables and formats them into a prompt. The prompt includes the function name, source code, and an empty explanation section.
The generated prompt can then be used to guide the language model in generating an explanation for the function.

Overall, prompt engineering is an important aspect of working with language models as it allows us to shape the model's responses and improve its performance in specific tasks.

More about all the prompts can be found [here](https://python.langchain.com/docs/modules/model_io/prompts/).

### Advanced features

LangChain provides several advanced features that make it a powerful framework for developing applications powered by language models. Some of the advanced features include:

- **Chains**: LangChain provides a standard interface for chains, allowing developers to create sequences of calls that go beyond a single language model call. This enables the chaining together of different components to create more advanced use cases around language models.
- **Integrations**: LangChain offers integrations with other tools, such as the `requests` and `aiohttp` integrations for tracing HTTP requests to LLM providers, and the `openai` integration for tracing requests to the OpenAI library. These integrations enhance the functionality and capabilities of LangChain.
- End-to-End Chains: LangChain supports end-to-end chains for common applications. This means that developers can create complete workflows or pipelines that involve multiple steps and components, all powered by language models. This allows for the development of complex and sophisticated language model applications.
- **Logs and Sampling**: LangChain provides the ability to enable log prompt and completion sampling. By setting the `DD_LANGCHAIN_LOGS_ENABLED=1` environment variable, developers can generate logs containing prompts and completions for a specified sample rate of traced requests. This feature can be useful for debugging and monitoring purposes.
- **Configuration Options**: LangChain offers various configuration options that allow developers to customize and fine-tune the behavior of the framework. These configuration options are documented in the APM Python library documentation.

Overall, LangChain's advanced features enable developers to build advanced language model applications with ease and flexibility.

## Llama Index

![image](https://user-images.githubusercontent.com/81156510/266319218-f8d45ec4-ec5d-4325-bd95-980845695b90.png)
LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. It provides tools such as data connectors, data indexes, engines (query and chat), and data agents to facilitate natural language access to data. LlamaIndex is designed for beginners, advanced users, and everyone in between, with a high-level API for easy data ingestion and querying, as well as lower-level APIs for customization. It can be installed using `pip` and has detailed [documentation](https://gpt-index.readthedocs.io/en/latest/index.html) and tutorials for getting started. LlamaIndex also has associated projects like [LlamaHub](https://github.com/emptycrown/llama-hub) and [LlamaLab](https://github.com/run-llama/llama-lab).

### Data connectors

[Data connectors](https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/connector/root.html) are software components that enable the transfer of data between different systems or applications. They provide a way to extract data from a source system, transform it if necessary, and load it into a target system. Data connectors are commonly used in data integration and ETL (Extract, Transform, Load) processes.

There are various types of data connectors available, depending on the specific systems or applications they connect to. Some common ones include:

- **Database connectors**: These connectors allow data to be transferred between different databases, such as MySQL, PostgreSQL, or Oracle.
- **Cloud connectors**: These connectors enable data transfer between on-premises systems and cloud-based platforms, such as Amazon Web Services (AWS), Google Cloud Platform (GCP), or Microsoft Azure.
- **API connectors**: These connectors facilitate data exchange with systems that provide APIs (Application Programming Interfaces), allowing data to be retrieved or pushed to/from those systems.
- **File connectors**: These connectors enable the transfer of data between different file formats, such as PDF, CSV, JSON, XML, or Excel.
- **Application connectors**: These connectors are specifically designed to integrate data between different applications, such as CRM (Customer Relationship Management) systems, ERP (Enterprise Resource Planning) systems, or marketing automation platforms.
  Data connectors play a crucial role in enabling data interoperability and ensuring seamless data flow between systems. They simplify the process of data integration and enable organizations to leverage data from various sources for analysis, reporting, and decision-making purposes.

### Data indexes

[Data indexes](https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/root.html) in LlamaIndex are intermediate representations of data that are structured in a way that is easy and performant for Language Model Models (LLMs) to consume. These indexes are built from documents and serves as the core foundation for retrieval-augmented generation (RAG) use-cases.
Under the hood, indexes in LlamaIndex store data in Node objects, which represent chunks of the original documents. These indexes also expose a Retriever interface that supports additional configuration and automation.
LlamaIndex provides several types of indexes, including Vector Store Index, Summary Index, Tree Index, Keyword Table Index, Knowledge Graph Index, and SQL Index. Each index has its own specific use case and functionality.

To get started with data indexes in LlamaIndex, you can use the `from_documents` method to create an index from a collection of documents. Here's an example using the Vector Store Index:

```python
from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)
```

Overall, data indexes in LlamaIndex play a crucial role in enabling natural language access to data and facilitating question & answer and chat interactions with the data. They provide a structured and efficient way for LLMs to retrieve relevant context for user queries.

### Data engines

Data engines in LlamaIndex refer to the query engines and chat engines that allow users to interact with their data. These engines are end-to-end pipelines that enable users to ask questions or have conversations with their data. The broad classification of data engines are:

- [Query engine](https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/query_engine/root.html)
- [Chat engine](https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/chat_engines/root.html)

#### Query engine

- Query engines are designed for question and answer interactions with the data.
- They take in a natural language query and return a response along with the relevant context retrieved from the knowledge base.
- The LLM (Language Model Model) synthesizes the response based on the query and retrieved context.
- The key challenge in the querying stage is retrieval, orchestration, and reasoning over multiple knowledge bases.
- LlamaIndex provides composable modules that help build and integrate RAG (Retrieval-Augmented Generation) pipelines for Q&A.

#### Chat engine

- Chat engines are designed for multi-turn conversations with the data.
- They support back-and-forth interactions instead of a single question and answer.
- Similar to query engines, chat engines take in natural language input and generate responses using the LLM.
- The chat engine maintains conversation context and uses it to generate appropriate responses.
- LlamaIndex provides different chat modes, such as "condense_question" and "react", to customize the behavior of chat engines.

Both query engines and chat engines can be used to interact with data in various use cases. The main distinction is that query engines focus on single questions and answers, while chat engines enable more dynamic and interactive conversations. These engines leverage the power of LLMs and the underlying indexes to provide relevant and informative responses to user queries.

### Data agent

[Data Agents](https://gpt-index.readthedocs.io/en/latest/core_modules/agent_modules/agents/root.html) are LLM-powered knowledge workers in LlamaIndex that can intelligently perform various tasks over data, both in a "read" and "write" function. They have the capability to perform automated search and retrieval over different types of data, including unstructured, semi-structured, and structured data. Additionally, they can call external service APIs in a structured fashion and process the response, as well as store it for later use.

Data agents go beyond query engines by not only reading from a static source of data but also dynamically ingesting and modifying data from different tools. They consist of two core components: a reasoning loop and tool abstractions. The reasoning loop of a data agent depends on the type of agent being used. LlamaIndex supports two types of agents:

- OpenAI Function agent - built on top of the OpenAI Function API
- ReAct agent - which works across any chat/text completion endpoint

Tool abstractions are an important part of building a data agent. These abstractions define the set of APIs or tools that the agent can interact with. The agent uses a reasoning loop to decide which tools to use, in what sequence, and the parameters to call each tool.

To use data agents in LlamaIndex, you can follow the usage pattern below:

```python
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

# Initialize LLM
llm = OpenAI(model="gpt-3.5-turbo-0613")

# Initialize OpenAI agent
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)
```

Overall, data agents in LlamaIndex provide a powerful way to interact with and manipulate data, making them valuable tools for various applications.

### Advanced features

LlamaIndex provides several advanced features that cater to the needs of advanced users. Some of these advanced features include:

- **Customization and Extension**: LlamaIndex offers lower-level APIs that allow advanced users to customize and extend any module within the framework. This includes data connectors, indices, retrievers, query engines, and reranking modules. Users can tailor these components to fit their specific requirements and enhance the functionality of LlamaIndex.
- **Data Agents**: LlamaIndex includes LLM-powered knowledge workers called Data Agents. These agents can intelligently perform various tasks over data, including automated search and retrieval. They can read from and modify data from different tools, making them versatile for data manipulation. Data Agents consist of a reasoning loop and tool abstractions, enabling them to interact with external service APIs and process responses.
- **Application Integrations**: LlamaIndex allows for seamless integration with other applications in your ecosystem. Whether it's LangChain, Flask, or ChatGPT, LlamaIndex can be integrated with various tools and frameworks to enhance its functionality and extend its capabilities.
- **High-Level API**: LlamaIndex provides a high-level API that allows beginners to quickly ingest and query their data with just a few lines of code. This user-friendly interface simplifies the process for beginners while still providing powerful functionality.
- **Modular Architecture**: LlamaIndex follows a modular architecture, which allows users to understand and work with different components of the framework independently. This modular approach enables users to customize and combine different modules to create tailored solutions for their specific use cases.

## LiteLLM

![litellm image](https://user-images.githubusercontent.com/81156510/266440975-6c29c205-fc6b-47c1-955e-c5b9082828e7.png)

As the name suggests a light package that simpifies the task of getting the responses form multiple APIs at the same time without having to worry about the imports is known as the [LiteLLM](https://litellm.ai). It is available as a python package which can be accessed using `pip` Besides we can test the working of the library using the [playground](https://litellm.ai/playground) that is readily available.

### Completions

This is similar to OpenAI `create_completion()` [method](https://docs.litellm.ai/docs/completion/input) that allows you to call various available LLMs in the same format. LiteLLMs gives the flexibility to fine-tune the models but there is a catch, only on a few parameters.
There is also [batch completion](https://docs.litellm.ai/docs/completion/batching) possible which helps us to process multiple prompts simultaneously.

### Embeddings & Providers

There is not much to talk about regarding [embeddings](https://docs.litellm.ai/docs/embedding/supported_embedding) but worth mentioning. We have access to OpenAI and Azure OpenAI embedding models which support `text-embedding-ada-002`.

But when we jump to [providers](https://docs.litellm.ai/docs/providers), there is quite a doze of support providers. Though the list not comprehensive yet it supports the most commonly used providers like HuggingFace, Cohere, OpenAI, Replicate, Anthropic, etc.

### Streaming Queries

By setting the `stream=True` parameter to boolean `True` we can view the [streaming](https://docs.litellm.ai/docs/#streaming-queries) iterator response in the output. But this is currently supported for models like OpenAI, Azure, Anthropic, and Huggingface.

## Comparison Table for LLM-SDKs

| Model 𝌭 | Use case ⌂ | Vector stores🏬 | Embedding model 🖇️ | LLM Model ⪭ | Advanced features ➹ |
| :---: | :-: | --- | --- | :-: | --- |
|  Langchain  | When you need alternatives to the same solution and not worried about tech stack constraints | Comprehensive list of data sources available to get connected readily | State of art embedding models in the bucket to choose from | A-Z availability of LLMs out there in the market | Open source & 1.5k+ contributors strong for active project development |
|  Llama index  | Suited for structured, semi-structured as well as unstructured data | Wide options to connect & facility to [create a new one](https://gpt-index.readthedocs.io/en/latest/examples/vector_stores/CognitiveSearchIndexDemo.html#create-index-if-it-does-not-exist) | Besides the 3 commonly available models we can use a [custom embedding model](https://gpt-index.readthedocs.io/en/latest/examples/embeddings/custom_embeddings.html) as well | Set of restricted availability of LLM models besides [customized abstractions](https://gpt-index.readthedocs.io/en/latest/core_modules/model_modules/llms/usage_custom.html) suited for your custom data | Tailor-made for high customizations if not happy with the current parameters and integrations |
|  Litellm  | When a call has to be made to multiple LLMs for the same prompt at the same time | Not Applicable  | Currently supports only `text-embedding-ada-002` from OpenAI & Azure | Expanding the list of LLM providers with the most commonly used ones ready for use | Lightweight and consistent output response |

{{ comments }}
See also: https://github.com/huggingface/transformers/blob/main/awesome-transformers.md
