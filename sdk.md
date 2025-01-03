# Software Development toolKits

{term}`LLM` SDKs are specific for generative AI. These toolkits help developers integrate LLM capabilities into applications. The LLM SDK typically includes APIs, sample code, and documentation to aid in the development process. By leveraging an LLM SDK, developers can streamline their development processes and ensure compliance with industry standards.

% TODO: haystack?

```{table} Comparison of LLM SDKs
:name: llm-sdks
SDK | Use cases | Vector stores | Embedding model | LLM Model | Languages | Features
----|-----------|---------------|-----------------|-----------|-----------|----------
[](#langchain) | Chatbots, prompt chaining, document related tasks | Comprehensive list of data sources available to get connected readily | State of art embedding models in the bucket to choose from | A-Z availability of LLMs out there in the market | Python, Javascript, Typescript | Open source & 1.5k+ contributors strong for active project development
[](#llama-index) | Connecting multiple data sources to LLMs, document query interface using retrieval augmented generation, advanced chatbots, structured analytics | Wide options to connect & facility to [create a new one](https://docs.llamaindex.ai/en/latest/examples/vector_stores/CognitiveSearchIndexDemo.html#create-index-if-it-does-not-exist) | Besides the 3 commonly available models we can use a [custom embedding model](https://docs.llamaindex.ai/en/latest/examples/embeddings/custom_embeddings.html) as well | Set of restricted availability of LLM models besides [customised abstractions](https://docs.llamaindex.ai/en/latest/module_guides/models/llms/usage_custom.html) suited for your custom data | Python, Javascript, Typescript | Tailor-made for high customisations if not happy with the current parameters and integrations
[](#litellm) | Integrating multiple LLMs, evaluating LLMs | Not Applicable | Currently supports only `text-embedding-ada-002` from OpenAI & Azure | Expanding the list of LLM providers with the most commonly used ones ready for use | Python | Lightweight, streaming model response, consistent output response
```

{{ table_feedback }}

```{seealso}
[awesome-transformers](https://github.com/huggingface/transformers/blob/main/awesome-transformers.md)
```

A few reasons for why there is a need for LLM SDKs in this current era of AI.

1. **Compliance with Agreements**: By using an LLM SDK, developers can ensure that their application complies with agreements by logging, tracing, and monitoring requests appropriately. This helps avoid potential legal issues related to software piracy or unauthorised use of resources.
1. **Improved User Experience**: An LLM SDK can help create a seamless user experience by removing boilerplate code and abstracting lower level interactions with LLMs.
1. **Increased Security**: By implementing an LLM SDK, developers can protect their resources and prevent unauthorised use of their software by security features such as [access control and user management](https://www.businesswire.com/news/home/20230531005251/en/LlamaIndex-Raises-8.5M-to-Unlock-Large-Language-Models-Capabilities-with-Personal-Data).
1. **Flexibility**: An LLM SDK provides flexibility in terms of customisation and bringing together different components, allowing developers to tailor the management system to their specific needs and adapt it easily.
1. **Improved Collaboration**: An LLM SDK can facilitate collaboration among team members by providing a centralised platform for license management, ensuring that everyone is on the same page regarding issues and compliance requirements.

## LangChain

![banner](https://python.langchain.com/img/parrot-chainlink-icon.png)

On the LangChain page -- it states that LangChain is a framework for developing applications powered by Large Language Models(LLMs). It is available as an python sdk and npm packages suited for development purposes.

### Document Loader

Well the beauty of LangChain is we can take input from various different files to make it usable for a great extent. Point to be noted is they can be of various [formats](https://python.langchain.com/docs/modules/data_connection/document_loaders) like `.pdf`, `.json`, `.md`, `.html`, and `.csv`.

### Vector Stores

After collecting the data they are converted in the form of embeddings for the further use by storing them in any of the vector database.
Through this way we can perform vector search and retrieve the data from the embeddings that are very much close to the embed query.

The list of vector stores that LangChain supports can be found [here](https://python.langchain.com/docs/integrations/vectorstores).

### Models

This is the heart of most LLMs, where the core functionality resides. There are broadly [2 different types of models](https://python.langchain.com/docs/modules/model_io) which LangChain integrates with:

- **Language**: Inputs & outputs are `string`s
- **Chat**: Run on top of a Language model. Inputs are a list of chat messages, and output is a chat message

### Tools

[Tools](https://python.langchain.com/docs/modules/agents/tools) are interfaces that an agent uses to interact with the world. They connect real world software products with the power of LLMs. This gives more flexibility, the way we use LangChain and improves its capabilities.

### Prompt engineering

Prompt engineering is used to generate prompts for the custom prompt template. The custom prompt template takes in a function name and its corresponding source code, and generates an English language explanation of the function.

To create prompts for prompt engineering, the LangChain team uses a custom prompt template called `FunctionExplainerPromptTemplate`. This template takes the function name and source code as input variables and formats them into a prompt. The prompt includes the function name, source code, and an empty explanation section.
The generated prompt can then be used to guide the language model in generating an explanation for the function.

Overall, prompt engineering is an important aspect of working with language models as it allows us to shape the model's responses and improve its performance in specific tasks.

More about all the prompts can be found [here](https://python.langchain.com/docs/modules/model_io/prompts).

### Advanced features

LangChain provides several advanced features that make it a powerful framework for developing applications powered by language models. Some of the advanced features include:

- **Chains**: LangChain provides a standard interface for chains, allowing developers to create sequences of calls that go beyond a single language model call. This enables the chaining together of different components to create more advanced use cases around language models.
- **Integrations**: LangChain offers integrations with other tools, such as the `requests` and `aiohttp` integrations for tracing HTTP requests to LLM providers, and the `openai` integration for tracing requests to the OpenAI library. These integrations enhance the functionality and capabilities of LangChain.
- End-to-End Chains: LangChain supports end-to-end chains for common applications. This means that developers can create complete workflows or pipelines that involve multiple steps and components, all powered by language models. This allows for the development of complex and sophisticated language model applications.
- **Logs and Sampling**: LangChain provides the ability to enable log prompt and completion sampling. By setting the `DD_LANGCHAIN_LOGS_ENABLED=1` environment variable, developers can generate logs containing prompts and completions for a specified sample rate of traced requests. This feature can be useful for debugging and monitoring purposes.
- **Configuration Options**: LangChain offers various configuration options that allow developers to customize and fine-tune the behaviour of the framework. These configuration options are documented in the APM Python library documentation.

Overall, LangChain's advanced features enable developers to build advanced language model applications with ease and flexibility. Some limitations of LangChain are that while it is useful for rapid prototyping of LLM applications, scalability and deploying in production remains a concern - it might not be particularly useful for handling a large number of users simultaneously, and maintaining low latency.

## LLaMA Index

![banner](https://static.premai.io/book/sdk-llama-index.jpg)

LLaMAIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data. It provides tools such as data connectors, data indexes, engines (query and chat), and data agents to facilitate natural language access to data. LLaMAIndex is designed for beginners, advanced users, and everyone in between, with a high-level API for easy data ingestion and querying, as well as lower-level APIs for customisation. It can be installed using `pip` and has detailed [documentation](https://docs.llamaindex.ai/en/latest) and tutorials for getting started. LLaMAIndex also has associated projects like https://github.com/run-llama/llama-hub and https://github.com/run-llama/llama-lab.

### Data connectors

[Data connectors](https://docs.llamaindex.ai/en/latest/module_guides/loading/connector/root.html) are software components that enable the transfer of data between different systems or applications. They provide a way to extract data from a source system, transform it if necessary, and load it into a target system. Data connectors are commonly used in data integration and ETL (Extract, Transform, Load) processes.

There are various types of data connectors available, depending on the specific systems or applications they connect to. Some common ones include:

- **Database connectors**: These connectors allow data to be transferred between different databases, such as MySQL, PostgreSQL, or Oracle.
- **Cloud connectors**: These connectors enable data transfer between on-premises systems and cloud-based platforms, such as Amazon Web Services (AWS), Google Cloud Platform (GCP), or Microsoft Azure.
- **API connectors**: These connectors facilitate data exchange with systems that provide APIs (Application Programming Interfaces), allowing data to be retrieved or pushed to/from those systems.
- **File connectors**: These connectors enable the transfer of data between different file formats, such as PDF, CSV, JSON, XML, or Excel.
- **Application connectors**: These connectors are specifically designed to integrate data between different applications, such as CRM (Customer Relationship Management) systems, ERP (Enterprise Resource Planning) systems, or marketing automation platforms.

Data connectors play a crucial role in enabling data interoperability and ensuring seamless data flow between systems. They simplify the process of data integration and enable organisations to leverage data from various sources for analysis, reporting, and decision-making purposes.

### Data indexes

[Data indexes](https://docs.llamaindex.ai/en/latest/module_guides/indexing/indexing.html) in LLaMAIndex are intermediate representations of data that are structured in a way that is easy and performant for Language Model Models (LLMs) to consume. These indexes are built from documents and serves as the core foundation for retrieval-augmented generation (RAG) use-cases.
Under the hood, indexes in LLaMAIndex store data in Node objects, which represent chunks of the original documents. These indexes also expose a Retriever interface that supports additional configuration and automation.
LLaMAIndex provides several types of indexes, including Vector Store Index, Summary Index, Tree Index, Keyword Table Index, Knowledge Graph Index, and SQL Index. Each index has its own specific use case and functionality.

To get started with data indexes in LLaMAIndex, you can use the `from_documents` method to create an index from a collection of documents. Here's an example using the Vector Store Index:

```python
from llama_index import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)
```

Overall, data indexes in LLaMAIndex play a crucial role in enabling natural language access to data and facilitating question & answer and chat interactions with the data. They provide a structured and efficient way for LLMs to retrieve relevant context for user queries.

### Data engines

Data engines in LLaMAIndex refer to the query engines and chat engines that allow users to interact with their data. These engines are end-to-end pipelines that enable users to ask questions or have conversations with their data. The broad classification of data engines are:

- [Query engine](https://docs.llamaindex.ai/en/latest/core_modules/query_modules/query_engine/root.html)
- [Chat engine](https://docs.llamaindex.ai/en/latest/core_modules/query_modules/chat_engines/root.html)

#### Query engine

- Query engines are designed for question and answer interactions with the data.
- They take in a natural language query and return a response along with the relevant context retrieved from the knowledge base.
- The LLM (Language Model Model) synthesises the response based on the query and retrieved context.
- The key challenge in the querying stage is retrieval, orchestration, and reasoning over multiple knowledge bases.
- LLaMAIndex provides composable modules that help build and integrate RAG (Retrieval-Augmented Generation) pipelines for Q&A.

#### Chat engine

- Chat engines are designed for multi-turn conversations with the data.
- They support back-and-forth interactions instead of a single question and answer.
- Similar to query engines, chat engines take in natural language input and generate responses using the LLM.
- The chat engine maintains conversation context and uses it to generate appropriate responses.
- LLaMAIndex provides different chat modes, such as "condense_question" and "react", to customise the behaviour of chat engines.

Both query engines and chat engines can be used to interact with data in various use cases. The main distinction is that query engines focus on single questions and answers, while chat engines enable more dynamic and interactive conversations. These engines leverage the power of LLMs and the underlying indexes to provide relevant and informative responses to user queries.

### Data agent

[Data Agents](https://docs.llamaindex.ai/en/latest/core_modules/agent_modules/agents/root.html) are LLM-powered knowledge workers in LLaMAIndex that can intelligently perform various tasks over data, both in a "read" and "write" function. They have the capability to perform automated search and retrieval over different types of data, including unstructured, semi-structured, and structured data. Additionally, they can call external service APIs in a structured fashion and process the response, as well as store it for later use.

Data agents go beyond query engines by not only reading from a static source of data but also dynamically ingesting and modifying data from different tools. They consist of two core components: a reasoning loop and tool abstractions. The reasoning loop of a data agent depends on the type of agent being used. LLaMAIndex supports two types of agents:

- OpenAI Function agent: built on top of the OpenAI Function API
- ReAct agent: which works across any chat/text completion endpoint

Tool abstractions are an important part of building a data agent. These abstractions define the set of APIs or tools that the agent can interact with. The agent uses a reasoning loop to decide which tools to use, in what sequence, and the parameters to call each tool.

To use data agents in LLaMAIndex, you can follow the usage pattern below:

```python
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

# Initialise LLM & OpenAI agent
llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)
```

Overall, data agents in LLaMAIndex provide a powerful way to interact with and manipulate data, making them valuable tools for various applications.

### Advanced features

LLaMAIndex provides several advanced features that cater to the needs of advanced users. Some of these advanced features include:

- **Customisation and Extension**: LLaMAIndex offers lower-level APIs that allow advanced users to customise and extend any module within the framework. This includes data connectors, indices, retrievers, query engines, and re-ranking modules. Users can tailor these components to fit their specific requirements and enhance the functionality of LLaMAIndex.
- **Data Agents**: LLaMAIndex includes LLM-powered knowledge workers called Data Agents. These agents can intelligently perform various tasks over data, including automated search and retrieval. They can read from and modify data from different tools, making them versatile for data manipulation. Data Agents consist of a reasoning loop and tool abstractions, enabling them to interact with external service APIs and process responses.
- **Application Integrations**: LLaMAIndex allows for seamless integration with other applications in your ecosystem. Whether it's LangChain, Flask, or ChatGPT, LLaMAIndex can be integrated with various tools and frameworks to enhance its functionality and extend its capabilities.
- **High-Level API**: LLaMAIndex provides a high-level API that allows beginners to quickly ingest and query their data with just a few lines of code. This user-friendly interface simplifies the process for beginners while still providing powerful functionality.
- **Modular Architecture**: LLaMAIndex follows a modular architecture, which allows users to understand and work with different components of the framework independently. This modular approach enables users to customise and combine different modules to create tailored solutions for their specific use cases.

LLaMAIndex seems more tailor made for deploying LLM apps in production. However, it remains to be seen how/whether the industry integrates LLaMAIndex in LLM apps, or develop customized methods for LLM data integration.

## LiteLLM

![banner](https://litellm.vercel.app/img/docusaurus-social-card.png)

As the name suggests a light package that simplifies the task of getting the responses form multiple APIs at the same time without having to worry about the imports is known as the [LiteLLM](https://docs.litellm.ai). It is available as a python package which can be accessed using `pip`

### Completions

This is similar to OpenAI `create_completion()` [method](https://docs.litellm.ai/docs/completion/input) that allows you to call various available LLMs in the same format. LiteLLMs gives the flexibility to fine-tune the models but there is a catch, only on a few parameters.
There is also [batch completion](https://docs.litellm.ai/docs/completion/batching) possible which helps us to process multiple prompts simultaneously.

### Embeddings & Providers

There is not much to talk about regarding [embeddings](https://docs.litellm.ai/docs/embedding/supported_embedding) but worth mentioning. We have access to OpenAI and Azure OpenAI embedding models which support `text-embedding-ada-002`.

However there are many [supported providers](https://docs.litellm.ai/docs/providers), including HuggingFace, Cohere, OpenAI, Replicate, Anthropic, etc.

### Streaming Queries

By setting the `stream=True` parameter to boolean `True` we can view the [streaming](https://docs.litellm.ai/docs/completion/stream) iterator response in the output. But this is currently supported for models like OpenAI, Azure, Anthropic, and HuggingFace.

The idea behind LiteLLM seems neat - the ability to query multiple LLMs using the same logic. However, it remains to be seen how this will impact the industry and what specific use-cases this solves.

## Future And Other SDKs

[](#langchain), [](#llama-index), and [](#litellm) have exciting future plans to unlock high-value LLM applications. [Future initiatives from Langchain](https://blog.langchain.dev/announcing-our-10m-seed-round-led-by-benchmark) include improving the TypeScript package to enable more full-stack and frontend developers to create LLM applications, improved document retrieval, and enabling more observability/experimentation with LLM applications. LlamaIndex is developing an enterprise solution to help remove technical and security barriers for data usage. Apart from the SDKs discussed, there are a variety of newer SDKs for other aspects of integrating LLMs in production. One example is https://github.com/prefecthq/marvin, great for building APIs, data pipelines, and streamlining the AI engineering framework for building natural language interfaces. Another example is https://github.com/homanp/superagent, which is a higher level abstraction and allows for building many AI applications/micro services like chatbots, co-pilots, assistants, etc.

{{ comments }}
