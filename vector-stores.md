# Vector Databases

Vector databases have exploded in popularity in the past year due to generative AI, but the concept of vectors and embeddings has been around since modern day neural networks.

In the field of computer vision where an engineer is performing image classification, the "features" that get extracted by the neural network are the vector embeddings. These vector embeddings contain information about the image that can be used for things like image classification or image similarity. 

In the context of textual data, vector embeddings serve a similar purpose. They capture the relationship between words, which allow models to understand language.

*What does a vector embedding look like and how are they created?*

![](https://static.premai.io/book/vector-databases-architecture.jpg)
[Image source for architecture](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)

## How embeddings are created for LLMs

Large language models are trained on a massive text corpus, like Wikipedia As the model processes this text, it learns representations for words based on their context.

As the models learns from the data, it represents each word as a high-dimensional vector, usually with hundreds or thousands of dimensions. The values in the vector encode the semantic meaning of the word. 

After training on large corpora, words with similar meanings end up close together in the vector space.

The resulting word vectors capture semantic relationships between words, which allows the model to generalize better on language tasks. These pre-trained embeddings are then used to initialize the first layer of large language models like BERT.

To summarize, by training the model on a large set of text data you end up with a model specifically designed to capture the relationship between words, AKA vector embeddings.

## How does text turn into an embedding?

![](https://static.premai.io/book/vector-databases-embedding.jpeg)

### Let's take the sentence from the image above as an example: "I want to adopt a puppy".

### 1. Each word in the sentence is mapped to its corresponding vector representation using the pre-trained word embeddings. For example, "adopt" may map to a 300-dimensional vector, "puppy" to another 300-dim vector, and so on.

### 2. The sequence of word vectors is then passed through the neural network architecture of the language model.

### 3. As the word vectors pass through the model, they interact with each other and get transformed mathematically through matrix multiplications. This allows the model to interpret the meaning of the full sequence.

### 4. The output of the model is a new vector that represents the embedding for the full input sentence. This sentence embedding encodes the semantic meaning of the entire sequence of words.

Many closed-source models like [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) from OpenAI and the [embeddings model](https://docs.cohere.com/docs/embeddings) from Cohere allow developers to convert raw text into vector embeddings. It's important to not that the models used to generate vector embeddings are NOT the same models used for text generation.


```{note}
For NLP, embeddings are trained on a language modeling objective. This means they are trained to predict surrounding words/context, not to generate text. Embeddings models are encoder-only models without decoders. They output an embedding, not generated text. Generation models like GPT-2/3 have a decoder component trained explicitly for text generation.
```

## Vector Databases

Vector databases allow for efficient search and storage of vector embeddings.

| Vector Database | Open Source | Sharding | Supported Distance Metrics                  |
|-----------------|-------------|----------|---------------------------------------------|
| Weaviate        | Yes         | Yes      | cosine, dot, l2 squared, hamming, manhattan |
| Qdrant          | Yes         | Yes      | cosine, dot, euclidean                      |
| Milvus          | Yes         | Yes      | cosine, dot, euclidean, jaccard, hamming    |
| Chroma          | Yes         | N/A      | N/A                                         |
| Pinecone        | No          | Yes      | cosine, dot, euclidean                      |



## TODO: 
## Indexing the Vectors
* Overview of popular options: Weaviate, Qdrant, Milvus, Chroma, Pinecone
* Ingesting vectors and metadata
* Indexing the vectors for efficient search
* Querying the Database

## API for adding new vectors and metadata
* Query by vector to find similar embeddings
* Different query types: approximate nearest neighbor, maximum inner product search
* Retrieving closest vectors and associated metadata as results
* Optimizing Search

## Algorithmic techniques like locality-sensitive hashing for fast lookups
* Tree-based algorithms like hierarchical navigable small world graphs
* Evaluating accuracy vs. performance tradeoffs

## Semantic search over text documents/paragraphs
* Recommendation systems
* Text classification and clustering
* Any application of text embeddings
* Current Limitations and Challenges

## Scale and memory usage for huge vector datasets
* Effective ranking of results
* Query latency and throughput

## Conclusion
* Summary of how vector databases enable applications of embeddings
* Discussion of future directions and emerging techniques