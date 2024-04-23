# Generative Question Answering Using RAG
Base Model - T5, Llama2

Detailed Results - https://docs.google.com/spreadsheets/d/1s9FNl4oVRXnhS3HE6_Na75iQXlE09eexE6i85Zyqu1w/edit?usp=sharing

## Dense Passage Retriever Techniques followed
3 Methods were followed for the Retriever model -

### Method 1 - 
* We curated and preserved all context passages from the training and development sets of the SQuAD dataset as our external data contexts. These external contexts were subsequently embedded and stored in a vector database.
* Each passage from the SQuAD dataset was treated as an individual chunk. This granularity allowed us to process and analyze each passage independently.
* To identify relevant contexts, we converted the user query using the same embedding function.
* By calculating the cosine similarity between the query and all passages, we selected the most pertinent passages from our external context pool.
Embedding function (encoder) : all-distilroberta-v1, multi-qa-mpnet-base-dot-v1 and all-MiniLM-L6-v2 (default).
* The top 5, top 3, and top 1 passage was then selected and given to the Generator model.

![image](https://github.com/tanishq51099/Generative-Question-Answering/assets/114322584/b9003815-56cd-468c-aa6e-182ae88ab5a7)


### Method 2 (Using Cross Encoder Ranker)- 
* A cross-encoder ranking method was then used to improve the accuracy of the retrieval model.
* In this method, the first 10 passages were retrieved using the dense passage retriever and then ranked based on a cross-encoder method.
* The cross-encoder method takes the user query and all the retrieved passages (in this case, 10) as input. It then computes similarity scores between the query and each retrieved passage.
* The transformer model serves as the backbone for our cross-encoder approach.
Dense Passage Retriever: ms-marco-MiniLM-L-6-v2
* The top 5, top 3, and top 1 passage was then selected and given to the Generator model.

![image](https://github.com/tanishq51099/Generative-Question-Answering/assets/114322584/48f0ce5e-3a09-4260-9916-211b6c9f4356)


### Method 3 (Using Cross Encoder Ranker with Query Expansion)-
* To further increase the accuracy of the retriever model, query expansion is implemented.
* In this method, the query was first given to an LLM model (in this case MISTRAL 7B was used) and a prompt was given to suggest an answer to the query. This provided relevant context.
* This generated answer is then augmented with the query and given to the retriever model.
* These top-ranked passages received from the encoder are then forwarded to the generation stage.

![image](https://github.com/tanishq51099/Generative-Question-Answering/assets/114322584/48161c0a-c6c7-4dc1-8bfc-d6cb68b04b2c)
