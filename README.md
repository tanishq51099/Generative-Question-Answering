# Generative Question Answering Using RAG
=======
Base Model - T5, Llama2

Detailed Results - https://docs.google.com/spreadsheets/d/1s9FNl4oVRXnhS3HE6_Na75iQXlE09eexE6i85Zyqu1w/edit?usp=sharing

## RAG Architecture

![image](https://github.com/tanishq51099/Generative-Question-Answering/assets/20563702/3a24ea24-65cd-4588-a5f0-320230e19a21)


## Implementation:
The learning examples consist of questions paired with their respective contexts from the SQuAD dataset. Prompts are designed to facilitate accurate answer generation. 

RETRIEVER : We implement both sparse (TF-IDF, BM25) and dense (DPR) retrieval methods to enhance the relevance of retrieved passages. We have employed 3 Methods to compare and decide which model and method combination gives better results for retrieval. 

### Method 1 - 
* We curated and preserved all context passages from the training and development sets of the SQuAD dataset as our external data contexts. These external contexts were subsequently embedded and stored in a vector database.
* Each passage from the SQuAD dataset was treated as an individual chunk. This granularity allowed us to process and analyze each passage independently.
* To identify relevant contexts, we converted the user query using the same embedding function.
* By calculating the cosine similarity between the query and all passages, we selected the most pertinent passages from our external context pool.
* Embedding function (encoder) :
  - all-distilroberta-v1, multi-qa-mpnet-base-dot-v1 and all-MiniLM-L6-v2 (default)
  - Sparse Retriever: BM25, TFIDF
* The top 5, top 3, and top 1 passage was then selected and given to the Generator model.

![image](https://github.com/tanishq51099/Generative-Question-Answering/assets/114322584/b9003815-56cd-468c-aa6e-182ae88ab5a7)


### Method 2 (Using Cross Encoder Ranker)- 
* A cross-encoder ranking method was then used to improve the accuracy of the retrieval model.
* In this method, the first 10 passages were retrieved using the dense passage retriever and then ranked based on a cross-encoder method.
* The cross-encoder method takes the user query and all the retrieved passages (in this case, 10) as input. It then computes similarity scores between the query and each retrieved passage.
* The transformer model serves as the backbone for our cross-encoder approach:
* - Dense Passage Retriever: ms-marco-MiniLM-L-6-v2
  - Sparse Retriever: All-MiniLM-L6-v2
* The top 5, top 3, and top 1 passage was then selected and given to the Generator model.

![image](https://github.com/tanishq51099/Generative-Question-Answering/assets/114322584/48f0ce5e-3a09-4260-9916-211b6c9f4356)


### Method 3 (Using Cross Encoder Ranker with Query Expansion)-
* To further increase the accuracy of the retriever model, query expansion is implemented.
* In this method, the query was first given to an LLM model (in this case MISTRAL 7B was used) and a prompt was given to suggest an answer to the query. This provided relevant context.
* This generated answer is then augmented with the query and given to the retriever model.
* These top-ranked passages received from the encoder are then forwarded to the generation stage.

![image](https://github.com/tanishq51099/Generative-Question-Answering/assets/114322584/48161c0a-c6c7-4dc1-8bfc-d6cb68b04b2c)


## GENERATOR
We utilize T5 and Llama 2 as the generators within the RAG framework. Fine-tuning is applied to adapt the T5 model specifically for the SQuAD dataset. Various prompt engineering techniques are employed to guide the model in generating precise answers.

**Prompt 1**: “Answer the question based on context provided: question:’ ’, context: ‘ ’”

**Prompt 2 (or NA prompt)**: “Answer the question based on the context provided. If there is no answer in the context, respond with 'no answer'. Question:’ ’, context: ‘“

**Prompt 3**: "Answer the question based on the context provided. The answer should be a phrase within the context, if there is no answer within the context, respond with 'no answer'. "

**Prompt (Llama2)**: “[INST]<<SYS>> Answer the question based on only the context provided. The answer should be a phrase within the context, if there is no answer within the context, respond with 'no answer'. Don't write sentences. Just give answers in minimum words. Don't write 'The answer is.'<</SYS>> question:{prompt}, context:{context} [/INST]"

T5 (Base + Prompt) :    “t5-base” model with prompt1

T5 (Base + NA Prompt) :    “t5-base” model with prompt2

T5 (Fine Tuned(2)):    Fine-tuned “t5-base” trained on Squad’s Training set (n_epochs=2) without any prompt.

T5 (Fine-Tuned(2) + Prompt) :    Fine-tuned “t5-base”.......... with prompt1 (n_epochs=2)

T5 (Fine-Tuned(2) + NA Prompt) :    Fine-tuned “t5-base”...... with prompt2 (n_epochs=2)

Llama2 (Base + NA Prompt) :    “Llama2” model with Llama2 prompt

### Fine Tuning
We chose the T5-base model from HuggingFace for our language model because of its extended context length limit and because it hadn't been pre-trained on SquAD, which would prevent bias. Initial tests without context yielded 0 accuracy, highlighting its limitations. Even with context, the base model struggled with Exact Match due to uncertainty in answer size. To address this, we fine-tuned T5 on SquAD's training set, resulting in a remarkable improvement across all metrics, with Exact Match rising from 28% to 65%.

### Prompt Engineering
Following this, we devised various prompts, including Prompt 1, Prompt 2, Prompt 3, and the Llama Prompt, to ensure robust performance on questions lacking an explicit answer in the context. Prompt 2, instructing the model to return "no answer" when appropriate, notably improved T5's Exact Match by 1%. Prompt 3 aimed to extract exact phrases from the paragraph, theoretically enhancing Exact Match, although initial results suggest otherwise.
For Llama2, we used the same prompt of returning “no answer” and wrote in the prompt format specific to Llama2.

=======

## Installation

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.

## Usage

- Run `main.py` to train and evaluate the model.

## Credits

- Developed by Parth Maheshwari, Sahithi Sane, Tanishq Tanmay.
