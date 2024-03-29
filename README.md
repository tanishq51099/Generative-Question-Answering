# Generative Question Answering Using RAG
Base Model - T5

Retrieval Model - BM25 (Sparse Retrieval Model) and TF-IDF (Sparse Retrieval Model)

Hybrid Retrieval Model - BGE-M3

The BGE-M3 model incorporates both sparse and dense retrieval methods. It’s designed to support hybrid retrieval, which combines the strengths of various methods, including dense retrieval like DPR and sparse retrieval like BM25, unicoil, and splade. This allows BGE-M3 to perform embedding and sparse retrieval efficiently, providing token weights similar to BM25 without additional cost when generating dense embeddings. Dense retrieval is used for mapping text into a single embedding vector, which is beneficial for tasks like sentence similarity. Sparse retrieval, on the other hand, calculates a weight only for tokens present in the text, which is typical for models like BM25 that are based on lexical matching. 

