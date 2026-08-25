# Experiment 01 — Basic RAG

## Objective

Build the smallest complete RAG pipeline without an orchestration framework.

The experiment makes the individual architectural components visible:

```text
Documents
    ↓
Chunking
    ↓
Embedding model
    ↓
Vector store
    ↓

User query
    ↓
Query embedding
    ↓
Similarity search
    ↓
Top-k chunks
    ↓
Augmented prompt
    ↓
LLM
    ↓
Grounded answer
```

This directly maps to the learning plan's Day 5 RAG objective: document ingestion,
chunking, embeddings, vector stores, similarity search, top-k retrieval, metadata,
grounding and generation.

## Why this is a baseline

This experiment is intentionally simple. Its purpose is to establish a **control
group** before changing individual RAG architecture variables.

The subsequent experiments will vary:

1. Chunk size
2. Chunk overlap
3. Top-k
4. Metadata filtering
5. Dense vs sparse vs hybrid retrieval
6. Reranking
7. Contextual retrieval
8. Retrieval evaluation
9. Generation evaluation

The important principle is:

> Change one architectural variable at a time so that the effect of the change
> can be observed.

---

# 1. Experimental Architecture

The implementation deliberately does not use LangChain, LlamaIndex or another
orchestration framework.

```text
                    INGESTION / INDEXING

       8 policy documents
                │
                ▼
           Simple chunker
                │
                ▼
        8 document chunks
                │
                ▼
     BAAI/bge-small-en-v1.5
                │
                ▼
       384-dimensional vectors
                │
                ▼
          Chroma vector store


                       QUERY

          User question
                │
                ▼
     Same embedding model
                │
                ▼
          Query vector
                │
                ▼
      Chroma similarity search
                │
                ▼
             Top-k = 3
                │
                ▼
       Retrieved text chunks
                │
                ▼
       Augmented LLM prompt
                │
          ┌─────┴─────┐
          ▼           ▼
       Qwen 3B     Gemma 1B
          │           │
          └─────┬─────┘
                ▼
             Answer
```

A key architectural distinction demonstrated by this experiment is:

```text
Embedding model
    ↓
Retrieval

LLM
    ↓
Generation
```

The generative LLM is **not involved in the vector retrieval step**.

---

# 2. Corpus

The experiment uses eight fictional fintech/regulatory-style policy documents:

- Customer Verification
- Transaction Limits
- Password Reset
- Refunds
- Fraud Monitoring
- Data Retention
- Complaints
- Access Control

The corpus intentionally resembles an enterprise knowledge base while remaining
small enough to inspect manually.

The resulting chunk file confirms:

```text
Documents: 8
Chunks:    8
```

Each source document currently produces one chunk. This is an intentional
baseline limitation rather than a production chunking strategy.

---

# 3. Implementation Choices

## Embedding model

`BAAI/bge-small-en-v1.5`

This is a local Sentence Transformers embedding model used for both document
and query embeddings.

The model produces 384-dimensional vectors.

## Vector store

Chroma with a local persistent collection.

The vector store is responsible for storing/indexing the embeddings and
performing similarity search.

## LLMs tested

Two local Ollama models were tested:

- `qwen2.5:3b`
- `gemma3:1b`

Testing two models is useful because it demonstrates that changing the
**generation model** does not change retrieval when the embedding model and
retrieval configuration remain unchanged.

## Retrieval

- Dense vector retrieval
- Cosine similarity
- `top_k = 3`

## Chunking

A simple word-based chunker.

The current baseline produces one chunk per document, so chunking has not yet
been meaningfully stressed. Chunking becomes the focus of Experiment 02.

---

# 4. Test Questions

Two deliberately contrasting questions were used.

## Valid / in-domain question

> What happens when a customer forgets their password and fails the identity checks?

The answer exists in `03-password-reset.md`.

## Invalid / out-of-domain question

> What is the maximum mortgage amount a customer can borrow?

The corpus contains no mortgage policy or mortgage amount.

This second query is important because it tests the distinction between:

```text
retrieving the nearest available chunks
```

and:

```text
retrieving evidence that actually answers the question
```

---

# 5. Results

## 5.1 Corpus and retrieval configuration

| Parameter | Result |
|---|---|
| Documents | 8 |
| Chunks | 8 |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Embedding dimension | 384 |
| Vector store | Chroma |
| Retrieval type | Dense vector similarity |
| Top-k | 3 |
| LLMs tested | `qwen2.5:3b`, `gemma3:1b` |

The chunk output confirms that the eight source documents currently produce eight
chunks. Each document is therefore represented by one vector in the baseline.

---

# 6. Valid Question Results

### Query

> What happens when a customer forgets their password and fails the identity checks?

Both LLM runs produced **identical retrieval results**:

| Rank | Source | Similarity |
|---:|---|---:|
| 1 | `03-password-reset.md` | 0.8035 |
| 2 | `01-customer-verification.md` | 0.7514 |
| 3 | `08-access-control.md` | 0.6214 |

The top result is exactly the policy that contains the answer.

The retrieval results are identical for Qwen and Gemma because both runs use the
same embedding model, query embedding, vector store and retrieval configuration.
The generative model is only invoked after retrieval.

Qwen generated:

> When a customer forgets their password and fails the identity checks in the
> password reset policy, the self-service reset must not proceed. The customer
> should be directed to the supported recovery process.

Gemma generated the same substantive answer, but more concisely:

> If the identity checks fail, the self-service reset must not proceed. The
> customer should be directed to the supported recovery process.

Both answers are supported by the retrieved context.

---

# 7. Invalid Question Results

### Query

> What is the maximum mortgage amount a customer can borrow?

There is no mortgage information in the corpus.

Nevertheless, vector retrieval still returned three results.

Both Qwen and Gemma produced the **same retrieval results**:

| Rank | Source | Similarity |
|---:|---|---:|
| 1 | `02-transaction-limits.md` | 0.6917 |
| 2 | `04-refunds.md` | 0.5260 |
| 3 | `01-customer-verification.md` | 0.5235 |

This is a particularly important result.

The retriever is performing a **nearest-neighbour search**. It is not answering
the question:

> "Does the corpus contain the answer?"

It is effectively answering:

> "Which available vectors are closest to this query vector?"

Therefore, even an out-of-domain question receives nearest-neighbour results.

Both LLMs nevertheless correctly abstained:

> I don't have enough information in the supplied documents.

This demonstrates that retrieval and generation need to be evaluated separately.

---

# 8. Qwen vs Gemma — What Changed and What Did Not?

This experiment initially raised a question about whether changing the LLM
should change the retrieval scores.

It should **not**, and the repeated invalid-query run confirmed this.

The pipeline is:

```text
                         QUERY
                           │
                           ▼
                BGE embedding model
                           │
                           ▼
                     Query vector
                           │
                           ▼
                 Chroma similarity
                           │
                           ▼
                    Top-k chunks
                           │
                           ▼
                ┌──────────┴──────────┐
                ▼                     ▼
             Qwen 3B              Gemma 1B
                │                     │
                ▼                     ▼
             Answer                Answer
```

Therefore:

- changing Qwen → Gemma does not change query embeddings
- it does not change vector similarity
- it does not change retrieved chunks
- it only changes the generation stage

This was demonstrated by the identical retrieval rankings and similarity scores
for both models on both questions.

---

# 9. Timing Results

The experiment was updated to record timing at individual stages.

## Valid question

| Stage | Qwen 3B | Gemma 1B |
|---|---:|---:|
| Document embedding | 5.137s | 4.289s |
| Query embedding | 0.071s | 0.059s |
| Vector retrieval | 0.002s | 0.001s |
| Similarity calculation | 0.000s | 0.000s |
| Prompt construction | 0.000s | 0.000s |
| Generation | 2.343s | 1.184s |
| End-to-end retrieval + generation | 2.416s | 1.245s |

## Invalid question

| Stage | Qwen 3B | Gemma 1B |
|---|---:|---:|
| Document embedding | 4.401s | 4.251s |
| Query embedding | 0.059s | 0.081s |
| Vector retrieval | 0.002s | 0.001s |
| Similarity calculation | 0.000s | 0.000s |
| Prompt construction | 0.000s | 0.000s |
| Generation | 0.556s | 0.262s |
| End-to-end retrieval + generation | 0.617s | 0.344s |

### Important observation

The dominant latency in these runs is **LLM generation**, not vector retrieval.

For example, for the valid Qwen run:

```text
Total:       2.416s
Generation:  2.343s
Retrieval:   0.002s
```

For the valid Gemma run:

```text
Total:       1.245s
Generation:  1.184s
Retrieval:   0.001s
```

Therefore, in this tiny local experiment:

```text
Retrieval latency
        ↓
     negligible

Generation latency
        ↓
     dominant
```

This should **not** be generalised to production RAG. The corpus is tiny, the
vector store is local, there is no network hop, and there is no reranking.
Later experiments will deliberately introduce additional retrieval work.

The document-embedding time is also indexing/setup work rather than per-query
retrieval latency. It should therefore be considered separately from the
user-facing query path.

---

# 10. Key Learnings

## 10.1 RAG is a pipeline of separate architectural components

The experiment made the following separation concrete:

```text
Documents
   ↓
Embedding
   ↓
Vector store
   ↓
Retrieval
   ↓
Context
   ↓
LLM
   ↓
Generation
```

The embedding model is not the LLM, and the vector store is not the LLM.

---

## 10.2 The embedding model controls retrieval, not the generation model

Changing:

```text
Qwen → Gemma
```

did not change retrieval because both runs used:

```text
BAAI/bge-small-en-v1.5
```

for document and query embeddings.

This is an important architectural distinction when selecting models for a RAG
system.

---

## 10.3 Similarity search does not determine whether an answer exists

The mortgage query has no answer in the corpus, yet the retriever returned:

```text
Transaction Limits
Refunds
Customer Verification
```

This is not necessarily a malfunction.

Nearest-neighbour retrieval will still identify the closest available vectors.

Therefore:

> A retrieved result is not automatically evidence that the corpus contains
> the answer.

---

## 10.4 Similarity score is a ranking signal, not an answer-confidence score

The invalid mortgage query produced a top similarity of:

```text
0.6917
```

while the valid password query produced:

```text
0.8035
```

The higher score coincided with a clearly relevant result in this small
experiment, but this does **not** establish a universal relevance threshold.

For example, we should not assume:

```text
score > 0.70 → relevant
score < 0.70 → irrelevant
```

A threshold would need to be established empirically for a particular
embedding model, corpus and retrieval setup.

---

## 10.5 Retrieval quality and generation quality are different

The valid question demonstrated good retrieval:

```text
Correct document
      ↓
Rank 1
      ↓
Good answer
```

The invalid question demonstrated:

```text
No answer exists
      ↓
Nearest chunks retrieved
      ↓
LLM abstains
```

Therefore, RAG evaluation needs to distinguish at least:

```text
Retrieval quality
        +
Generation quality
```

A good LLM cannot compensate for retrieval that consistently misses the
required evidence.

---

## 10.6 Prompt grounding can encourage abstention

The invalid question resulted in:

> I don't have enough information in the supplied documents.

for both tested models.

This demonstrates that the generation layer can be instructed to use only the
supplied context and abstain when evidence is insufficient.

However, this experiment does **not** prove that the system is hallucination
proof. We have not yet tested conflicting, misleading, partially relevant or
incorrect retrieved context.

---

## 10.7 Retrieval latency and generation latency are different architectural concerns

The timing instrumentation showed:

```text
Query embedding
      ↓
Vector retrieval
      ↓
Prompt construction
```

taking very little time in this local experiment, while LLM generation dominated
the end-to-end query time.

This provides the first concrete introduction to the later production concerns
of:

- latency
- model selection
- model routing
- reranking overhead
- caching
- cost

---

## 10.8 The current corpus does not yet exercise chunking

The experiment currently has:

```text
8 documents
↓
8 chunks
```

Each document is one chunk.

Therefore, Experiment 01 demonstrates the RAG pipeline but does not yet
demonstrate the real retrieval trade-offs introduced by chunking.

This is intentional.

---

# 11. Limitations

This experiment is intentionally small and should not be treated as a
production RAG architecture.

### Corpus

Only eight short fictional documents are used.

### Chunking

One chunk per document means chunk boundaries have not yet been stress-tested.

### Retrieval

Only dense vector retrieval is implemented.

There is no:

- sparse retrieval
- hybrid search
- RRF
- metadata filtering
- reranking
- contextual retrieval

### Evaluation

Only two queries were used for this baseline.

No formal:

- Recall@K
- Precision@K
- MRR
- groundedness score
- LLM-as-judge

has been implemented yet.

### Performance

The measured timings are local experimental timings and should not be used as
production benchmarks.

---

# 12. What This Experiment Has Demonstrated

The basic RAG flow can now be understood as:

```text
                    INDEXING

Documents
   │
   ▼
Chunking
   │
   ▼
Embedding model
   │
   ▼
Vector representations
   │
   ▼
Vector store


                    QUERY

Question
   │
   ▼
Query embedding
   │
   ▼
Similarity search
   │
   ▼
Top-k candidate chunks
   │
   ▼
Prompt augmentation
   │
   ▼
Generative LLM
   │
   ▼
Grounded answer
```

The most important architectural insight from this baseline is:

> **Retrieval finds candidate evidence; generation uses that evidence to
> produce an answer. They are separate stages and therefore need separate
> evaluation.**

---