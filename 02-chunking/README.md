# Experiment 02 — Chunking Strategies in RAG

## 1. Objective

This experiment investigates how **chunking strategy affects semantic retrieval in a RAG system**.

The experiment keeps the following constant:

- Embedding model: `BAAI/bge-small-en-v1.5`
- Top-k: `3`
- Documents: 4 policy-style Markdown documents
- Queries: 4 factual questions
- Similarity metric: cosine similarity

Only the way documents are divided into retrieval units changes.

The experiment compares:

1. Whole-document chunks
2. Fixed-size chunks
3. Fixed-size chunks with overlap
4. Paragraph-based chunks

The fixed-size experiments were run with several configurations:

| Run | Chunk size | Overlap |
|---|---:|---:|
| Run 1 | 60 words | 15 words |
| Run 2 | 30 words | 5 words |
| Run 3 | 30 words | 8 words |
| Run 4 | 100 words | 25 words |

This makes the experiment useful not only for understanding **chunk size**, but also for observing the effect of **overlap** and the trade-off between retrieval precision, context, corpus size and redundancy.

---

## 2. RAG Context

Chunking sits between document ingestion and embedding:

```text
Documents
    │
    ▼
Ingestion
    │
    ▼
Chunking  ◄── this experiment
    │
    ▼
Embedding
    │
    ▼
Vector Store
    │
    ▼
User Query
    │
    ▼
Query Embedding
    │
    ▼
Similarity Search
    │
    ▼
Top-k Chunks
    │
    ▼
LLM Context
```

The key architectural idea is:

> **A chunk is the unit of knowledge that the retrieval system can independently return to the LLM.**

Therefore, chunking is not merely a preprocessing detail. It directly influences what information can be retrieved and supplied to generation.

---

## 3. Dataset

The experiment uses four small policy documents:

- `mortgage-policy.md`
- `payments-policy.md`
- `data-retention-policy.md`
- `access-control-policy.md`

The four queries are:

1. **What is the maximum mortgage LTV for a first-time buyer?**
2. **What happens when a mortgage application exceeds the standard LTV limit?**
3. **What approval is required for a high-value payment?**
4. **How long must customer records be retained?**

These queries deliberately test different retrieval situations:

- a precise factual value
- an exception/procedure
- an approval rule
- a retention requirement

---

# 4. Chunking strategies

## 4.1 Whole-document chunking

Each document remains one retrieval unit.

```text
Document
└── One large chunk
```

For the four documents:

```text
4 documents → 4 chunks
```

### Advantage

The chunk contains all the document's context.

### Disadvantage

A query about one small fact retrieves the entire document, including unrelated information.

---

## 4.2 Fixed-size chunking

The document is split into chunks based on a fixed word count.

Example:

```text
30 words

Chunk 0: words 1–30
Chunk 1: words 31–60
Chunk 2: words 61–90
...
```

This creates much smaller retrieval units.

### Advantage

A highly specific query can retrieve a focused piece of information.

### Disadvantage

Important information can be split across a boundary.

---

## 4.3 Fixed-size chunking with overlap

Overlap repeats some words from the previous chunk.

For example:

```text
Chunk 0:
[1 ................. 30]

Chunk 1:
       [23 ................. 52]
        └── 8-word overlap ──┘
```

The purpose is to reduce the chance that a concept is broken at a chunk boundary.

The experiments tested:

- 60 / 15
- 30 / 5
- 30 / 8
- 100 / 25

where the notation is:

```text
chunk size / overlap
```

---

## 4.4 Paragraph chunking

Instead of using a word count, Markdown paragraph boundaries are used.

This attempts to preserve the document's natural structure.

It produced:

```text
28 chunks
```

across the corpus.

An important discovery was that the source documents contain headings as independently retrievable units. This creates some very small chunks containing only headings.

---

# 5. Results

## 5.1 Chunk counts

The experiments produced the following approximate corpus sizes:

| Strategy | Chunk configuration | Chunks |
|---|---|---:|
| Whole document | — | 4 |
| Fixed | 100 / 0 | 4 |
| Fixed + overlap | 100 / 25 | 6 |
| Fixed | 60 / 0 | 8 |
| Fixed + overlap | 60 / 15 | 9 |
| Fixed | 30 / 0 | 13 |
| Fixed + overlap | 30 / 5 | 14 |
| Fixed + overlap | 30 / 8 | 17 |
| Paragraph | — | 28 |

The exact counts depend on document length and boundary handling, but the architectural trend is clear:

> **Smaller chunks and more overlap increase the number of vectors that must be stored and searched.**

---

# 6. What happened with whole-document retrieval?

Whole-document retrieval consistently identified the correct policy document.

For example:

### Mortgage LTV

Similarity:

```text
0.8709
```

The retrieved chunk contains the entire mortgage policy.

### High-value payment

Similarity:

```text
0.8228
```

Again, the complete payment policy is returned.

### Retention

Similarity:

```text
0.9011
```

The complete retention policy is returned.

This gives a useful baseline.

However, the retrieved context is much larger than the information actually required to answer the question.

For example, a question about LTV retrieves:

```text
LTV
+
exceptions
+
affordability
```

even though the answer may only require one sentence.

### Learning

> **Whole-document retrieval provides context, but sacrifices retrieval granularity.**

---

# 7. What happened with 30-word chunks?

This was one of the most interesting configurations.

For:

> What is the maximum mortgage LTV for a first-time buyer?

the 30-word fixed configuration retrieved:

```text
The standard maximum loan-to-value ratio for a
first-time buyer is 90%. For other residential
borrowers, the standard maximum loan-to-value ratio is 80%.
```

with similarity:

```text
0.8473
```

The similarity is **lower** than the whole-document result of `0.8709`.

But the retrieved chunk is much more focused.

This demonstrates an important RAG lesson:

> **The highest similarity score does not automatically mean the best retrieval result.**

Retrieval quality must ultimately be evaluated by whether the returned context contains the information required to answer the query.

---

# 8. The strongest 30-word example: LTV exceptions

For:

> What happens when a mortgage application exceeds the standard LTV limit?

the 30-word fixed configuration retrieved:

```text
Applications above the standard LTV limit require
enhanced underwriting review. The reviewer must document
the reason for the exception and obtain approval from
a senior credit officer.
```

Similarity:

```text
0.7995
```

This is a considerably better retrieval unit than the entire mortgage document because the retrieved chunk is almost exactly the knowledge required by the question.

This is a strong example of why chunking matters.

---

# 9. The 30-word + overlap results

Adding overlap changed the behaviour.

With 30-word chunks and 8-word overlap, the LTV exception query retrieved:

```text
borrowers, the standard maximum loan-to-value ratio is 80%.
## Exceptions Applications above the standard LTV limit require
enhanced underwriting review. The reviewer must document the
reason for the exception and
```

with similarity:

```text
0.8033
```

The overlap preserves context from the preceding chunk.

Compare this with the non-overlapping version:

```text
## Exceptions Applications above the standard LTV limit require
enhanced underwriting review...
```

The difference is subtle but architecturally important.

### Learning

Overlap can help when the meaning of a chunk depends on information immediately before or after a boundary.

---

# 10. 30-word / 5-word overlap versus 30-word / 8-word overlap

The two runs also show that increasing overlap does not automatically improve retrieval.

For the high-value payment query:

| Configuration | Top similarity |
|---|---:|
| 30 words / 0 | 0.8675 |
| 30 words / 5 | 0.8494 |
| 30 words / 8 | 0.8494 |

The answer remained retrievable, but more overlap did not produce a higher score.

The 8-word configuration also produced more chunks:

```text
30 / 5  → 14 chunks
30 / 8  → 17 chunks
```

Therefore, more overlap creates additional storage and retrieval redundancy without necessarily improving semantic retrieval.

### Learning

> **Overlap is a trade-off, not a free accuracy improvement.**

---

# 11. What happened with 100-word chunks?

The 100-word configuration is particularly useful because it shows the other end of the spectrum.

With:

```text
100 words / 0
```

the corpus produced only:

```text
4 chunks
```

which is effectively equivalent to whole-document retrieval for these small documents.

The top similarity values were therefore identical to the whole-document baseline.

For example:

```text
Mortgage LTV       0.8709
LTV exception      0.8492
High-value payment 0.8228
Retention          0.9011
```

The reason is straightforward: the documents themselves are shorter than the 100-word chunk size.

---

# 12. 100-word chunks + 25-word overlap

Adding 25-word overlap produced:

```text
6 chunks
```

But because the source documents are so small, this did not materially change the retrieval results.

The top results remained equivalent to the whole-document baseline.

This is an important experimental observation:

> **Overlap cannot help if the documents are already smaller than the chunking boundary.**

In other words, chunking parameters must be evaluated against realistic document lengths.

A 100-word chunk size might behave very differently on a 5,000-word policy document.

---

# 13. Paragraph chunking

Paragraph chunking produced:

```text
28 chunks
```

This strategy produced some excellent results.

For the LTV question:

```text
The standard maximum loan-to-value ratio for a first-time buyer
is 90%. For other residential borrowers, the standard maximum
loan-to-value ratio is 80%.
```

Similarity:

```text
0.8488
```

This is a highly focused answer-bearing chunk.

Similarly, the retention question retrieved:

```text
Customer account records must normally be retained for seven
years after the relationship ends. Transaction records must also
be retained for seven years unless a longer legal or regulatory
requirement applies.
```

Similarity:

```text
0.9189
```

This was the highest observed similarity for that query.

### Learning

> **Natural document structure can produce excellent retrieval units when the source document is well structured.**

---

# 14. But paragraph chunking also exposed a problem

For the high-value payment query, paragraph retrieval returned:

```text
Rank 1: ## High-value payments
Similarity: 0.8139

Rank 2: # Payments Approval Policy
Similarity: 0.7976

Rank 3: Payments of £10,000 or more require...
Similarity: 0.7909
```

The actual answer is **rank 3**, not rank 1.

This is a very important retrieval failure.

The heading is semantically related to the query but contains no answer.

Therefore:

```text
High similarity
        ≠
Useful context
```

This is one of the most valuable lessons from the experiment.

---

# 15. Similarity score is not enough

The experiments repeatedly demonstrate that cosine similarity should not be treated as the sole retrieval-quality metric.

For example:

```text
Whole document
0.8709
```

may be less useful than:

```text
Focused chunk
0.8473
```

because the latter contains precisely the required information.

Similarly:

```text
"## High-value payments"
0.8139
```

can be less useful than:

```text
"Payments of £10,000 or more require..."
0.7909
```

because the second chunk actually answers the question.

### Architectural implication

A production RAG evaluation should eventually consider metrics such as:

- answer/context relevance
- context precision
- context recall
- retrieval hit rate / Recall@k
- ranking quality
- duplicate retrievals
- downstream answer quality

Cosine similarity is useful for ranking candidates, but it is not the same thing as retrieval quality.

---

# 16. Overlap introduces redundancy

The experiment also shows another effect.

With overlap, multiple retrieved chunks can come from the same document and contain repeated information.

For example, with 30-word / 8-word overlap, the LTV query returned:

```text
Chunk 0
Chunk 1
```

from the same mortgage document.

The chunks share part of their content.

This can be useful because the answer may span the boundary.

But it can also waste top-k slots.

Instead of:

```text
Chunk A — mortgage
Chunk B — payments
Chunk C — retention
```

you might get:

```text
Chunk A — mortgage
Chunk B — overlapping mortgage
Chunk C — overlapping mortgage
```

This introduces a new retrieval concern:

> **Do we want the top-k most similar chunks, or the top-k most useful and sufficiently diverse chunks?**

That question leads naturally into later experiments involving reranking and diversity.

---

# 17. Chunk size is therefore a three-way trade-off

The experiments suggest thinking about chunk size like this:

```text
              Larger chunks
                  ▲
                  │
        More context / coherence
                  │
                  │
                  │
Less precise ◄────┼────► More precise
                  │
                  │
                  │
        Less context / fragmentation
                  │
                  ▼
              Smaller chunks
```

### Large chunks

Pros:

- preserve context
- less likely to separate related information
- fewer vectors
- lower indexing/storage overhead

Cons:

- retrieve irrelevant information
- larger LLM context
- potentially lower retrieval precision

### Small chunks

Pros:

- highly focused retrieval
- lower irrelevant context
- potentially better retrieval precision

Cons:

- can lose context
- concepts may cross boundaries
- more vectors
- more retrieval candidates
- higher indexing/storage overhead

---

# 18. Overlap adds another trade-off

```text
No overlap
    │
    ├── fewer chunks
    ├── less storage
    └── greater boundary risk

More overlap
    │
    ├── better boundary continuity
    ├── more chunks
    ├── more storage
    └── more redundancy
```

The experiment demonstrates that increasing overlap from 5 to 8 words increased the corpus from:

```text
14 → 17 chunks
```

without producing a clear improvement in the retrieval results.

---

# 19. Important limitation of this experiment

This experiment uses a **very small corpus**:

```text
4 documents
4 queries
```

and relatively short documents.

Therefore, the results should not be interpreted as:

> "30-word chunks are optimal."

They aren't.

Instead, the experiment demonstrates the **mechanics and trade-offs** of chunking.

A production system would need to evaluate chunking against:

- much larger documents
- more diverse document structures
- more queries
- queries whose answers cross chunk boundaries
- tables
- lists
- headings
- code
- PDFs
- legal documents
- policies with nested sections

The next experiments can progressively introduce these complications.

---

# 20. Key findings

### Finding 1 — Chunking determines retrieval granularity

Whole documents are coarse retrieval units.

Smaller chunks allow individual facts or rules to be retrieved independently.

---

### Finding 2 — Smaller chunks can improve retrieval focus

The 30-word configuration often returned the exact rule required by the question.

This was particularly clear for:

- LTV exceptions
- high-value payment approval

---

### Finding 3 — Higher similarity does not necessarily mean better retrieval

The experiment produced cases where:

```text
higher similarity → less useful context
```

and:

```text
lower similarity → better answer-bearing context
```

This is one of the most important RAG lessons from Experiment 02.

---

### Finding 4 — Overlap protects against chunk-boundary problems

30-word chunks with overlap preserved surrounding context when information crossed a boundary.

---

### Finding 5 — More overlap is not automatically better

Moving from 5 to 8 words of overlap increased the number of chunks but did not show a clear retrieval-quality improvement.

---

### Finding 6 — Natural document structure can be valuable

Paragraph chunking worked very well when paragraphs represented coherent pieces of knowledge.

---

### Finding 7 — Structural chunks can also be too small

Headings became independent retrieval units and sometimes outranked the actual answer-bearing paragraph.

---

### Finding 8 — Chunking affects infrastructure cost

More/smaller chunks mean:

```text
more chunks
    ↓
more embeddings
    ↓
larger vector index
    ↓
more candidates to search
    ↓
potentially more redundancy
```

Therefore chunking is also a **cost and performance decision**, not just a retrieval-quality decision.

---

# 21. Architecture-level conclusion

The main conclusion from Experiment 02 is:

> **There is no universally optimal chunk size or overlap. Chunking is a retrieval-design decision that must balance semantic coherence, retrieval precision, context preservation, redundancy, storage and downstream LLM context cost.**

For an AI/Application Architect, the important question is therefore not:

> "What chunk size should I use?"

but:

> **"What should constitute an independently retrievable unit of knowledge for this particular document corpus and query workload?"**

That is the more useful production architecture question.

---

# 22. What this experiment sets up next

Experiment 02 naturally leads to the next RAG questions:

```text
Chunking
   │
   ▼
Top-k retrieval
   │
   ├── How large should k be?
   │
   ├── What happens when relevant chunks are ranked 2nd/3rd?
   │
   ├── What happens when chunks from the same document dominate?
   │
   ▼
Reranking
   │
   ├── Can we improve ordering?
   │
   ▼
Hybrid search
   │
   ├── Dense semantic retrieval
   ├── Sparse / keyword retrieval
   └── Combination
```

In particular, the heading problem and the overlapping-chunk redundancy problem are excellent motivation for studying **retrieval evaluation and reranking** next.

---

## 23. Final takeaway

The experiment started with a deceptively simple question:

> **How should we split documents before embedding them?**

The experiments show that this decision changes the entire retrieval behaviour of the RAG system.

```text
Document
   ↓
Chunking decision
   ↓
Retrieval units
   ↓
Embeddings
   ↓
Similarity ranking
   ↓
Top-k context
   ↓
LLM
   ↓
Answer
```

So although chunking happens early in the pipeline, its consequences propagate all the way to the final answer.

**Experiment 02 takeaway:**

> **Good RAG retrieval starts with good retrieval units. Chunking should preserve enough context to make a chunk understandable while keeping it focused enough that similarity search can surface the specific knowledge required by the query.**
