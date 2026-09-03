# Experiment 03 — Top-K Retrieval and Metadata Filtering

This experiment explores two retrieval controls that become important once a basic vector-search RAG pipeline is working:

1. **Top-k** — how many of the most similar chunks should be returned for a query?
2. **Metadata filtering** — how should structured document attributes be used to constrain retrieval before similarity ranking?

The goal was not to improve the embedding model or chunking strategy. Those were deliberately held constant so that the effect of retrieval-time controls could be observed independently.

---

## Why this experiment?

A vector database does not understand which documents are *operationally valid* in the same way a human does.

It primarily answers:

> "Which chunks are closest to this query in embedding space?"

That is useful, but similarity alone is not always enough.

In a real enterprise RAG system:

- multiple policy versions may exist;
- archived documents may still be semantically very similar to current ones;
- the same terms may appear across unrelated business domains;
- retrieving too few chunks may miss useful evidence;
- retrieving too many chunks may introduce irrelevant context;
- a wrong metadata filter can remove the correct answer entirely.

This experiment was designed to make those behaviours visible.

---

## Experiment setup

The retrieval pipeline uses:

- **Vector database:** ChromaDB
- **Similarity metric:** cosine similarity
- **Embedding model:** `BAAI/bge-small-en-v1.5`
- **Embedding dimension:** `384`
- **Documents:** `7`
- **Generated chunks:** `56`
- **Chunk size:** `30` words
- **Chunk overlap:** `5` words

The chunking configuration was intentionally kept the same as the previous experiment so that this experiment isolates **retrieval controls**, rather than changing multiple variables at once.

Each chunk is stored with both its embedding and structured metadata.

Example metadata:

```text
country=UK
status=current
policy_type=retention
product=all
version=5.0
chunk_index=2
```

This allows two different retrieval mechanisms to work together:

```text
User query
   |
   v
Metadata constraints
   |
   v
Eligible chunks
   |
   v
Vector similarity search
   |
   v
Top-k chunks
   |
   v
Context for the LLM
```

---

# Part 1 — Understanding Top-K

## What is top-k?

`top_k` controls the number of nearest chunks returned by vector search.

For example:

```python
collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
```

means:

> Return the 3 chunks with the highest similarity to the query.

The experiment tested:

```text
k = 1
k = 2
k = 3
k = 5
```

using the query:

> **What happens when a customer forgets their password?**

No metadata filter was applied.

---

## Results

| Top-k | Highest similarity | Retrieved content |
|---:|---:|---|
| 1 | 0.6108 | Data-retention policy |
| 2 | 0.6108 | Retention + fraud policy |
| 3 | 0.6108 | Retention + fraud + mortgage policy |
| 5 | 0.6108 | Adds access-control chunks, but still no direct password-reset answer |

The top result remained:

> `UK Customer and Transaction Data Retention Policy`

with a similarity of approximately:

```text
0.6108
```

As `k` increased, additional chunks were returned from:

- fraud monitoring,
- mortgage lending,
- enterprise access control.

Some of these were superficially related to concepts such as customers, credentials or access, but none directly answered the password-reset question.

---

## What this demonstrates

Increasing `top_k` improves **retrieval breadth**, not retrieval intelligence.

If the correct chunk is ranked fourth, increasing:

```text
k = 3  ->  k = 5
```

may recover it.

But if the relevant information does not exist in the indexed corpus, or the embedding model does not rank it sufficiently highly, increasing `k` simply retrieves more weakly related material.

In this experiment:

```text
k = 1
```

returned one irrelevant result.

Increasing to:

```text
k = 5
```

returned five results — but did not make the answer correct.

### Key lesson

> **Top-k is a recall control, not a relevance guarantee.**

A larger `k` increases the chance that useful evidence is included, but it also increases the chance of introducing noise.

---

## Why not always choose a large top-k?

Every retrieved chunk may eventually become part of the LLM context.

A very large `top_k` can therefore:

- increase token usage;
- increase latency further down the pipeline;
- add irrelevant or contradictory evidence;
- make generation less focused;
- consume context-window capacity;
- increase reranking workload if a reranker is used.

The architectural goal is therefore not:

> "retrieve as many chunks as possible"

but:

> **retrieve enough chunks to achieve good recall, while keeping the evidence set relevant and manageable.**

In production systems, `top_k` is usually tuned using retrieval evaluation rather than chosen arbitrarily.

---

# Part 2 — Metadata Filtering

The second part tested whether structured metadata could make retrieval safer and more precise.

Query:

> **How long must customer records be retained?**

The retrieval always used:

```text
top_k = 3
```

but different metadata filters were applied.

---

## Case 1 — No metadata filter

```text
Filter: none
```

Results:

| Rank | Similarity | Document | Status |
|---:|---:|---|---|
| 1 | 0.9185 | UK Customer and Transaction Data Retention Policy | current |
| 2 | 0.8448 | UK Customer Data Retention Policy — Archived | archived |
| 3 | 0.8323 | UK Customer and Transaction Data Retention Policy | current |

The correct current policy ranked first.

However, an **archived policy ranked second** because its wording was semantically very similar.

This is a critical enterprise-RAG problem.

The vector database correctly identified semantic similarity, but it had no reason to know that an archived document should not normally be used for current operational guidance.

---

## Case 2 — Filter to current documents

```python
where={"status": "current"}
```

Results:

| Rank | Similarity | Status |
|---:|---:|---|
| 1 | 0.9185 | current |
| 2 | 0.8323 | current |
| 3 | 0.8003 | current |

The archived document disappeared completely.

This illustrates an important distinction:

### Without filtering

```text
Search everything
      |
      v
Rank by semantic similarity
```

### With filtering

```text
Keep only status=current
      |
      v
Search eligible chunks
      |
      v
Rank by semantic similarity
```

Metadata filtering does not simply "boost" current documents.

It changes the **candidate search space**.

---

## Case 3 — Filter to archived documents

```python
where={"status": "archived"}
```

Results:

| Rank | Similarity | Document |
|---:|---:|---|
| 1 | 0.8448 | Archived retention policy |
| 2 | 0.7858 | Archived retention policy |
| 3 | 0.6829 | Archived retention policy |

Now the vector search operates only over archived material.

This is useful when the user's intent genuinely requires historical information.

For example:

> "What did the previous policy say?"

The same metadata that protects a current-policy assistant can therefore also enable deliberate historical retrieval.

---

## Case 4 — Filter by policy type

```python
where={"policy_type": "retention"}
```

Results:

| Rank | Similarity | Status |
|---:|---:|---|
| 1 | 0.9185 | current |
| 2 | 0.8448 | archived |
| 3 | 0.8323 | current |

This removed unrelated policy domains, but it did **not** solve the version-validity problem because both current and archived policies had:

```text
policy_type=retention
```

### Lesson

A metadata field solves only the constraint represented by that field.

Filtering by:

```text
policy_type=retention
```

does not imply:

```text
status=current
```

Both constraints must be expressed if both matter.

---

## Case 5 — Combined metadata filtering

The experiment then applied:

```python
where={
    "$and": [
        {"status": "current"},
        {"policy_type": "retention"}
    ]
}
```

Results:

| Rank | Similarity | Document |
|---:|---:|---|
| 1 | 0.9185 | Current retention policy |
| 2 | 0.8323 | Current retention policy |
| 3 | 0.8003 | Current retention policy |

This produced the cleanest candidate set for the operational question.

The retrieval system was now effectively being told:

> Search for semantic similarity, **but only inside current retention policies**.

This is closer to how enterprise retrieval should behave.

---

# Part 3 — What happens when the metadata filter is wrong?

The final test deliberately applied the wrong constraint:

```python
where={
    "$and": [
        {"status": "current"},
        {"policy_type": "mortgage"}
    ]
}
```

while still asking:

> **How long must customer records be retained?**

Results:

| Rank | Similarity | Document |
|---:|---:|---|
| 1 | 0.7090 | UK Residential Mortgage Lending Policy |
| 2 | 0.6522 | UK Residential Mortgage Lending Policy |
| 3 | 0.6191 | UK Residential Mortgage Lending Policy |

The actual retention-policy chunks were not returned at all.

Why?

Because the metadata filter excluded them before semantic ranking happened.

The vector database therefore did the best it could **inside the wrong search space**.

This produced mortgage-policy chunks containing terms such as:

```text
retained
record keeping
records
```

which were semantically related to the query, but did not answer the intended policy question.

---

# Metadata filtering is powerful — and dangerous

The experiment highlights an important architectural trade-off.

Metadata filtering can dramatically improve retrieval precision:

```text
large corpus
     |
     | metadata filter
     v
small relevant corpus
     |
     | vector similarity
     v
top-k results
```

But a wrong filter can cause **false exclusion**:

```text
correct answer exists in vector DB
             |
             | wrong filter
             X
       never considered
```

This means metadata filters should not be created casually.

The system needs confidence that the metadata constraint is appropriate.

---

# Top-K vs Metadata Filtering

These controls solve different problems.

| Control | Main purpose | What it changes |
|---|---|---|
| `top_k` | Recall / breadth | Number of ranked chunks returned |
| Metadata filter | Scope / eligibility | Which chunks are allowed to compete |
| Similarity score | Semantic ranking | Ordering of eligible chunks |

A useful mental model is:

```text
Metadata filtering asks:
"Where am I allowed to search?"

Vector similarity asks:
"Within that space, what looks most relevant?"

Top-k asks:
"How many of those ranked results should I keep?"
```

These are separate architectural decisions.

---

# Similarity scores are relative, not proof of correctness

Another important lesson from the experiment is that a similarity score should not be interpreted as a confidence score.

For the password question, the highest result had:

```text
similarity = 0.6108
```

but was still irrelevant.

For the retention query, the correct result had:

```text
similarity = 0.9185
```

which was much stronger.

However, there is no universal rule such as:

```text
similarity > 0.8 = correct
```

because scores depend on:

- embedding model;
- corpus;
- query wording;
- chunking;
- similarity metric;
- document distribution.

Similarity tells us which vectors are closer to one another.

It does **not** prove that a chunk contains the correct answer.

---

# Retrieval architecture learned from this experiment

A stronger RAG retrieval pipeline is therefore not simply:

```text
query
  |
embedding
  |
vector search
  |
top-k
  |
LLM
```

A more realistic architecture is:

```text
                    User query
                         |
                         v
                Query understanding
                         |
              +----------+----------+
              |                     |
              v                     v
      Metadata constraints      Query embedding
              |                     |
              +----------+----------+
                         |
                         v
              Filtered vector search
                         |
                         v
                   Candidate set
                         |
                         v
                Top-k / reranking
                         |
                         v
                Relevant context
                         |
                         v
                       LLM
```

The exact implementation may vary, but the architectural principle remains:

> **Semantic similarity should operate inside the correct business and governance constraints.**

---

# Practical implications for enterprise RAG

Metadata commonly used in real systems may include:

```text
document_status
effective_date
expiry_date
version
country
jurisdiction
business_unit
product
document_type
policy_type
security_classification
customer_segment
language
tenant_id
access_control_group
```

These fields are not merely descriptive.

They can become part of the retrieval logic.

For example:

```text
country = UK
AND
status = current
AND
policy_type = retention
```

can prevent semantically similar but operationally invalid documents from reaching the generation stage.

This is especially important in regulated environments where retrieving an outdated policy can be more dangerous than retrieving no answer.

---

# Important design insight: filters must come from trustworthy signals

This experiment also exposes the next architectural question:

> Who decides which metadata filters should be applied?

Possible approaches include:

- filters supplied explicitly by the application;
- filters derived from authenticated user context;
- deterministic routing rules;
- query classification;
- LLM-generated structured filters;
- hybrid approaches combining fixed constraints and inferred filters.

Some constraints should normally be **system-controlled** rather than inferred from natural language.

Examples:

```text
tenant_id
user permissions
document_status=current
security classification
jurisdiction
```

Other filters may reasonably come from the user's query.

For example:

```text
"What was the archived retention policy?"
```

could intentionally select:

```text
status=archived
```

The experiment therefore shows that metadata filtering is not merely a vector-database feature; it is part of the **RAG system's query-planning and governance architecture**.

---

# What I learned

## 1. Top-k is a trade-off between recall and noise

A low `k` risks missing supporting evidence.

A high `k` increases retrieval breadth but may add irrelevant context.

Increasing `k` does not fix a fundamentally poor retrieval match.

---

## 2. Semantic relevance and operational validity are different

An archived policy can be extremely similar to the user's query while still being the wrong source for a current operational answer.

Vector similarity alone cannot reliably enforce concepts such as:

```text
current vs archived
allowed vs restricted
UK vs US
customer A vs customer B
```

Structured metadata is needed for these constraints.

---

## 3. Metadata filtering reduces the candidate search space

Filtering happens before ranking.

The vector search ranks only the chunks that survive the filter.

This can greatly improve precision.

---

## 4. Metadata filters can also hide the correct answer

An incorrect filter does not merely lower the correct document's rank.

It can remove the document from consideration entirely.

This makes filter-generation logic a critical part of the RAG architecture.

---

## 5. Metadata is part of retrieval design, not just storage

Metadata fields should be designed with future retrieval requirements in mind.

A useful question during ingestion is not only:

> "What information can I store about this document?"

but also:

> **"What dimensions might I later need to filter or govern retrieval by?"**

---

## 6. Retrieval quality is produced by several controls working together

A production RAG system may combine:

```text
metadata filtering
        +
vector similarity
        +
top-k selection
        +
similarity thresholds
        +
hybrid / keyword search
        +
reranking
        +
retrieval evaluation
```

No single mechanism guarantees the right evidence.

---

# Result summary

The experiment produced three clear behaviours.

### Top-k experiment

```text
Increasing k:
1 -> 2 -> 3 -> 5

increased the number of retrieved chunks,
but did not produce the missing password-reset answer.
```

### Correct metadata filtering

```text
status=current
AND
policy_type=retention
```

removed archived and unrelated documents and returned only current retention-policy chunks.

### Incorrect metadata filtering

```text
status=current
AND
policy_type=mortgage
```

removed the correct retention-policy chunks entirely and forced the vector search to rank the best matching mortgage chunks instead.

---

# Key takeaway

> **Vector search answers "what is semantically similar?" — not "what is valid for this user and this question?"**

Top-k determines how much evidence is retained.

Metadata filtering determines where retrieval is allowed to search.

Used together, they provide much more control over the evidence supplied to an LLM.

But metadata filtering also introduces a new architectural responsibility:

> **the system must apply the right constraints, because a wrong filter can exclude the correct answer before semantic search even begins.**

That distinction is central to designing reliable RAG systems.

---

## Files

```text
experiment.py
data/
  documents.json
results/
  retrieval_results_<timestamp>.csv
  results_<timestamp>.txt
chroma_db/
```

The CSV contains the structured retrieval results, including:

- experiment name;
- query;
- filter;
- top-k;
- rank;
- similarity;
- document metadata;
- latency;
- retrieved chunk text.

The text log captures the same experiment in a human-readable console format.

---

## Running the experiment

Install the required packages:

```bash
pip install chromadb sentence-transformers
```

Run:

```bash
python experiment.py
```

The script rebuilds the Chroma collection, embeds all chunks, runs the top-k and metadata-filter experiments, and writes timestamped result files under:

```text
results/
```

---

## Next questions

This experiment naturally leads to several further RAG design questions:

- How should an appropriate `top_k` be selected using retrieval metrics?
- When should a similarity threshold reject weak results entirely?
- Should metadata filters be applied before vector search, after retrieval, or both?
- How can filters be inferred safely from natural-language queries?
- When should hybrid search outperform pure semantic search?
- Can reranking remove the noise introduced by a larger candidate `k`?
- How should retrieval be evaluated using metrics such as Recall@K, Precision@K and MRR?
