from datetime import datetime
from pathlib import Path
import csv
import time
from typing import List, Dict

# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
import ollama


# ============================================================
# Experiment 01 — Basic RAG
# ============================================================
#
# Architecture:
#
# Documents
#    ↓
# Chunk
#    ↓
# Embedding model
#    ↓
# Chroma vector store
#
# Query
#    ↓
# Query embedding
#    ↓
# Similarity search
#    ↓
# Top-k chunks
#    ↓
# Context construction
#    ↓
# Ollama LLM
#    ↓
# Grounded answer + source names
#
# Deliberately NO LangChain/LlamaIndex:
# the purpose is to make each RAG stage visible.
# ============================================================


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHROMA_DIR = ROOT / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
#OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_MODEL = "gemma3:1b"
COLLECTION_NAME = "basic_rag"
TOP_K = 3

# Keep the baseline intentionally simple.
# Later experiments will vary chunk size, overlap, top-k, metadata,
# hybrid retrieval, reranking and contextual retrieval.


def load_documents() -> List[Dict]:
    """Load markdown documents from data/."""
    documents = []

    for path in sorted(DATA_DIR.glob("*.md")):
        documents.append(
            {
                "source": path.name,
                "text": path.read_text(encoding="utf-8").strip(),
            }
        )

    if not documents:
        raise RuntimeError(f"No markdown documents found in {DATA_DIR}")

    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Very simple word-based chunker for the baseline.

    This is deliberately NOT a production chunker.
    The goal is to expose chunking as a variable that later experiments
    can change.

    chunk_size / overlap are measured approximately in words here,
    not model tokens.
    """
    words = text.split()

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(documents: List[Dict]) -> List[Dict]:
    """Chunk every document and preserve source metadata."""
    chunks = []

    for document in documents:
        pieces = chunk_text(document["text"])

        for index, piece in enumerate(pieces):
            chunks.append(
                {
                    "id": f'{document["source"]}::chunk-{index}',
                    "source": document["source"],
                    "chunk_index": index,
                    "text": piece,
                }
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunks_path = RESULTS_DIR / f"chunks_{timestamp}.txt"

    with chunks_path.open("w", encoding="utf-8") as f:
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write("=" * 70 + "\n\n")
        for chunk in chunks:
            f.write(f"ID: {chunk['id']}\n")
            f.write(f"Source: {chunk['source']} | Chunk Index: {chunk['chunk_index']}\n")
            f.write("Text:\n")
            f.write(chunk["text"])
            f.write("\n" + "-" * 70 + "\n\n")

    return chunks


def create_vector_store(chunks: List[Dict], embedder: SentenceTransformer):
    """
    Create a fresh local Chroma collection.

    Chroma is being used as the local vector store so that the experiment
    visibly contains a separate embedding model and vector-store layer.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )

    return collection


def retrieve(
    collection,
    embedder: SentenceTransformer,
    query: str,
    top_k: int,
):
    """Embed the query and retrieve top-k chunks."""
    start_query_embedding = time.perf_counter()
    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    elapsed_query_embedding = time.perf_counter() - start_query_embedding
    print(f"Embedding query: {elapsed_query_embedding:.3f}s")

    start_retrieval = time.perf_counter()
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    elapsed_retrieval = time.perf_counter() - start_retrieval
    print(f"Retrieval from vector store: {elapsed_retrieval:.3f}s")

    retrieved = []
    start_similarity = time.perf_counter()

    for i in range(len(result["documents"][0])):
        # Chroma cosine distance = 1 - cosine similarity.
        distance = result["distances"][0][i]
        similarity = 1.0 - distance

        retrieved.append(
            {
                "rank": i + 1,
                "text": result["documents"][0][i],
                "source": result["metadatas"][0][i]["source"],
                "chunk_index": result["metadatas"][0][i]["chunk_index"],
                "similarity": similarity,
            }
        )
    
    elapsed_similarity = time.perf_counter() - start_similarity
    print(f"Calculating similarity: {elapsed_similarity:.3f}s")

    timings = {
        "elapsed_query_embedding": elapsed_query_embedding,
        "elapsed_retrieval": elapsed_retrieval,
        "elapsed_similarity": elapsed_similarity,
    }

    return retrieved, timings


def build_prompt(query: str, retrieved: List[Dict]) -> str:
    
    """Build the augmented prompt sent to the LLM."""
    context_blocks = []

    for item in retrieved:
        context_blocks.append(
            f"[Source: {item['source']} | chunk {item['chunk_index']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the supplied context.

Rules:
- If the context does not contain enough information, say:
  "I don't have enough information in the supplied documents."
- Do not invent policy details.
- Cite the source filename(s) used in your answer.

CONTEXT
=======
{context}

QUESTION
========
{query}

ANSWER
======
""".strip()


def generate_answer(prompt: str):
    """Call a local Ollama model."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


def save_results(query: str, retrieved: List[Dict], answer: str, timings: Dict[str, float]):
    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"retrieval_results_{timestamp}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "source",
                "chunk_index",
                "similarity",
                "text",
            ],
        )
        writer.writeheader()

        for item in retrieved:
            writer.writerow(item)

    output_path = RESULTS_DIR / f"{OLLAMA_MODEL}_run_{timestamp}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        f.write("RAG BASIC BASELINE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Embedding model: {EMBEDDING_MODEL}\n")
        f.write(f"LLM: {OLLAMA_MODEL}\n")
        f.write(f"Top-k: {TOP_K}\n")
        f.write(f"Query: {query}\n\n")

        f.write("TIMING BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"Embedding documents in vector store (elapsed_embedding): {timings.get('elapsed_embedding', 0.0):.3f}s\n")
        f.write(f"Embedding query (elapsed_query_embedding): {timings.get('elapsed_query_embedding', 0.0):.3f}s\n")
        f.write(f"Retrieval from vector store (elapsed_retrieval): {timings.get('elapsed_retrieval', 0.0):.3f}s\n")
        f.write(f"Calculating similarity (elapsed_similarity): {timings.get('elapsed_similarity', 0.0):.3f}s\n")
        f.write(f"Building prompt (elapsed_building_prompt): {timings.get('elapsed_building_prompt', 0.0):.3f}s\n")
        f.write(f"Generating answer (elapsed_generation): {timings.get('elapsed_generation', 0.0):.3f}s\n")
        f.write(f"End-to-end retrieval + generation (elapsed_total): {timings.get('elapsed_total', 0.0):.3f}s\n\n")

        f.write("RETRIEVED CHUNKS\n")
        f.write("-" * 70 + "\n")

        for item in retrieved:
            f.write(
                f"\nRank {item['rank']}\n"
                f"Source: {item['source']}\n"
                f"Chunk: {item['chunk_index']}\n"
                f"Similarity: {item['similarity']:.4f}\n"
                f"Text:\n{item['text']}\n"
            )

        f.write("\n\nGENERATED ANSWER\n")
        f.write("-" * 70 + "\n")
        f.write(answer)
        f.write("\n")

    return csv_path, output_path


def main():
    print("=" * 70)
    print("EXPERIMENT 01 — BASIC RAG")
    print("=" * 70)

    start_embedding = time.perf_counter()

    documents = load_documents()

    print(f"\nDocuments loaded: {len(documents)}")

    chunks = build_chunks(documents)

    print(f"Chunks created: {len(chunks)}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(
        "Embedding dimension:",
        embedder.get_sentence_embedding_dimension(),
    )

    print("\nBuilding vector store...")
    collection = create_vector_store(chunks, embedder)

    print(f"\nVector store built: {collection.count()} chunks")

    elapsed_embedding  = time.perf_counter() - start_embedding
    print(f"Embedding documents in vector store: {elapsed_embedding:.3f}s")

    query = input(
        "\nEnter a question "
        "(press Enter for the default question): "
    ).strip()

    if not query:
        query = (
            "What happens when a customer forgets their password "
            "and fails the identity checks?"
        )

    print(f"\nQuery: {query}")
    print(f"Retrieving top-{TOP_K} chunks...")

    start = time.perf_counter()

    retrieved, retrieval_timings = retrieve(
        collection,
        embedder,
        query,
        TOP_K,
    )

    start_building_prompt = time.perf_counter()
    prompt = build_prompt(query, retrieved)
    elapsed_building_prompt = time.perf_counter() - start_building_prompt
    print(f"Building prompt: {elapsed_building_prompt:.3f}s")

    start_generation = time.perf_counter()

    print("\nGenerating answer with", OLLAMA_MODEL, " ...")
    answer = generate_answer(prompt)

    elapsed_generation = time.perf_counter() - start_generation
    print(f"Generating answer: {elapsed_generation:.3f}s")

    elapsed_total = time.perf_counter() - start
    print(f"End-to-end retrieval + generation time: {elapsed_total:.3f}s")

    timings = {
        "elapsed_embedding": elapsed_embedding,
        "elapsed_query_embedding": retrieval_timings["elapsed_query_embedding"],
        "elapsed_retrieval": retrieval_timings["elapsed_retrieval"],
        "elapsed_similarity": retrieval_timings["elapsed_similarity"],
        "elapsed_building_prompt": elapsed_building_prompt,
        "elapsed_generation": elapsed_generation,
        "elapsed_total": elapsed_total,
    }

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for item in retrieved:
        print(
            f"\n{item['rank']}. "
            f"{item['source']} "
            f"(similarity={item['similarity']:.4f})"
        )
        print(item["text"])

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    csv_path, output_path = save_results(
        query,
        retrieved,
        answer,
        timings,
    )

    print("\nSaved:")
    print(csv_path)
    print(output_path)


if __name__ == "__main__":
    main()
