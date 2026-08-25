"""
Experiment 02 — Chunking

Purpose:
    Isolate the effect of chunking strategy on vector retrieval.

Fixed:
    - embedding model
    - documents
    - queries
    - top-k
    - Chroma retrieval

Variable:
    - chunking strategy

Strategies:
    1. whole_document
    2. fixed
    3. fixed_overlap
    4. paragraph
"""

from pathlib import Path
from datetime import datetime
import json
import re

# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
QUERIES_FILE = BASE_DIR / "queries.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Keep these deliberately easy to change while experimenting.
#FIXED_CHUNK_WORDS = 60
FIXED_CHUNK_WORDS = 100
OVERLAP_WORDS = 25
TOP_K = 3


def load_documents():
    documents = []

    for path in sorted(DATA_DIR.glob("*.md")):
        documents.append(
            {
                "id": path.stem,
                "text": path.read_text(encoding="utf-8"),
                "source": path.name,
            }
        )

    return documents


def load_queries():
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))


def normalise_text(text):
    return re.sub(r"\s+", " ", text).strip()


def whole_document_chunks(document):
    return [
        {
            "chunk_id": f"{document['id']}_chunk_000",
            "document_id": document["id"],
            "source": document["source"],
            "chunk_index": 0,
            "text": normalise_text(document["text"]),
        }
    ]


def fixed_chunks(document, chunk_words=FIXED_CHUNK_WORDS, overlap_words=0):
    text = normalise_text(document["text"])
    words = text.split()

    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks = []
    start = 0
    index = 0
    step = chunk_words - overlap_words

    while start < len(words):
        chunk = words[start : start + chunk_words]

        if chunk:
            chunks.append(
                {
                    "chunk_id": f"{document['id']}_chunk_{index:03d}",
                    "document_id": document["id"],
                    "source": document["source"],
                    "chunk_index": index,
                    "text": " ".join(chunk),
                }
            )

        start += step
        index += 1

    return chunks


def paragraph_chunks(document):
    # Markdown blank lines are treated as paragraph boundaries.
    paragraphs = re.split(r"\n\s*\n", document["text"].strip())

    chunks = []
    index = 0

    for paragraph in paragraphs:
        text = normalise_text(paragraph)

        if not text:
            continue

        chunks.append(
            {
                "chunk_id": f"{document['id']}_chunk_{index:03d}",
                "document_id": document["id"],
                "source": document["source"],
                "chunk_index": index,
                "text": text,
            }
        )
        index += 1

    return chunks


def build_chunks(documents, strategy):
    all_chunks = []

    for document in documents:
        if strategy == "whole_document":
            chunks = whole_document_chunks(document)
        elif strategy == "fixed":
            chunks = fixed_chunks(document)
        elif strategy == "fixed_overlap":
            chunks = fixed_chunks(
                document,
                chunk_words=FIXED_CHUNK_WORDS,
                overlap_words=OVERLAP_WORDS,
            )
        elif strategy == "paragraph":
            chunks = paragraph_chunks(document)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        all_chunks.extend(chunks)

    return all_chunks


def create_collection(client, strategy):
    # Ephemeral collection: nothing from one strategy can leak into another.
    collection_name = f"rag_exp02_{strategy}"
    return client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(collection, chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {
                "document_id": chunk["document_id"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )


def retrieve(collection, query, model):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        # Chroma's cosine distance is:
        #     1 - cosine_similarity
        #
        # Therefore:
        similarity = 1 - distance

        retrieved.append(
            {
                "document_id": metadata["document_id"],
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "similarity": similarity,
                "text": document,
            }
        )

    return retrieved


def print_retrieval(strategy, query, retrieved):
    print("\n" + "=" * 90)
    print(f"STRATEGY: {strategy}")
    print(f"QUERY:    {query}")
    print("=" * 90)

    for rank, item in enumerate(retrieved, start=1):
        print(f"\nRank {rank}")
        print(f"Source:       {item['source']}")
        print(f"Chunk index:  {item['chunk_index']}")
        print(f"Similarity:   {item['similarity']:.4f}")
        print(f"Text:         {item['text']}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Fixed chunk size: {FIXED_CHUNK_WORDS} words")
    print(f"Overlap: {OVERLAP_WORDS} words")
    print(f"Top-k: {TOP_K}")

    documents = load_documents()
    queries = load_queries()

    print(f"\nLoaded {len(documents)} documents.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.Client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    strategies = [
        "whole_document",
        "fixed",
        "fixed_overlap",
        "paragraph",
    ]

    results = {
        "experiment": "02-chunking",
        "timestamp": timestamp,
        "embedding_model": EMBEDDING_MODEL,
        "fixed_chunk_words": FIXED_CHUNK_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "top_k": TOP_K,
        "strategies": {},
    }

    for strategy in strategies:
        chunks = build_chunks(documents, strategy)

        print("\n" + "#" * 90)
        print(f"# {strategy}")
        print(f"# Number of chunks: {len(chunks)}")
        print("#" * 90)

        collection = create_collection(client, strategy)
        index_chunks(collection, chunks, model)

        strategy_results = {
            "chunk_count": len(chunks),
            "queries": [],
        }

        for query_item in queries:
            query = query_item["query"]

            retrieved = retrieve(collection, query, model)
            print_retrieval(strategy, query, retrieved)

            strategy_results["queries"].append(
                {
                    "query": query,
                    "retrieved": retrieved,
                }
            )

        results["strategies"][strategy] = strategy_results

    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print(f"Results written to: {output_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
