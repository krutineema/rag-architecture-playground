from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


class Tee:
    """Redirects stdout to both console and a log file."""
    def __init__(self, filepath: Path):
        self.file = filepath.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "documents.json"
RESULTS_DIR = ROOT / "results"
DB_DIR = ROOT / "chroma_db"

RESULTS_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# Held constant from Experiment 02 so this experiment isolates retrieval controls.
CHUNK_SIZE = 30
OVERLAP = 5


def load_documents():
    with DATA_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(documents):
    chunks = []
    for doc in documents:
        for index, text in enumerate(chunk_text(doc["text"])):
            chunks.append({
                "id": f"{doc['id']}-chunk-{index}",
                "text": text,
                "metadata": {
                    **doc["metadata"],
                    "source": doc["id"],
                    "title": doc["title"],
                    "chunk_index": index,
                },
            })
    return chunks


def create_collection(chunks, model):
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.create_collection(
        name="experiment03",
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )
    return collection


def retrieve(collection, model, query, top_k, where=None):
    query_embedding = model.encode(
        [query], normalize_embeddings=True
    ).tolist()[0]

    start = time.perf_counter()

    params = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
    }
    if where is not None:
        params["where"] = where

    result = collection.query(**params)
    latency_ms = (time.perf_counter() - start) * 1000

    rows = []
    for i, document in enumerate(result["documents"][0]):
        metadata = result["metadatas"][0][i]
        distance = result["distances"][0][i]
        rows.append({
            "rank": i + 1,
            "id": result["ids"][0][i],
            "title": metadata["title"],
            "document_id": metadata["source"],
            "similarity": 1 - distance,
            "text": document,
            "metadata": metadata,
        })
    return rows, latency_ms


def print_results(label, query, filter_description, rows, latency_ms):
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Filter: {filter_description}")
    print(f"Latency: {latency_ms:.2f} ms")

    for row in rows:
        m = row["metadata"]
        print(
            f"\nRank {row['rank']} | similarity={row['similarity']:.4f} | "
            f"{row['title']} | chunk={m['chunk_index']}"
        )
        print(
            f"metadata: country={m['country']}, status={m['status']}, "
            f"policy_type={m['policy_type']}, product={m['product']}, "
            f"version={m['version']}"
        )
        print(f"text: {row['text']}")


def save_results(path, experiment, query, filter_description,
                 top_k, rows, latency_ms):
    exists = path.exists()
    fields = [
        "experiment", "query", "filter", "top_k", "rank", "similarity",
        "id", "title", "document_id", "country", "status", "policy_type",
        "product", "version", "chunk_index", "latency_ms", "text"
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            m = row["metadata"]
            writer.writerow({
                "experiment": experiment,
                "query": query,
                "filter": filter_description,
                "top_k": top_k,
                "rank": row["rank"],
                "similarity": f"{row['similarity']:.6f}",
                "id": row["id"],
                "title": row["title"],
                "document_id": row["document_id"],
                "country": m["country"],
                "status": m["status"],
                "policy_type": m["policy_type"],
                "product": m["product"],
                "version": m["version"],
                "chunk_index": m["chunk_index"],
                "latency_ms": f"{latency_ms:.3f}",
                "text": row["text"],
            })


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"results_{timestamp}.txt"
    original_stdout = sys.stdout
    tee = Tee(log_file)
    sys.stdout = tee

    try:
        documents = load_documents()
        chunks = build_chunks(documents)
        model = SentenceTransformer(EMBEDDING_MODEL)
        collection = create_collection(chunks, model)

        print(f"Documents: {len(documents)}")
        print(f"Chunks: {len(chunks)}")
        print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

        results = RESULTS_DIR / f"retrieval_results_{timestamp}.csv"
        if results.exists():
            results.unlink()

        # PART 1: Top-k
        query = "What happens when a customer forgets their password?"
        for k in [1, 2, 3, 5]:
            rows, latency = retrieve(collection, model, query, k)
            print_results(f"TOP-K = {k}", query, "none", rows, latency)
            save_results(results, f"top_k_{k}", query, "none", k, rows, latency)

        # PART 2: Metadata filters
        query = "How long must customer records be retained?"
        filters = [
            ("none", None),
            ("status=current", {"status": "current"}),
            ("status=archived", {"status": "archived"}),
            ("policy_type=retention", {"policy_type": "retention"}),
            ("status=current AND policy_type=retention", {
                "$and": [{"status": "current"}, {"policy_type": "retention"}]
            }),
        ]

        for description, where in filters:
            rows, latency = retrieve(collection, model, query, 3, where)
            print_results(
                f"METADATA FILTER = {description}",
                query, description, rows, latency
            )
            save_results(
                results, "metadata_filtering", query, description, 3, rows, latency
            )

        # PART 3: deliberately incorrect filter
        where = {
            "$and": [{"status": "current"}, {"policy_type": "mortgage"}]
        }
        rows, latency = retrieve(collection, model, query, 3, where)
        print_results(
            "INCORRECT FILTER",
            query, "status=current AND policy_type=mortgage", rows, latency
        )
        save_results(
            results, "incorrect_filter", query,
            "status=current AND policy_type=mortgage", 3, rows, latency
        )

        print(f"\nResults written to {results}")
        print(f"Console log written to {log_file}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
