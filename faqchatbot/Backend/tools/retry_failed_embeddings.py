"""
Retry queued failed embeddings and upsert them into Pinecone.

Usage:
  source .venv/bin/activate
  python tools/retry_failed_embeddings.py

The script reads `failed_embeddings_queue.jsonl` in the project root
and attempts to embed each queued chunk using `embed_texts`. Successfully
processed entries are removed from the queue; remaining failures are
rewritten back to the queue file.
"""
import json
import os
from pathlib import Path

from app.services.chunking_service import Chunk
from app.services.embedding_service import embed_texts
from app.services.pinecone_service import upsert_chunks


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "failed_embeddings_queue.jsonl"


def load_queue():
    if not QUEUE_PATH.exists():
        print("No failed embeddings queue found.")
        return []
    entries = []
    with QUEUE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries


def save_queue(entries):
    with QUEUE_PATH.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main():
    entries = load_queue()
    if not entries:
        print("Queue empty — nothing to retry.")
        return

    print(f"Retrying {len(entries)} queued chunks...")

    remaining = []
    # Process in batches to reuse embed_texts batching
    batch_size = 64
    for i in range(0, len(entries), batch_size):
        batch = entries[i : i + batch_size]
        texts = [e["text"] for e in batch]
        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            print(f"Embedding call failed for batch starting at {i}: {e}")
            remaining.extend(batch)
            continue

        chunks = []
        embs = []
        topics_per_chunk = []
        for e, emb in zip(batch, embeddings):
            if emb is None or all(v == 0.0 for v in emb):
                remaining.append(e)
                continue
            chunks.append(Chunk(chunk_id=e["chunk_id"], text=e["text"], source_url=e["source_url"], question=e.get("question", "")))
            embs.append(emb)
            topics_per_chunk.append(e.get("topics", []))

        if chunks and embs:
            try:
                upsert_chunks(chunks, embs, topics_per_chunk)
                print(f"Upserted {len(chunks)} chunks from batch {i}-{i+len(batch)-1}")
            except Exception as e:
                print(f"Upsert failed for batch starting at {i}: {e}")
                # On failure, put these back into remaining for retry later
                for c, t in zip(chunks, topics_per_chunk):
                    remaining.append({
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source_url": c.source_url,
                        "question": c.question,
                        "topics": t,
                    })

    save_queue(remaining)
    print(f"Retry complete. {len(remaining)} chunks remain queued.")


if __name__ == "__main__":
    main()
