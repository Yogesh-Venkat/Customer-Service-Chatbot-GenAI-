# prepare_kb_chroma.py
"""
Prepare KB: parse, chunk, embed, save chunks.csv, and persist to ChromaDB.
Usage:
    python prepare_kb_chroma.py
Outputs:
    - chroma_db/               (Chroma persistent DB)
    - chunks.csv               (chunk table with metadata)
"""

import os
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# ---------- CONFIG ----------
KB_FILE = "kb.txt"
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "kb_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKS_CSV = "chunks.csv"

# Chunking params
MAX_CHARS = 600
OVERLAP = 120


# ---------- TEXT PARSING & CHUNKING ----------
def load_kb(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"KB file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def normalize(text: str) -> str:
    text = text.replace("\r", "")
    # normalize multiple newlines
    text = re.sub(r"\n{2,}", "\n\n", text.strip())
    return text


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split KB into sections using "Section <num>:" headings.
    Returns list of (section_heading_line, section_text).
    If no Section markers exist, returns a single section with whole text.
    Handles the case where everything is on one line as well.
    """
    pattern = r"(Section\s+\d+\s*:[^\n]*)"
    parts = re.split(pattern, text)

    # No sections at all -> full KB as a single section
    if len(parts) == 1:
        return [("Full KB", text)]

    # Sometimes with no newlines, entire KB may be swallowed into a single "header".
    # Example: ["", "Section 1: ... (rest of file)", ""]
    # In that case, just treat full text as one section.
    if len(parts) == 3 and not parts[0].strip() and not parts[2].strip():
        return [("Full KB", text)]

    sections = []
    i = 1
    while i < len(parts):
        header = parts[i].strip()
        body = parts[i + 1].strip() if (i + 1) < len(parts) else ""
        sections.append((header, body))
        i += 2

    if not sections:
        return [("Full KB", text)]

    return sections


def extract_headings(section_text: str) -> List[Tuple[str, str]]:
    """
    Extract heading blocks inside a section.
    Heading pattern examples in your KB: AP-001: Order Confirmation...
    Returns list of (heading_line, heading_body)
    If no heading pattern found, returns a single heading 'SectionBody' with full text.
    """
    # Heading pattern: CODE: Title (CODE may be letters+dash+digits)
    pattern = r"(^[A-Z]{2,}-\d{3,}:[^\n]+)"
    parts = re.split(pattern, section_text, flags=re.M)

    if len(parts) == 1:
        return [("SectionBody", section_text.strip())]

    headings = []
    j = 1
    while j < len(parts):
        heading = parts[j].strip()
        body = parts[j + 1].strip() if (j + 1) < len(parts) else ""
        headings.append((heading, body))
        j += 2

    if not headings:
        return [("SectionBody", section_text.strip())]

    return headings


def chunk_text_block(text: str, max_chars=MAX_CHARS, overlap=OVERLAP) -> List[str]:
    """
    Overlapping character-based chunking that prefers sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    L = len(text)

    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end]

        # try to avoid cutting mid-sentence
        if end < L:
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            cut = max(last_period, last_newline)
            if cut != -1 and cut > int(max_chars * 0.3):
                end = start + cut + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end == L:
            break

        start = max(0, end - overlap)

    return [c for c in chunks if c]


# ---------- EMBEDDING HELPERS ----------
def load_embedder():
    print(f"Loading embedder: {EMBED_MODEL_NAME}")
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_embeddings(model, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return emb.astype("float32")


# ---------- CHROMA DB ----------
def init_chroma(persist_dir: str = CHROMA_DB_DIR, collection_name: str = COLLECTION_NAME):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    # Remove existing collection (rebuild) - optional
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}' (rebuild).")
    except Exception:
        pass

    collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def main():
    print("Loading KB...")
    raw = load_kb(KB_FILE)
    raw = normalize(raw)

    print("Splitting into sections...")
    sections = split_into_sections(raw)
    print(f"Found {len(sections)} section(s).")

    all_chunks: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    for sec_idx, (sec_header, sec_body) in enumerate(sections):
        # extract internal headings like AP-001: ...
        headings = extract_headings(sec_body)
        if not headings:
            # fallback: chunk whole section
            blocks = [("SectionBody", sec_body)]
        else:
            blocks = headings

        for head_idx, (heading_line, heading_body) in enumerate(blocks):
            # create smaller paragraphs first by splitting on double newlines
            paras = [p.strip() for p in re.split(r"\n{2,}", heading_body) if p.strip()]
            if not paras:
                paras = [heading_body.strip()]

            for para in paras:
                para_chunks = chunk_text_block(para)
                for c in para_chunks:
                    chunk_id = f"sec{sec_idx}_h{head_idx}_c{len(all_chunks)}"
                    # parse heading_line into code/title if possible
                    if ":" in heading_line:
                        code, title = heading_line.split(":", 1)
                        code = code.strip()
                        title = title.strip()
                    else:
                        code = ""
                        title = heading_line.strip()

                    meta = {
                        "section_header": sec_header,
                        "heading_line": heading_line,
                        "heading_code": code,
                        "heading_title": title,
                        "source_file": KB_FILE,
                    }
                    all_chunks.append(c)
                    metadatas.append(meta)
                    ids.append(chunk_id)

    # Fallback: if structured parsing produced no chunks, chunk the whole KB
    if not all_chunks:
        print("[WARN] Structured parsing produced no chunks. Falling back to full KB.")
        fallback_chunks = chunk_text_block(raw, max_chars=MAX_CHARS, overlap=OVERLAP)
        if not fallback_chunks:
            raise RuntimeError("No chunks created even from full KB. Check kb.txt content.")

        for i, c in enumerate(fallback_chunks):
            chunk_id = f"fallback_c{i}"
            meta = {
                "section_header": "Full KB",
                "heading_line": "Full KB",
                "heading_code": "",
                "heading_title": "Full KB",
                "source_file": KB_FILE,
            }
            all_chunks.append(c)
            metadatas.append(meta)
            ids.append(chunk_id)

    print(f"Total chunks created: {len(all_chunks)}")

    # Save chunks.csv
    print("Saving chunks.csv ...")
    df = pd.DataFrame({
        "chunk_id": ids,
        "section_header": [m["section_header"] for m in metadatas],
        "heading_line": [m["heading_line"] for m in metadatas],
        "heading_code": [m["heading_code"] for m in metadatas],
        "heading_title": [m["heading_title"] for m in metadatas],
        "chunk_text": all_chunks,
    })
    df.to_csv(CHUNKS_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(df)} chunks to {CHUNKS_CSV}")

    # Embeddings
    embedder = load_embedder()
    embeddings = get_embeddings(embedder, all_chunks)
    print("Embeddings generated.")

    # Chroma
    client, collection = init_chroma(CHROMA_DB_DIR, COLLECTION_NAME)
    # add to collection (Chroma requires python-native lists)
    print("Ingesting into ChromaDB collection...")
    collection.add(
        ids=ids,
        documents=all_chunks,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )
    print("Ingest complete.")
    print(f"Chroma DB persisted to: {CHROMA_DB_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
