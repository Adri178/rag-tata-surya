"""
RAG dengan ChromaDB -- Fase 2 (vector database sungguhan)
============================================================
Perubahan dari Fase 1: retrieval numpy manual diganti ChromaDB.
Keuntungan: embedding dihitung SEKALI lalu disimpan ke disk.
Jalankan kedua kalinya -> langsung siap, tanpa download & embedding ulang.

Cara pakai:
    pip install wikipedia sentence-transformers groq chromadb
    export GROQ_API_KEY="...."      (Windows: set GROQ_API_KEY=....)
    python rag_chroma_fase2.py

Mau index ulang (mis. ganti topik)? Hapus folder ./chroma_db lalu jalankan lagi.
"""

import os
import time
import wikipedia
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# =========================================================
# 1. AMBIL DATA  (loader dengan retry, dari perbaikan sebelumnya)
# =========================================================
wikipedia.set_lang("en")            # "id" untuk artikel bahasa Indonesia
TOPICS = ["Solar System", "Mars", "Jupiter", "Saturn", "Black hole",
          "Venus", "Earth", "Moon"]


def load_documents(topics, retries=3):
    docs = []
    for t in topics:
        for attempt in range(retries):
            try:
                page = wikipedia.page(t, auto_suggest=False)
                docs.append({"title": page.title, "text": page.content})
                print(f"  loaded: {page.title} ({len(page.content)} chars)")
                break
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"  skip {t}: ambigu -> {e.options[:3]}")
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1.5)
                else:
                    print(f"  skip {t}: {e}")
        time.sleep(0.5)
    return docs


# =========================================================
# 2. CHUNKING
# =========================================================
def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_chunks(docs):
    all_chunks = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"])):
            all_chunks.append({"source": d["title"], "chunk_id": i, "text": c})
    return all_chunks


# =========================================================
# 3. EMBEDDING
# =========================================================
print("Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-small")


def embed(texts, prefix="passage: "):
    # e5: "passage: " untuk dokumen, "query: " untuk pertanyaan
    return model.encode([prefix + t for t in texts], normalize_embeddings=True)


# =========================================================
# 4. VECTOR DATABASE  (ChromaDB, tersimpan ke disk)
# =========================================================
# PersistentClient menulis ke folder ./chroma_db. Data bertahan antar-run.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Pakai jarak cosine -- cocok untuk embedding yang sudah dinormalisasi.
# Kalau versi chromadb-mu lawas dan baris 'configuration' error,
# ganti dengan:  metadata={"hnsw:space": "cosine"}
collection = chroma_client.get_or_create_collection(
    name="wiki_rag",
    configuration={"hnsw": {"space": "cosine"}},
)


def index_chunks(chunks):
    print("   membuat embedding + memasukkan ke ChromaDB...")
    vecs = embed([c["text"] for c in chunks])          # prefix "passage: "
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=vecs.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "chunk_id": c["chunk_id"]}
                   for c in chunks],
    )
    print(f"   {len(chunks)} chunk tersimpan ke disk")


# =========================================================
# 5. RETRIEVAL  (query ke ChromaDB, bukan numpy lagi)
# =========================================================
def retrieve(query, k=4):
    q_vec = embed([query], prefix="query: ")[0]
    res = collection.query(query_embeddings=[q_vec.tolist()], n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0],
                               res["metadatas"][0],
                               res["distances"][0]):
        out.append({
            "text": doc,
            "source": meta["source"],
            "chunk_id": meta["chunk_id"],
            "score": 1 - dist,          # cosine distance -> similarity
        })
    return out


# =========================================================
# 6. GENERATION  (LLM, sama seperti sebelumnya)
# =========================================================
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def answer(query, retrieved):
    context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
    prompt = f"""Jawab pertanyaan HANYA berdasarkan konteks di bawah.
Jika informasinya tidak ada di konteks, katakan "Tidak ditemukan di dokumen."
Sebutkan sumber [judul] yang kamu pakai.

Konteks:
{context}

Pertanyaan: {query}
Jawaban:"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # cek model aktif: console.groq.com/docs/models
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    # Indexing hanya dijalankan kalau koleksi masih kosong.
    if collection.count() == 0:
        print("Koleksi masih kosong -> indexing (hanya sekali)...")
        docs = load_documents(TOPICS)
        chunks = build_chunks(docs)
        index_chunks(chunks)
    else:
        print(f"Memakai koleksi tersimpan ({collection.count()} chunk). "
              "Hapus folder ./chroma_db untuk indexing ulang.")

    print("\nSiap! Ketik pertanyaan (atau 'exit' untuk keluar).\n")
    while True:
        q = input("Tanya: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        retrieved = retrieve(q, k=4)
        print("\n--- Jawaban ---")
        print(answer(q, retrieved))
        print("\n--- Sumber teratas ---")
        for r in retrieved:
            print(f"  [{r['score']:.3f}] {r['source']} (chunk {r['chunk_id']})")
        print()
