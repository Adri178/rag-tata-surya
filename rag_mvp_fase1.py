"""
RAG sederhana untuk pembelajaran -- Fase 1 (MVP)
====================================================
Corpus : artikel Wikipedia (pilih topik yang kamu kuasai!)
Alur   : ambil artikel -> chunk -> embed -> simpan -> retrieve -> jawab

Sengaja TANPA vector database dulu. Retrieval pakai cosine similarity
manual (numpy) supaya kamu paham apa yang sebenarnya dikerjakan vector DB.
Setelah ini jalan, baru ganti bagian retrieval dengan ChromaDB.

Cara pakai:
    pip install wikipedia sentence-transformers groq numpy
    export GROQ_API_KEY="...."   (daftar gratis di console.groq.com)
    python rag_mvp_fase1.py
"""

import os
import numpy as np
import time
import wikipedia
from sentence_transformers import SentenceTransformer

# =========================================================
# 1. AMBIL DATA
# =========================================================
# Ganti topik ini dengan sesuatu yang kamu kuasai supaya mudah
# mengecek kebenaran jawaban. Pakai "id" untuk artikel bahasa Indonesia.
wikipedia.set_lang("en")
TOPICS = ["Solar System", "Mars", "Jupiter", "Saturn", "Black hole"]


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
# 2. CHUNKING  (pecah teks panjang jadi potongan kecil)
# =========================================================
# overlap menjaga agar kalimat tidak terpotong di tengah konteks.
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
# 3. EMBEDDING  (ubah teks jadi vektor)
# =========================================================
print("Loading embedding model...")
model = SentenceTransformer("intfloat/multilingual-e5-small")


def embed(texts, prefix="passage: "):
    # model e5 butuh prefix: "passage: " untuk dokumen, "query: " untuk pertanyaan
    return model.encode([prefix + t for t in texts], normalize_embeddings=True)


# =========================================================
# 4. RETRIEVAL  (cari chunk paling mirip dengan pertanyaan)
# =========================================================
def retrieve(query, chunks, chunk_vecs, k=4):
    q_vec = embed([query], prefix="query: ")[0]
    # vektor sudah dinormalisasi, jadi dot product = cosine similarity
    scores = chunk_vecs @ q_vec
    top_idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# =========================================================
# 5. GENERATION  (LLM menjawab berdasarkan chunk yang diambil)
# =========================================================
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def answer(query, retrieved):
    context = "\n\n".join(f"[{r['source']}] {r['text']}" for r, _ in retrieved)
    prompt = f"""Jawab pertanyaan HANYA berdasarkan konteks di bawah.
Jika informasinya tidak ada di konteks, katakan "Tidak ditemukan di dokumen."
Sebutkan sumber [judul] yang kamu pakai.

Konteks:
{context}

Pertanyaan: {query}
Jawaban:"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # cek model aktif di console.groq.com/docs/models
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("1. Mengambil dokumen...")
    docs = load_documents(TOPICS)

    print("2. Memecah jadi chunk...")
    chunks = build_chunks(docs)
    print(f"   total {len(chunks)} chunks")

    print("3. Membuat embedding (sekali di awal)...")
    chunk_vecs = embed([c["text"] for c in chunks])

    print("\nSiap! Ketik pertanyaan (atau 'exit' untuk keluar).\n")
    while True:
        q = input("Tanya: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        retrieved = retrieve(q, chunks, chunk_vecs, k=4)
        print("\n--- Jawaban ---")
        print(answer(q, retrieved))
        print("\n--- Sumber teratas ---")
        for r, s in retrieved:
            print(f"  [{s:.3f}] {r['source']} (chunk {r['chunk_id']})")
        print()
