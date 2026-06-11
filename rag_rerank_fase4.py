"""
RAG dengan Reranking -- Fase 4
=================================
Pola dua tahap:
  1. Vector search (cepat)  -> jaring 20 kandidat dari ChromaDB
  2. Cross-encoder (teliti) -> nilai ulang 20 itu, sisakan 4 terbaik

Bi-encoder (e5) meng-embed pertanyaan & dokumen terpisah: cepat tapi kasar.
Cross-encoder membaca keduanya bersamaan: akurat tapi lambat -> hanya
dipakai pada sedikit kandidat hasil tahap 1.

Tidak perlu install baru (cross-encoder bagian dari sentence-transformers).
Model reranker akan terunduh otomatis saat pertama dijalankan.

Butuh: rag_chroma_fase2.py dan rag_eval_fase3.py di folder yang sama.

Cara pakai:
    python rag_rerank_fase4.py
"""

from sentence_transformers import CrossEncoder
from rag_chroma_fase2 import retrieve as retrieve_base, answer
from rag_eval_fase3 import TESTSET

# Reranker. Kecil & cepat, tapi English-only (korpus kita Wikipedia English).
# Kalau nanti pertanyaanmu banyak Bahasa Indonesia, ganti dengan multilingual:
#   "BAAI/bge-reranker-v2-m3"  (lebih akurat lintas bahasa, tapi lebih berat)
print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# =========================================================
# RETRIEVAL + RERANK
# =========================================================
def retrieve_rerank(query, fetch_k=20, top_k=4):
    # Tahap 1: jaring banyak kandidat lewat vector search (cepat)
    candidates = retrieve_base(query, k=fetch_k)
    for pos, c in enumerate(candidates, 1):
        c["vector_rank"] = pos                      # catat peringkat asli

    # Tahap 2: cross-encoder menilai ulang setiap (pertanyaan, chunk)
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    # Urutkan ulang berdasar skor reranker, ambil top_k terbaik
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


# =========================================================
# PERBANDINGAN: sebelum vs sesudah reranking (pakai TESTSET)
# =========================================================
def score_retriever(retrieve_fn, testset, k=4):
    hits, rr = 0, 0.0
    for item in testset:
        sources = [r["source"] for r in retrieve_fn(item["q"], k)]
        if item["source"] in sources:
            hits += 1
            rr += 1 / (sources.index(item["source"]) + 1)
    n = len(testset)
    return hits / n, rr / n


def base_4(q, k=4):
    return retrieve_base(q, k=k)


def rerank_4(q, k=4):
    return retrieve_rerank(q, fetch_k=20, top_k=k)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("\nMengukur dampak reranking pada TESTSET...\n")
    hr_base, mrr_base = score_retriever(base_4, TESTSET, k=4)
    hr_rer,  mrr_rer  = score_retriever(rerank_4, TESTSET, k=4)

    print("=== Dampak reranking (k=4) ===")
    print(f"  {'':<16}{'hit_rate@4':>12}{'MRR':>9}")
    print(f"  {'Tanpa rerank':<16}{hr_base:>11.0%}{mrr_base:>9.3f}")
    print(f"  {'Dengan rerank':<16}{hr_rer:>11.0%}{mrr_rer:>9.3f}")
    print("\n  (Perhatikan MRR -- reranking biasanya mengangkat chunk yang benar")
    print("   ke peringkat lebih atas, jadi MRR yang paling terasa naik.)")

    # Mode interaktif: lihat reranking menyusun ulang hasil secara langsung
    print("\nKetik pertanyaan untuk melihat reranking bekerja (atau 'exit').\n")
    while True:
        q = input("Tanya: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        results = retrieve_rerank(q, fetch_k=20, top_k=4)
        print("\n--- Jawaban ---")
        print(answer(q, results))
        print("\n--- Chunk terpilih (setelah rerank) ---")
        for r in results:
            print(f"  rerank {r['rerank_score']:6.2f} | semula peringkat vektor #{r['vector_rank']:<2} "
                  f"| {r['source']} (chunk {r['chunk_id']})")
        print()
