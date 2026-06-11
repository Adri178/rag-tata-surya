"""
Evaluasi RAG -- Fase 3
=========================
Mengukur kualitas dengan ANGKA, bukan sekadar "kelihatannya benar".

Dua dimensi yang diukur terpisah:
  1. RETRIEVAL  -> apakah chunk dari artikel yang BENAR muncul di top-k?
                  Metrik: hit rate@k dan MRR. Tidak butuh LLM, cepat,
                  dan tidak terpengaruh bahasa jawaban.
  2. JAWABAN    -> apakah jawaban akhir benar & berdasar konteks (tidak ngarang)?
                  Metrik: LLM-as-judge (model lain menilai jawaban).

Butuh file rag_chroma_fase2.py di folder yang SAMA.

Cara pakai:
    python rag_eval_fase3.py
"""

from rag_chroma_fase2 import (
    retrieve, answer, collection, client,
    load_documents, build_chunks, index_chunks, TOPICS,
)

# Pastikan koleksi sudah terisi (kalau belum, index dulu).
if collection.count() == 0:
    print("Koleksi kosong -> indexing dulu...")
    index_chunks(build_chunks(load_documents(TOPICS)))


# =========================================================
# TEST SET
# =========================================================
# Tulis pertanyaan yang kamu TAHU jawabannya, beserta artikel sumber
# yang SEHARUSNYA dipakai. Inilah "kunci jawaban" untuk menilai retrieval.
# Tambah terus seiring waktu -- makin banyak, makin terpercaya angkanya.
TESTSET = [
    {"q": "How many moons does Mars have?",                  "source": "Mars"},
    {"q": "What is the largest planet in the Solar System?", "source": "Jupiter"},
    {"q": "What is the largest moon of Saturn?",             "source": "Saturn"},
    {"q": "Which planet is the hottest in the Solar System?","source": "Venus"},
    {"q": "What is Mars commonly nicknamed?",                "source": "Mars"},
    {"q": "How far is the Moon from Earth on average?",      "source": "Moon"},
    {"q": "How many planets are in the Solar System?",       "source": "Solar System"},
    {"q": "What is the boundary of a black hole called?",    "source": "Black hole"},
    {"q": "What is Saturn most famous for?",                 "source": "Saturn"},
    {"q": "What gases make up most of Jupiter?",             "source": "Jupiter"},
]


# =========================================================
# METRIK 1 -- RETRIEVAL  (hit rate@k & MRR)
# =========================================================
def eval_retrieval(testset, k=4):
    """hit_rate@k = berapa % pertanyaan yang artikel benarnya muncul di top-k.
       MRR       = rata-rata 1/peringkat kemunculan pertama artikel benar."""
    hits, reciprocal_rank = 0, 0.0
    for item in testset:
        sources = [r["source"] for r in retrieve(item["q"], k=k)]
        if item["source"] in sources:
            hits += 1
            rank = sources.index(item["source"]) + 1   # peringkat 1 = paling atas
            reciprocal_rank += 1 / rank
    n = len(testset)
    return hits / n, reciprocal_rank / n


# =========================================================
# METRIK 2 -- JAWABAN  (LLM-as-judge)
# =========================================================
def judge_answer(question, retrieved, ans):
    """Model menilai apakah jawaban benar DAN berdasar konteks.
       Lebih tahan bahasa & sinonim dibanding pencocokan kata kunci."""
    context = "\n\n".join(r["text"] for r in retrieved)
    prompt = f"""Anda adalah evaluator yang ketat. Jawab HANYA dengan satu kata: YA atau TIDAK.

Apakah JAWABAN di bawah menjawab PERTANYAAN dengan benar DAN sepenuhnya didukung oleh KONTEKS (tidak mengarang fakta di luar konteks)?

KONTEKS:
{context}

PERTANYAAN: {question}
JAWABAN: {ans}

Benar dan didukung konteks? (YA/TIDAK):"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = resp.choices[0].message.content.strip().lower()
    return verdict.startswith("ya")


# =========================================================
# JALANKAN EVALUASI
# =========================================================
if __name__ == "__main__":
    K = 4
    print(f"\nMengevaluasi {len(TESTSET)} pertanyaan (k={K})...\n")
    print(f"{'#':>2}  {'retrieval':<12} {'jawaban':<8}  pertanyaan")
    print("-" * 70)

    good_answers = 0
    for i, item in enumerate(TESTSET, 1):
        retrieved = retrieve(item["q"], k=K)
        sources = [r["source"] for r in retrieved]

        if item["source"] in sources:
            rank = sources.index(item["source"]) + 1
            retr_str = f"OK (rank {rank})"
        else:
            retr_str = "MISS"

        ans = answer(item["q"], retrieved)
        ok = judge_answer(item["q"], retrieved, ans)
        good_answers += ok

        print(f"{i:>2}  {retr_str:<12} {'OK' if ok else 'BURUK':<8}  {item['q']}")

    # ---- Ringkasan ----
    hit_rate, mrr = eval_retrieval(TESTSET, k=K)
    ans_quality = good_answers / len(TESTSET)

    print("-" * 70)
    print("\n=== RINGKASAN ===")
    print(f"  Hit rate@{K}      : {hit_rate:.0%}   (retrieval menemukan artikel benar)")
    print(f"  MRR             : {mrr:.3f}  (makin tinggi = artikel benar makin di atas)")
    print(f"  Kualitas jawaban: {ans_quality:.0%}   (dinilai oleh LLM-as-judge)")

    # ---- Kurva hit rate vs k (cepat, tanpa LLM) ----
    print("\n=== Hit rate untuk berbagai nilai k ===")
    print("  (lihat bagaimana mengambil lebih banyak chunk menaikkan peluang menemukan artikel benar)")
    for k in [1, 2, 4, 8]:
        hr, _ = eval_retrieval(TESTSET, k=k)
        bar = "#" * int(hr * 20)
        print(f"  k={k:<2} {hr:>4.0%}  {bar}")

    print("\nIni baseline-mu. Catat angka ini, ubah satu hal (ukuran chunk, nilai k,")
    print("atau tambah reranking), lalu jalankan lagi -- bandingkan apakah membaik.")
