"""
RAG UI -- Streamlit
===================
Antarmuka web untuk pipeline RAG (Fase 2 + reranking Fase 4).

  Vector search (ChromaDB) -> jaring fetch_k kandidat
  Cross-encoder rerank     -> sisakan top_k terbaik   (bisa dimatikan)
  LLM (Groq)               -> menyusun jawaban dari konteks

Cara pakai (lokal):
    # set kunci API dulu (PowerShell):
    $env:GROQ_API_KEY = "gsk_...."
    streamlit run app.py

Atau masukkan kunci lewat sidebar saat aplikasi berjalan.
Butuh chroma_db sudah terisi (jalankan rag_chroma_fase2.py sekali kalau belum).
"""

import os
import streamlit as st

st.set_page_config(page_title="RAG Tata Surya", page_icon="🪐", layout="wide")


# =========================================================
# KUNCI API  (sidebar / secrets / environment)
# =========================================================
def resolve_api_key():
    # Prioritas: st.secrets -> environment -> input sidebar
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    key = key or os.environ.get("GROQ_API_KEY", "")
    return key


# =========================================================
# MUAT PIPELINE  (sekali saja, di-cache antar-rerun)
# =========================================================
@st.cache_resource(show_spinner="Memuat model embedding, reranker & ChromaDB...")
def load_pipeline():
    # Groq() di rag_chroma_fase2 dibuat saat import -> butuh env terisi
    # agar import tidak crash. Isi placeholder kalau belum ada kunci asli.
    os.environ.setdefault("GROQ_API_KEY", "placeholder-for-import")

    from rag_chroma_fase2 import retrieve, collection
    from rag_rerank_fase4 import retrieve_rerank

    return {
        "retrieve": retrieve,
        "retrieve_rerank": retrieve_rerank,
        "n_chunks": collection.count(),
    }


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key: str):
    # Klien terpisah supaya kunci dari sidebar langsung dipakai
    # (tidak terikat ke env saat import).
    from groq import Groq
    return Groq(api_key=api_key)


def build_answer(client, query, retrieved):
    context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
    prompt = f"""Jawab pertanyaan HANYA berdasarkan konteks di bawah.
Jika informasinya tidak ada di konteks, katakan "Tidak ditemukan di dokumen."
Sebutkan sumber [judul] yang kamu pakai.

Konteks:
{context}

Pertanyaan: {query}
Jawaban:"""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    default_key = resolve_api_key()
    api_key = st.text_input(
        "GROQ API Key",
        value=default_key,
        type="password",
        help="Dapatkan di console.groq.com. Bisa juga di-set lewat $env:GROQ_API_KEY atau .streamlit/secrets.toml",
    )

    use_rerank = st.toggle("Pakai reranking (cross-encoder)", value=True,
                           help="Tahap 2: nilai ulang kandidat agar chunk benar naik ke atas.")
    top_k = st.slider("Jumlah chunk dipakai (top_k)", 1, 8, 4)
    fetch_k = st.slider("Kandidat awal vector search (fetch_k)", top_k, 40, 20,
                        disabled=not use_rerank,
                        help="Hanya dipakai saat reranking aktif.")

    st.divider()
    st.caption("Pipeline: ChromaDB + e5-small + ms-marco reranker + Groq Llama-3.3-70B")


# =========================================================
# MUAT PIPELINE
# =========================================================
try:
    pipe = load_pipeline()
except Exception as e:
    st.error(f"Gagal memuat pipeline: {e}")
    st.stop()

if pipe["n_chunks"] == 0:
    st.error("ChromaDB kosong. Jalankan `python rag_chroma_fase2.py` sekali untuk mengindeks dulu.")
    st.stop()


# =========================================================
# HALAMAN UTAMA
# =========================================================
st.title("🪐 RAG Tata Surya")
st.caption(f"Tanya apa saja tentang tata surya — {pipe['n_chunks']} chunk terindeks. "
           f"Jawaban berbasis Wikipedia, disusun oleh LLM dengan kutipan sumber.")

with st.form("tanya"):
    query = st.text_input("Pertanyaan",
                          placeholder="mis. How many moons does Mars have?")
    submitted = st.form_submit_button("Tanya", type="primary")

if submitted and query.strip():
    if not api_key or api_key == "placeholder-for-import":
        st.warning("Masukkan GROQ API Key di sidebar dulu untuk menghasilkan jawaban.")
        st.stop()

    # ---- Retrieval (+ rerank) ----
    with st.spinner("Mencari konteks..."):
        if use_rerank:
            results = pipe["retrieve_rerank"](query, fetch_k=fetch_k, top_k=top_k)
        else:
            results = pipe["retrieve"](query, k=top_k)

    # ---- Generation ----
    with st.spinner("Menyusun jawaban..."):
        try:
            client = get_groq_client(api_key)
            ans = build_answer(client, query, results)
        except Exception as e:
            st.error(f"Gagal memanggil Groq: {e}")
            st.stop()

    # ---- Tampilkan ----
    st.subheader("Jawaban")
    st.markdown(ans)

    st.subheader("Sumber")
    for i, r in enumerate(results, 1):
        if use_rerank and "rerank_score" in r:
            label = (f"**{i}. {r['source']}** (chunk {r['chunk_id']}) — "
                     f"rerank `{r['rerank_score']:.2f}`, semula peringkat vektor #{r['vector_rank']}")
        else:
            label = f"**{i}. {r['source']}** (chunk {r['chunk_id']}) — skor `{r.get('score', 0):.3f}`"
        with st.expander(label):
            st.write(r["text"])
