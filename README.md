# 🪐 RAG Tata Surya

Aplikasi Retrieval-Augmented Generation (RAG) tentang tata surya, dengan UI Streamlit.

**Pipeline:** ChromaDB (vector search) → cross-encoder reranking → Groq Llama-3.3-70B.
Korpus: 8 artikel Wikipedia (Solar System, Mars, Jupiter, Saturn, Black hole, Venus, Earth, Moon), sudah ter-indeks di `chroma_db/`.

## Jalankan lokal

```powershell
pip install -r requirements.txt
$env:GROQ_API_KEY = "gsk_..."      # kunci dari console.groq.com
streamlit run app.py
```

Atau masukkan kunci lewat sidebar saat app berjalan. Buka http://localhost:8501.

## Deploy ke Streamlit Community Cloud

1. **Push ke GitHub.** Repo ini sudah siap di-commit (lihat langkah di bawah).
2. Buka **https://share.streamlit.io** → login dengan GitHub → **Create app** → **Deploy a public app from GitHub**.
3. Pilih repo ini, branch `main`, dan **Main file path** = `app.py`.
4. Klik **Advanced settings**:
   - **Python version**: pilih **3.12** (atau 3.13).
   - **Secrets**: tempel kunci API kamu:
     ```toml
     GROQ_API_KEY = "gsk_..."
     ```
5. **Deploy**. Build pertama agak lama (mengunduh torch + model embedding/reranker). Setelah itu cepat.

### Catatan deploy
- `chroma_db/` ikut di-commit, jadi app **tidak** perlu re-index dari Wikipedia saat start.
- `.streamlit/secrets.toml` **tidak** di-commit (ada di `.gitignore`) — kunci asli dimasukkan lewat panel Secrets di Cloud.
- Model `intfloat/multilingual-e5-small` dan `cross-encoder/ms-marco-MiniLM-L-6-v2` terunduh otomatis saat pertama jalan.

## File
| File | Isi |
|------|-----|
| `app.py` | UI Streamlit |
| `rag_chroma_fase2.py` | retrieval (ChromaDB) + generation (Groq) |
| `rag_rerank_fase4.py` | reranking cross-encoder |
| `rag_eval_fase3.py` | evaluasi (hit rate, MRR, LLM-as-judge) |
| `chroma_db/` | indeks vektor siap pakai |
