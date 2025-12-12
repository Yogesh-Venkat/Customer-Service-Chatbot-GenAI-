# RAG-Based Customer Service Chatbot using Gemini (Gen AI)

A Retrieval-Augmented Generation (RAG) chatbot that uses a ChromaDB knowledge base and Google Gemini (GenAI) to answer customer support queries. This repository includes scripts to prepare the knowledge base, create embeddings, persist to ChromaDB, and run a Streamlit UI for interactive queries.

---

## 🔧 Project Structure
```
├── kb.txt                      # Source knowledge base (text)
├── prepare_kb_chroma.py        # Parse, chunk, embed, save chunks.csv, persist to ChromaDB
├── chunks.csv                  # Generated (after running prepare_kb_chroma.py)
├── chroma_db/                  # Persisted ChromaDB (after ingestion)
├── app.py         # Streamlit app (RAG UI using ChromaDB + Gemini)
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## ✅ Features
- Parse and chunk knowledge base text into overlapping chunks.
- Generate embeddings using `sentence-transformers` (all-MiniLM-L6-v2).
- Persist chunks and embeddings into a local ChromaDB persistent store.
- Streamlit chat UI that queries ChromaDB, builds a prompt, and generates answers with Google Gemini.
- Debug & source inspection UI in Streamlit.

---

## ⚙️ Prerequisites
- Python 3.8 or newer
- Git
- (Optional) GPU for faster embedding generation (sentence-transformers)
- Google Cloud API access and a Gemini-compatible API key (set as `GOOGLE_API_KEY`)

---

## 🛠️ Setup (Local)

1. Clone the repo:
```bash
git clone <repo-url>
cd <repo-dir>
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your Google API key locally:
Create a file named `GOOGLE_API_KEY.env` at the repo root with the following content (no spaces):
```
GOOGLE_API_KEY=AIza...your_key_here...
```
and ensure `GOOGLE_API_KEY.env` is in `.gitignore`.

> Alternatively, set `GOOGLE_API_KEY` as an environment variable in your shell:
```bash
export GOOGLE_API_KEY=AIza...your_key_here...     # Linux/macOS
setx GOOGLE_API_KEY "AIza...your_key_here..."     # Windows (persisted)
```

---

## 📦 Prepare Knowledge Base (Run once or after updates)
This step parses `kb.txt`, chunks the text, generates embeddings, saves `chunks.csv`, and persists into a local ChromaDB directory.

```bash
python prepare_kb_chroma.py
```

- Output:
  - `chunks.csv` (metadata + chunk text)
  - `chroma_db/` (persistent ChromaDB folder)

If the script raises FileNotFoundError, ensure `kb.txt` exists in repo root.

---

## ▶️ Run Streamlit App (Local)
Start the interactive RAG chatbot UI:

```bash
streamlit run app.py
```

Open the provided URL in your browser (default: `http://localhost:8501`). The app expects `chroma_db/` to exist (run the prepare script first) and the `GOOGLE_API_KEY` environment variable to be set.

---

## 🔒 Handling Secrets & GitHub Actions
**Add the `GOOGLE_API_KEY` to GitHub repository secrets** (`Settings → Secrets → Actions → New repository secret`) with the name `GOOGLE_API_KEY`.

### Example GitHub Actions workflow to build the KB and upload artifact
Create `.github/workflows/build-kb.yml`:
```yaml
name: Build KB & Upload ChromaDB
on: [push]

jobs:
  build-kb:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Create GOOGLE_API_KEY.env
        run: |
          echo "GOOGLE_API_KEY=${{ secrets.GOOGLE_API_KEY }}" > GOOGLE_API_KEY.env
      - name: Run prepare script
        run: python prepare_kb_chroma.py
      - name: Upload chroma_db artifact
        uses: actions/upload-artifact@v4
        with:
          name: chroma_db
          path: chroma_db/
```
This builds the KB using the secret in Actions and uploads `chroma_db` as an artifact for later download or deployment.


---

## 🧾 requirements.txt (example)
```
streamlit
sentence-transformers
chromadb
google-generativeai
python-dotenv
pandas
numpy
```
Pin versions in your repo for reproducibility.

---

## 🧪 Troubleshooting
- `FileNotFoundError: kb.txt not found` → Add `kb.txt` to repo root.
- `No ChromaDB folder` → Run `python prepare_kb_chroma.py` before launching Streamlit.
- `Missing GOOGLE_API_KEY` → Ensure `GOOGLE_API_KEY.env` exists or environment variable is set.
- Long embedding times → Consider smaller embedding batches or GPU acceleration.

---

## 📝 Notes & Security
- Never commit `GOOGLE_API_KEY.env` or `chroma_db/` containing secrets. Add them to `.gitignore` or handle via secrets management.
- Use a `.env.example` (without keys) to show the required env variables.
- Consider using Pinecone or hosted vector DB for production.

---

## ✨ Contribution
Contributions are welcome! Open issues for bugs or feature requests. Submit PRs for improvements, model-switching, or deployment options.

---

## © License
Specify your preferred license (e.g., MIT) in `LICENSE` file.
