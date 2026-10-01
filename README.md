# Healthcare Knowledge Assistant — full project

This is the complete, aligned project: your collaborator's real
`app/rag/`, `app/llm/`, `app/core/`, and `app/guardrails/` files,
merged with the `app/database/`, `app/api/`, `frontend/`, and
`app/main.py` backend/frontend layer, plus a corrected test suite.

## Status as of this build
- All 24 Python files traced against each other for interface
  consistency — imports, function signatures, dataclass fields all match.
- `tests/test_guardrails.py` REPLACED — the version that was floating
  around imported from a nonexistent `guardrails` package with
  function names that no longer exist (leftover from an early
  pre-alignment draft). This version tests the real
  `app.guardrails.*` interface and passes.
- 14/14 tests pass with the real LangChain/FAISS stack installed.
- `vector_db/healthcare_kb.faiss` + `.pkl` are your already-built
  index — ingestion has already run successfully at some point.

## Run it
```bash
pip install -r requirements.txt -r requirements-rag.txt
cp .env.example .env   # add your real OPENROUTER_API_KEY

# If you need to (re-)ingest the knowledge base docs (put them in data/ first):
python -m scripts.ingest

# Quick sanity check without the API layer:
python try_it.py

# Run the test suite:
pytest tests/ -v

# Run the backend:
uvicorn app.main:app --reload
python -m uvicorn main:app --reload

# In a second terminal, run the frontend:
streamlit run frontend/streamlit_app.py
```

## Known non-blocking items
- `requirements-rag.txt` has `langgraph>=0.2.0` unpinned — pin it to
  whatever version was tested if you hit LangGraph API surprises.
- `app/guardrails/response_filter.py` imports `app.rag.citation_generator`,
  which pulls in the full LangChain/FAISS stack just to use the
  `Citation`/`format_sources_line` helpers. This means *any* guardrail
  import requires the full RAG stack to be installed — worth knowing
  if you ever want to unit-test guardrails in a lighter environment.
