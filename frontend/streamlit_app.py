"""
frontend/streamlit_app.py

Chat UI for the Healthcare Knowledge Assistant. Talks to the FastAPI
backend over HTTP — run the backend first:
    uvicorn app.main:app --reload
Then:
    streamlit run frontend/streamlit_app.py
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Healthcare Knowledge Assistant", page_icon="🩺")
st.title("🩺 Healthcare Knowledge Assistant")
st.caption("Answers are grounded only in the ingested knowledge base — not medical advice.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, citations, blocked_reason}]

with st.sidebar:
    st.header("Conversation")
    if st.button("New conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.history_choice = None
        st.rerun()

    try:
        conversations_response = requests.get(f"{API_BASE}/conversations", timeout=10)
        conversations_response.raise_for_status()
        conversations = conversations_response.json()
    except requests.RequestException as e:
        conversations = []
        st.error(f"Could not load saved conversations: {e}")

    conversation_by_id = {item["id"]: item for item in conversations}
    st.selectbox(
        "Saved conversations",
        options=[None, *conversation_by_id],
        format_func=lambda conversation_id: (
            "Select a conversation"
            if conversation_id is None
            else f"{conversation_by_id[conversation_id]['title']} · "
            f"{conversation_by_id[conversation_id]['message_count']} messages · "
            f"{conversation_id[:8]}"
        ),
        key="history_choice",
    )
    if st.button("Load conversation", disabled=not st.session_state.history_choice):
        try:
            detail_response = requests.get(
                f"{API_BASE}/conversations/{st.session_state.history_choice}",
                timeout=10,
            )
            detail_response.raise_for_status()
            conversation = detail_response.json()
            st.session_state.conversation_id = conversation["id"]
            st.session_state.messages = conversation["messages"]
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Could not load that conversation: {e}")

    st.divider()
    st.caption(
        "Document ingestion is handled by the RAG team's ingestion script "
        "(app/rag/rag_pipeline.py::ingest_knowledge_base), not this UI."
    )

BLOCKED_LABELS = {
    "emergency": "🚨 Emergency safety response",
    "injection": "🛡️ Blocked: instruction override attempt",
    "medical": "⚕️ Blocked: needs a licensed professional",
    "out_of_scope": "❓ Out of scope for this assistant",
}


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("blocked_reason") in BLOCKED_LABELS:
            st.caption(BLOCKED_LABELS[msg["blocked_reason"]])
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"Sources ({len(msg['citations'])})"):
                for i, c in enumerate(msg["citations"], start=1):
                    page = f", page {c['page']}" if c.get("page") else ""
                    st.markdown(f"**[{i}] {c['source']}{page}**")
                    if c.get("snippet"):
                        st.caption(c["snippet"])


for msg in st.session_state.messages:
    render_message(msg)

if prompt := st.chat_input("Ask about medical terms, care guidelines, diseases, or insurance..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "citations": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"conversation_id": st.session_state.conversation_id, "message": prompt},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.conversation_id = data["conversation_id"]

                if data.get("blocked_reason") in BLOCKED_LABELS:
                    st.caption(BLOCKED_LABELS[data["blocked_reason"]])
                st.markdown(data["answer"])

                if data.get("citations"):
                    with st.expander(f"Sources ({len(data['citations'])})"):
                        for i, c in enumerate(data["citations"], start=1):
                            page = f", page {c['page']}" if c.get("page") else ""
                            st.markdown(f"**[{i}] {c['source']}{page}**")
                            if c.get("snippet"):
                                st.caption(c["snippet"])

                st.session_state.messages.append({
                    "role": "assistant", "content": data["answer"],
                    "citations": data.get("citations", []),
                    "blocked_reason": data.get("blocked_reason"),
                })
            except requests.RequestException as e:
                error_text = f"Something went wrong talking to the backend: {e}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text, "citations": []}
                )
