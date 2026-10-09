"""
app/core/constants.py

Centralized constants for the RAG / LLM / Guardrails subsystem.
Keeping these in one place avoids magic numbers/strings scattered
across the codebase and makes tuning (chunk size, top_k, etc.) trivial.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]          # project root
DATA_DIR = BASE_DIR / "data"                             # raw source docs
VECTOR_DB_DIR = BASE_DIR / "vector_db"                    # persisted FAISS index
FAISS_INDEX_NAME = "healthcare_kb"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # switch to "cuda" if a GPU is available

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 4
SIMILARITY_SCORE_THRESHOLD = 0.50  # below this, we treat the chunk as irrelevant
MAX_CONTEXT_CHARS = 6000           # hard cap on context passed to the LLM

# ---------------------------------------------------------------------------
# LLM (served via OpenRouter's OpenAI-compatible API)
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Any model slug OpenRouter hosts, e.g. "google/gemini-2.0-flash-001",
# "openai/gpt-4o-mini", "anthropic/claude-3.5-haiku". Overridable via the
# OPENROUTER_MODEL_NAME env var — see llm/llm_factory.py.
OPENROUTER_MODEL_NAME_DEFAULT = "google/gemini-2.0-flash-001"
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 900
WEB_SEARCH_MAX_RESULTS = 5

# ---------------------------------------------------------------------------
# Guardrail response strings
# ---------------------------------------------------------------------------
NO_ANSWER_FOUND_MSG = (
    "I couldn't find information about that in the healthcare knowledge base "
    "I have access to. I don't want to guess, so please consult a qualified "
    "healthcare professional or rephrase your question."
)

DIAGNOSIS_REFUSAL_MSG = (
    "I'm not able to diagnose medical conditions, interpret your symptoms, "
    "prescribe medication, or recommend dosages or treatment plans. Please "
    "consult a qualified healthcare professional for that. I can, however, "
    "share general, source-backed information from the knowledge base."
)

EMERGENCY_MSG = (
    "⚠️ This sounds like it could be a medical emergency. Please call your "
    "local emergency number (911 in the US) or go to the nearest emergency "
    "room right now. If you are having thoughts of suicide or self-harm, "
    "please contact a crisis line immediately (e.g., 988 Suicide & Crisis "
    "Lifeline in the US). I'm not able to help further with this message — "
    "your safety comes first."
)

OUT_OF_SCOPE_MSG = (
    "I'm a healthcare knowledge assistant, so I can only help with "
    "healthcare and medical-terminology questions grounded in my knowledge "
    "base. Could you ask something related to that?"
)

GREETING_MSG = "Hello! I am your healthcare chatbot. How can I help you?"

PROMPT_INJECTION_MSG = (
    "I can't follow instructions that try to override my configured role, "
    "reveal internal system prompts, or bypass my guardrails. Happy to help "
    "with a healthcare question instead."
)

# ---------------------------------------------------------------------------
# Emergency keywords / signals (used by emergency_detector.py)
# ---------------------------------------------------------------------------
# Broad recall list: any of these appearing in the message is a *candidate*
# emergency mention. Whether it's a REAL emergency or just an informational
# question about the topic is then decided by the distress vs. informational
# signals below.
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "difficulty breathing", "shortness of breath severe", "severe bleeding",
    "suicidal", "suicide", "kill myself", "want to die", "self harm",
    "self-harm", "overdose", "unconscious", "not breathing", "anaphylaxis",
    "seizure", "choking", "severe allergic reaction",
]

# First-person, present-tense distress language. If any of these co-occur
# with an EMERGENCY_KEYWORDS hit, we treat it as a real emergency regardless
# of anything else in the message (safety-first).
EMERGENCY_DISTRESS_PATTERNS = [
    r"\bi'?m (having|experiencing)\b",
    r"\bi (have|am having)\b.{0,25}\b(chest pain|trouble breathing|difficulty breathing)\b",
    r"\bmy chest (hurts|is tight|feels heavy|is killing me)\b",
    r"\bi (think|feel like) i('m| am) (having|dying)\b",
    r"\bhelp me\b",
    r"\bcall (911|an ambulance|emergency services)\b",
    r"\bi can'?t breathe\b",
    r"\bthis is happening (to me )?right now\b",
    r"\bi feel (dizzy|faint|like i'?m going to pass out)\b",
    r"\bi'?m (bleeding|choking) (badly|a lot|right now)?\b",
    r"\bi (want to|am going to) (kill myself|hurt myself|end it)\b",
    r"\bright now\b.{0,20}\b(chest|breath|bleeding)\b",
]

# Educational/informational framing. If a keyword co-occurs with one of
# these AND no distress signal is present, treat it as a normal question,
# not an emergency.
EMERGENCY_INFORMATIONAL_PATTERNS = [
    r"\bwhat (is|are|causes|counts as)\b",
    r"\bsymptoms of\b",
    r"\bsigns of\b",
    r"\bhow (do|can) (i|you|someone) (know|tell|recognize|prevent|avoid)\b",
    r"\btell me about\b",
    r"\bexplain\b",
    r"\bdifference between\b",
    r"\bdefine\b",
    r"\brisk factors (for|of)\b",
    r"\bcauses of\b",
    r"\btreatment(s)? for\b",
    r"\binformation (on|about)\b",
    r"\bwhat should (i|someone) do if\b",
    r"\bhow common is\b",
    r"\bwhat happens during\b",
    r"\bcan you (describe|explain)\b",
]

EMERGENCY_INFO_SAFETY_NOTE = (
    "\n\n_Note: if you or someone nearby is experiencing these symptoms "
    "right now, please call emergency services immediately rather than "
    "waiting for an answer here._"
)

# ---------------------------------------------------------------------------
# Prompt-injection patterns (used by prompt_injection.py)
# ---------------------------------------------------------------------------
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior|the) instructions",
    r"disregard (all|any|previous|prior|the) instructions",
    r"reveal (your|the) system prompt",
    r"show (me )?(your|the) (system|hidden) prompt",
    r"act as (my|a) doctor",
    r"act as (my|a) physician",
    r"pretend (you are|to be) (a doctor|my doctor)",
    r"forget (the|your) knowledge base",
    r"forget (everything|all) (you know|above)",
    r"you are now",
    r"jailbreak",
    r"developer mode",
    r"bypass (your|the) (guardrails|rules|restrictions)",
]

# Diagnosis-seeking patterns (used by medical_guardrails.py)
DIAGNOSIS_PATTERNS = [
    r"do i have\b",
    r"am i (having|suffering)",
    r"what('s| is) wrong with me",
    r"diagnose me",
    r"what dose (of|should)",
    r"how many (mg|milligrams|pills)",
    r"what medication should i take",
    r"is this cancer",
    r"prescribe",
]