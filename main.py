import os
import re
import faiss
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Literal

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# 1. PAGE CONFIG & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Mastek Enterprise BI Assistant",
    page_icon="📊",
    layout="wide",
)

PROJECT_PATH = Path(__file__).resolve().parent
ENV_FILE = PROJECT_PATH / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE, override=False)

GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
).strip()

if not GROQ_API_KEY:
    st.error("Missing `GROQ_API_KEY`. Add it in Streamlit App Settings → Secrets.")
    st.stop()

MODEL = "openai/gpt-oss-120b"

# ------------------------------------------------------------------------------
# 2. SYSTEM PROMPT & CONSTANTS
# ------------------------------------------------------------------------------
CLEAN_SYSTEM_PROMPT = """
You are the Mastek Enterprise Business Intelligence & Strategic Solutions Assistant (grounded strictly in https://www.mastek.com/ and authorized internal Mastek data).

### STRICT DOMAIN BOUNDARY (MANDATORY):
- You ONLY answer questions directly related to **Mastek** (its financials, corporate strategy, services, AI/Cloud solutions, internal projects, business units, or enterprise use cases relevant to Mastek).
- If the user asks a general knowledge question, coding question, personal question, or anything unrelated to Mastek (even if framed as a generic business idea not tied to Mastek), **DO NOT ANSWER IT**.
- Instead, reply strictly with:
  > "I am restricted to answering questions exclusively related to **Mastek** (https://www.mastek.com/) and authorized internal Mastek operations. Please ask a Mastek-related question."

### HOW TO ANSWER MASTEK QUERIES:
1. **Use Retrieved Context First (Grounded Facts):**
   - Check `<retrieved_chunks>`. If relevant Mastek data or internal metrics are present AND `<rbac_verified>` is `true` (or the chunk is `PUBLIC_EXTERNAL`), incorporate those exact numbers and cite them strictly in this exact format (never wrap citations in backticks and never omit closing parentheses):
     [PUBLIC] (Chunk: CHK-PUB-2025-Q3-01 | Source: Mastek Q3 FY25 Investor Presentation | Period: Q3 FY25)
     or [INTERNAL] (Chunk: <id> | Source: <source> | Period: <period>).
   - If `<retrieved_chunks>` is empty or a chunk is marked `CONFIDENTIAL_INTERNAL` while `<rbac_verified>` is `false`, **do not guess or reveal confidential metrics**; state clearly that authorized internal records are unavailable for your current access level and discuss only Mastek's publicly available capabilities if relevant.

2. **Executive Recommendations (Mastek-Scoped Only):**
   - Always write out full time units in tables and prose (e.g., `0-6 months`, `6-18 months`, `18-36 months`) — never abbreviate `months` as `mo`.
   - For valid Mastek business/strategy queries, tailor your response to `<authenticated_role>` using Markdown headings, bullet points, and tables covering:
     - **Direct Answer / Mastek Strategic Alignment**
     - **Actionable Implementation Roadmap**
     - **Financial & KPI Impact**
     - **Key Risks & Mitigation Strategies**
"""

REFUSAL_MESSAGE = (
    "I can only answer questions related to **Mastek** "
    "([https://www.mastek.com/](https://www.mastek.com/)) and authorized internal Mastek operations. "
    "Please ask a Mastek-specific question."
)

# ------------------------------------------------------------------------------
# 3. KNOWLEDGE BASE
# ------------------------------------------------------------------------------
KNOWLEDGE_CHUNKS = [
    {
        "chunk_id": "CHK-PUB-2025-Q3-01",
        "entity": "Mastek Limited",
        "classification": "PUBLIC_EXTERNAL",
        "source": "Mastek Q3 FY25 Investor Presentation",
        "reporting_period": "Q3 FY25",
        "verification_status": "UNVERIFIED_SOURCE",
        "content": (
            "Mastek reported Q3 FY25 operating revenue of INR 869.5 crore, representing 11.0% "
            "year-on-year growth in reported currency. The 12-month order backlog stood at "
            "INR 2,145.8 crore. Operating EBITDA margin was 16.2%."
        ),
    },
    {
        "chunk_id": "CHK-PUB-2025-AR-04",
        "entity": "Mastek Limited",
        "classification": "PUBLIC_EXTERNAL",
        "source": "Mastek Annual Report FY25",
        "reporting_period": "FY24-25",
        "verification_status": "UNVERIFIED_SOURCE",
        "content": (
            "Mastek's stated strategic focus includes AI-led digital transformation, Oracle Cloud "
            "modernization, Data and AI initiatives including ADOPT AI and OpenAna platforms, and "
            "Digital and Cloud Engineering. Its business operates across the UK, North America, and AMEA markets."
        ),
    },
    {
        "chunk_id": "CHK-INT-PRJ-882",
        "entity": "Project Titan",
        "classification": "CONFIDENTIAL_INTERNAL",
        "source": "Internal PMO Delivery Governance & RAID Register",
        "reporting_period": "November 2026",
        "verification_status": "INTERNAL_SOURCE_REQUIRES_VALIDATION",
        "content": (
            "As of November 2026, Sprint 14, Project Titan, an Oracle Cloud ERP migration project, "
            "is three weeks behind baseline milestone M4. The verified budget burn rate is 108% of budget. "
            "RAID item R-12 identifies pending UAT sign-off dependencies from the client security team. "
            "Internal resource attrition is reported at 0%."
        ),
    },
    {
        "chunk_id": "CHK-INT-FIN-409",
        "entity": "UK Healthcare Business Unit",
        "classification": "CONFIDENTIAL_INTERNAL",
        "source": "Internal BU P&L & Utilisation Telemetry Dashboard",
        "reporting_period": "October 2026 MTD",
        "verification_status": "INTERNAL_SOURCE_REQUIRES_VALIDATION",
        "content": (
            "In October 2026 month-to-date, UK Healthcare Business Unit billable utilisation declined "
            "from 84.5% to 79.1%, a decrease of 5.4 percentage points. The reported reason was the "
            "transition of 14 senior architects from Project Alpha to the bench. Gross margin realization "
            "was 24.3%, compared with a target of 27.0%, representing a gap of 2.7 percentage points."
        ),
    },
    {
        "chunk_id": "CHK-PUB-2026-SOL-05",
        "entity": "Mastek Enterprise Solutions",
        "classification": "PUBLIC_EXTERNAL",
        "source": "Mastek Enterprise AI & Services-as-Software Brief",
        "reporting_period": "FY26",
        "verification_status": "VERIFIED_PUBLIC_SOURCE",
        "content": (
            "Mastek delivers outcome-led enterprise solutions across key business functions: "
            "(1) Finance & BFSI: AI-driven invoice automation and real-time fraud detection improving detection rates by 15% and reclaiming 300+ operational hours weekly; "
            "(2) Operations & Supply Chain: IoT predictive maintenance, demand forecasting, and dynamic route optimization; "
            "(3) Digital Engineering & Legacy Modernization: AI-led code refactoring and automated QA achieving 35%+ SDLC efficiency gains and 40-50% manual effort reduction; "
            "(4) HR & Sales Intelligence: Agentic HR screening, attrition prediction, and real-time deal intelligence systems."
        ),
    },
]

# ------------------------------------------------------------------------------
# 4. CACHED RESOURCES (LOADS ONCE)
# ------------------------------------------------------------------------------
@st.cache_resource
def init_resources():
    groq_client = Groq(api_key=GROQ_API_KEY)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{c['entity']} | {c['content']}" for c in KNOWLEDGE_CHUNKS]
    vecs = embed_model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(vecs)
    return groq_client, embed_model, vecs

client, embedder, embeddings = init_resources()

# ------------------------------------------------------------------------------
# 5. DATA MODELS & RBAC RETRIEVAL LOGIC
# ------------------------------------------------------------------------------
class UserSession(BaseModel):
    authenticated_role: Literal[
        "CEO", "COO", "CFO", "CTO", "CMO",
        "Regional Presidents", "Vice Presidents",
        "Directors", "Associate Directors",
        "Associate Lead Managers", "Project Managers",
        "Technical Architects",
    ]
    authorized_classification: Literal["PUBLIC_EXTERNAL", "CONFIDENTIAL_INTERNAL"]
    rbac_verified: bool


def escape_xml_attr(value) -> str:
    if value is None:
        return "N/A"
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def get_authorized_indices(user_session: UserSession) -> List[int]:
    authorized = []
    for i, chunk in enumerate(KNOWLEDGE_CHUNKS):
        if chunk["classification"] == "PUBLIC_EXTERNAL":
            authorized.append(i)
        elif (
            chunk["classification"] == "CONFIDENTIAL_INTERNAL"
            and user_session.authorized_classification == "CONFIDENTIAL_INTERNAL"
            and user_session.rbac_verified
        ):
            authorized.append(i)
    return authorized


def search_faiss(user_query: str, allowed_indices: List[int], top_k: int = 3):
    if not allowed_indices:
        return [], []
    allowed_vectors = embeddings[allowed_indices].copy()
    search_index = faiss.IndexFlatIP(allowed_vectors.shape[1])
    search_index.add(allowed_vectors)

    query_vec = embedder.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = search_index.search(query_vec, min(top_k, len(allowed_indices)))
    return scores[0], indices[0]


def secure_faiss_retrieve(user_query: str, user_session: UserSession, top_k: int = 4, min_score: float = 0.30):
    allowed_indices = get_authorized_indices(user_session)
    scores, indices = search_faiss(user_query, allowed_indices, top_k)

    root = ET.Element("retrieved_chunks")
    max_score = float(scores[0]) if len(scores) > 0 else 0.0

    for score, idx in zip(scores, indices):
        if idx == -1 or score < min_score:
            continue
        chunk = KNOWLEDGE_CHUNKS[allowed_indices[idx]]
        item = ET.SubElement(
            root,
            "chunk",
            {
                "id": str(chunk["chunk_id"]),
                "classification": str(chunk["classification"]),
                "source": str(chunk["source"]),
                "period": str(chunk["reporting_period"]),
                "verification_status": str(chunk["verification_status"]),
            },
        )
        ET.SubElement(item, "text").text = chunk["content"]

    return ET.tostring(root, encoding="unicode"), max_score


def is_mastek_related(user_query: str, max_similarity_score: float) -> bool:
    mastek_keywords = [
        "mastek", "adopt ai", "openana", "project titan", "project alpha",
        "uk healthcare", "q3 fy25", "oracle cloud", "amea", "order backlog",
    ]
    query_lower = user_query.lower()
    if any(kw in query_lower for kw in mastek_keywords):
        return True
    return max_similarity_score >= 0.32


def ask_mastek_bi_assistant(user_query: str, user_session: UserSession, chat_history: List[dict]) -> str:
    context_xml, max_score = secure_faiss_retrieve(user_query, user_session, top_k=4)

    if not is_mastek_related(user_query, max_score):
        return REFUSAL_MESSAGE

    messages = [{"role": "system", "content": CLEAN_SYSTEM_PROMPT}]
    messages.extend(chat_history[-6:])

    session_xml = (
        f"<session_context>\n"
        f"  <authenticated_role>{escape_xml_attr(user_session.authenticated_role)}</authenticated_role>\n"
        f"  <authorized_classification>{escape_xml_attr(user_session.authorized_classification)}</authorized_classification>\n"
        f"  <rbac_verified>{str(user_session.rbac_verified).lower()}</rbac_verified>\n"
        f"</session_context>"
    )

    prompt = f"{session_xml}\n{context_xml}\n\nUSER QUERY: {user_query}"
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=4096,
    )

    reply = response.choices[0].message.content
    reply = re.sub(r'(\b\d+)[\s\u00a0\u202f]*mo\b', r'\1 months', reply, flags=re.IGNORECASE)
    return reply

# ------------------------------------------------------------------------------
# 6. STREAMLIT UI
# ------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("🛡️ RBAC & Session Controls")
    roles = [
        "CEO", "COO", "CFO", "CTO", "CMO",
        "Regional Presidents", "Vice Presidents",
        "Directors", "Associate Directors",
        "Associate Lead Managers", "Project Managers",
        "Technical Architects",
    ]
    selected_role = st.selectbox("Authenticated Role", roles, index=6)
    selected_class = st.selectbox(
        "Authorized Classification",
        ["PUBLIC_EXTERNAL", "CONFIDENTIAL_INTERNAL"],
        index=1,
    )
    rbac_verified = st.checkbox("RBAC Verified", value=True)

    st.divider()
    st.markdown(f"**Model:** `{MODEL}`")
    st.markdown(f"**Indexed Chunks:** `{len(KNOWLEDGE_CHUNKS)}`")
    st.markdown("**Scope:** Mastek Domain Only")
    st.divider()

    if st.button("🔄 Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

st.title("📊 Mastek Enterprise BI & Strategic Assistant")
st.caption(f"Active Session: **{selected_role}** | `{selected_class}` | `RBAC={rbac_verified}`")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a Mastek business intelligence, financial, or project governance question..."):
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    user_session = UserSession(
        authenticated_role=selected_role,
        authorized_classification=selected_class,
        rbac_verified=rbac_verified,
    )

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Mastek knowledge base..."):
            reply = ask_mastek_bi_assistant(
                user_query=user_query,
                user_session=user_session,
                chat_history=st.session_state.chat_history[:-1],
            )
            st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})