import os
import re
import secrets
import faiss
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, session
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

print("Starting Mastek Enterprise BI Assistant...", flush=True)

# ------------------------------------------------------------------------------
# 1. ENVIRONMENT & FLASK SETUP
# ------------------------------------------------------------------------------
PROJECT_PATH = Path(__file__).resolve().parent
ENV_FILE = PROJECT_PATH / ".env"

if not ENV_FILE.exists():
    ENV_FILE.write_text('GROQ_API_KEY=""\n', encoding="utf-8")

load_dotenv(ENV_FILE, override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    raise ValueError(f"Missing GROQ_API_KEY. Please add your API key to {ENV_FILE}")

MODEL = "openai/gpt-oss-120b"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

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
        "category": "Financial Performance",
        "document_type": "Investor Presentation",
        "source": "Mastek Q3 FY25 Investor Presentation",
        "source_url": "https://www.mastek.com/investors",
        "reporting_period": "Q3 FY25",
        "period_end_date": "2024-12-31",
        "publication_date": None,
        "tier": "Tier 2",
        "keywords": ["Q3 FY25", "revenue", "YoY growth", "order backlog", "EBITDA", "financial performance"],
        "metrics": {
            "operating_revenue_crore": 869.5,
            "revenue_yoy_growth_percent": 11.0,
            "twelve_month_order_backlog_crore": 2145.8,
            "operating_ebitda_margin_percent": 16.2,
        },
        "units": "INR crore and percentage",
        "access_level": "PUBLIC",
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
        "category": "Corporate Strategy",
        "document_type": "Annual Report",
        "source": "Mastek Annual Report FY25",
        "source_url": "https://www.mastek.com/",
        "reporting_period": "FY24-25",
        "period_end_date": None,
        "publication_date": None,
        "tier": "Tier 1",
        "keywords": [
            "AI-led transformation", "Oracle Cloud", "Data and AI", "ADOPT AI",
            "OpenAna", "Digital Engineering", "Cloud Engineering", "UK", "North America", "AMEA"
        ],
        "metrics": {},
        "units": "Not applicable",
        "access_level": "PUBLIC",
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
        "category": "Project Delivery Governance",
        "document_type": "PMO RAID Register",
        "source": "Internal PMO Delivery Governance & RAID Register",
        "source_url": None,
        "reporting_period": "November 2026",
        "period_end_date": None,
        "publication_date": None,
        "sprint": 14,
        "tier": "Internal Governance",
        "keywords": [
            "Project Titan", "Oracle Cloud ERP", "milestone M4", "project delay",
            "budget burn rate", "RAID R-12", "UAT sign-off", "client security", "resource attrition"
        ],
        "metrics": {
            "schedule_delay_weeks": 3,
            "budget_burn_rate_percent": 108.0,
            "resource_attrition_percent": 0.0,
        },
        "units": "Weeks and percentage",
        "access_level": "RESTRICTED",
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
        "category": "Financial and Resource Performance",
        "document_type": "Internal BU P&L Dashboard",
        "source": "Internal BU P&L & Utilisation Telemetry Dashboard",
        "source_url": None,
        "reporting_period": "October 2026 MTD",
        "period_end_date": None,
        "publication_date": None,
        "tier": "Internal Financial",
        "keywords": [
            "UK Healthcare", "billable utilisation", "senior architects",
            "bench resources", "Project Alpha", "gross margin", "margin target"
        ],
        "metrics": {
            "previous_billable_utilisation_percent": 84.5,
            "current_billable_utilisation_percent": 79.1,
            "utilisation_change_percentage_points": -5.4,
            "architects_transitioned_to_bench": 14,
            "actual_gross_margin_percent": 24.3,
            "target_gross_margin_percent": 27.0,
            "gross_margin_gap_percentage_points": -2.7,
        },
        "units": "Percentage, percentage points and employees",
        "access_level": "RESTRICTED",
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
        "category": "Business Solutions & AI Use Cases",
        "document_type": "Solution Architecture Brief",
        "source": "Mastek Enterprise AI & Services-as-Software Brief",
        "source_url": "https://www.mastek.com/industries/",
        "reporting_period": "FY26",
        "period_end_date": "2026-10-05",
        "publication_date": "2026-10-05",
        "tier": "Tier 3",
        "keywords": [
            "business solutions", "AI use cases", "supply chain", "finance",
            "HR", "legacy modernization", "automation", "ROI"
        ],
        "metrics": {
            "manual_effort_reduction_percent": 45.0,
            "fraud_detection_improvement_percent": 15.0,
            "sdlc_efficiency_gain_percent": 35.0,
        },
        "units": "Percentage",
        "access_level": "PUBLIC",
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
# 4. INITIALIZE GROQ & EMBEDDINGS ONCE AT STARTUP
# ------------------------------------------------------------------------------
print("Loading embedding model and indexing knowledge chunks...", flush=True)
client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

knowledge_text = [f"{chunk['entity']} | {chunk['content']}" for chunk in KNOWLEDGE_CHUNKS]
embeddings = embedder.encode(knowledge_text, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
print(f"Indexed {len(KNOWLEDGE_CHUNKS)} Mastek knowledge chunks successfully.", flush=True)

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

    scores, indices = search_index.search(
        query_vec, min(top_k, len(allowed_indices))
    )
    return scores[0], indices[0]


def secure_faiss_retrieve(
    user_query: str, user_session: UserSession, top_k: int = 3, min_score: float = 0.30
):
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


def ask_mastek_bi_assistant(
    user_query: str,
    user_session: UserSession,
    chat_history: List[dict],
    top_k: int = 4,
) -> str:
    context_xml, max_score = secure_faiss_retrieve(user_query, user_session, top_k=top_k)

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
    # Guarantee "months" instead of "mo" in tables and prose (handles regular & non-breaking spaces)
    reply = re.sub(r'(\b\d+)[\s\u00a0\u202f]*mo\b', r'\1 months', reply, flags=re.IGNORECASE)

    return reply


# ------------------------------------------------------------------------------
# 6. EMBEDDED HTML TEMPLATE
# ------------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mastek Enterprise BI Assistant</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    :root {
      --bg: #ffffff;
      --panel: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #1e3a8a;         /* Corporate Navy for Inside Highlights */
      --accent-hover: #4338ca;   /* Indigo Blue for Focus & Accents */
      --accent-soft: #eef2ff;    /* Soft Indigo Tint */
      --accent-border: #c7d2fe;
      --user-bubble: #1e40af;    /* Deep Royal Blue User Bubble */
      --user-text: #ffffff;
      --bot-bubble: #ffffff;
      --table-header: #eef2ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #ffffff;
      color: var(--text);
      display: flex;
      height: 100vh;
    }
    .sidebar {
      width: 300px;
      background: #ffffff;
      border-right: 1px solid var(--border);
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .sidebar h2 {
      margin: 0;
      font-size: 1.05rem;
      color: var(--accent);
      font-weight: 700;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 9px;
    }
    .icon-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 30px;
      height: 30px;
      border-radius: 8px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid var(--accent-border);
      flex-shrink: 0;
    }
    .header-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 34px;
      border-radius: 9px;
      background: linear-gradient(135deg, #1e3a8a 0%, #4338ca 100%);
      color: #ffffff;
      flex-shrink: 0;
    }
    label {
      display: block;
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 6px;
    }
    select, button {
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: var(--text);
      font-size: 0.9rem;
      transition: all 0.2s ease;
    }
    select:focus {
      outline: none;
      border-color: var(--accent-hover);
      box-shadow: 0 0 0 3px rgba(67, 56, 202, 0.14);
    }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;
      padding: 4px 0;
    }
    .checkbox-row input {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
      cursor: pointer;
    }
    /* Clear Conversation Button: Warm Amber Reset Style */
    .btn-clear {
      background: #fffbeb;
      color: #b45309;
      border: 1px solid #fcd34d;
      cursor: pointer;
      font-weight: 600;
      margin-top: auto;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .btn-clear:hover {
      background: #fef3c7;
      border-color: #f59e0b;
      color: #92400e;
    }
    .meta-info {
      font-size: 0.8rem;
      color: var(--muted);
      border-top: 1px solid var(--border);
      padding-top: 14px;
      line-height: 1.7;
    }
    .meta-info strong {
      color: var(--text);
    }
    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      background: #ffffff;
    }
    .header {
      padding: 18px 28px;
      border-bottom: 1px solid var(--border);
      background: #ffffff;
    }
    .header h1 {
      margin: 0;
      font-size: 1.25rem;
      color: var(--text);
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 11px;
    }
    .header p {
      margin: 6px 0 0 45px;
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 500;
    }
    .chat-box {
      flex: 1;
      padding: 24px 28px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      background: #ffffff;
    }
    .message {
      max-width: 82%;
      padding: 16px 20px;
      border-radius: 12px;
      line-height: 1.6;
      font-size: 0.95rem;
    }
    .message.user {
      align-self: flex-end;
      background: var(--user-bubble);
      color: var(--user-text);
      box-shadow: 0 2px 6px rgba(30, 64, 175, 0.18);
    }
    .message.assistant {
      align-self: flex-start;
      background: #ffffff;
      color: var(--text);
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent-hover);
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .message.assistant h1, .message.assistant h2, .message.assistant h3 {
      color: var(--accent);
      margin-top: 16px;
      margin-bottom: 8px;
    }
    .message.assistant table {
      border-collapse: collapse;
      width: 100%;
      margin: 14px 0;
      font-size: 0.9rem;
    }
    .message.assistant th, .message.assistant td {
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
    }
    .message.assistant th {
      background: var(--table-header);
      color: var(--accent);
      font-weight: 600;
    }
    .input-area {
      padding: 16px 28px;
      border-top: 1px solid var(--border);
      background: #ffffff;
      display: flex;
      gap: 12px;
      align-items: center;
    }
    textarea {
      flex: 1;
      resize: none;
      height: 52px;
      padding: 13px 16px;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: var(--text);
      font-family: inherit;
      font-size: 0.95rem;
      transition: all 0.2s ease;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent-hover);
      box-shadow: 0 0 0 3px rgba(67, 56, 202, 0.14);
    }
    /* Run Analysis Button: Emerald Teal Gradient (Distinct from Navy/Indigo Highlights) */
    .btn-send {
      width: 158px;
      height: 52px;
      background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
      color: #ffffff;
      border: none;
      border-radius: 10px;
      font-weight: 600;
      letter-spacing: 0.01em;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 2px 6px rgba(15, 118, 110, 0.25);
    }
    .btn-send:hover {
      background: linear-gradient(135deg, #115e59 0%, #0d9488 100%);
      transform: translateY(-1px);
      box-shadow: 0 4px 10px rgba(15, 118, 110, 0.32);
    }
    .btn-send:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      transform: none;
    }
  </style>
</head>
<body>
  <div class="sidebar">
    <h2>
      <span class="icon-badge">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="m9 12 2 2 4-4"/>
        </svg>
      </span>
      RBAC & Session Controls
    </h2>
    <div>
      <label for="role">Authenticated Role</label>
      <select id="role">
        <option>CEO</option>
        <option>COO</option>
        <option>CFO</option>
        <option>CTO</option>
        <option>CMO</option>
        <option>Regional Presidents</option>
        <option selected>Vice Presidents</option>
        <option>Directors</option>
        <option>Associate Directors</option>
        <option>Associate Lead Managers</option>
        <option>Project Managers</option>
        <option>Technical Architects</option>
      </select>
    </div>

    <div>
      <label for="classification">Authorized Classification</label>
      <select id="classification">
        <option value="PUBLIC_EXTERNAL">PUBLIC_EXTERNAL</option>
        <option value="CONFIDENTIAL_INTERNAL" selected>CONFIDENTIAL_INTERNAL</option>
      </select>
    </div>

    <div class="checkbox-row">
      <input type="checkbox" id="rbac" checked />
      <label for="rbac" style="margin:0; color:var(--text);">RBAC Verified</label>
    </div>

    <div class="meta-info">
      <div><strong>Model:</strong> {{ model }}</div>
      <div><strong>Indexed Chunks:</strong> {{ chunk_count }}</div>
      <div><strong>Scope:</strong> Mastek Domain Only</div>
    </div>

    <button class="btn-clear" onclick="clearChat()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
        <path d="M3 3v5h5"/>
      </svg>
      Clear Conversation
    </button>
  </div>

  <div class="main">
    <div class="header">
      <h1>
        <span class="header-badge">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="20" x2="18" y2="10"/>
            <line x1="12" y1="20" x2="12" y2="4"/>
            <line x1="6" y1="20" x2="6" y2="14"/>
          </svg>
        </span>
        Mastek Enterprise BI & Strategic Assistant
      </h1>
      <p id="statusBar">Active Session: Vice Presidents | CONFIDENTIAL_INTERNAL | RBAC=true</p>
    </div>

    <div class="chat-box" id="chatBox"></div>

    <div class="input-area">
      <textarea id="queryInput" placeholder="Ask a Mastek business intelligence, financial, or project governance question..."></textarea>
      <button class="btn-send" id="sendBtn" onclick="sendQuery()">
        <span>Run Analysis</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"/>
          <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
  </div>

  <script>
    const roleEl = document.getElementById("role");
    const classEl = document.getElementById("classification");
    const rbacEl = document.getElementById("rbac");
    const statusEl = document.getElementById("statusBar");
    const chatBox = document.getElementById("chatBox");
    const queryInput = document.getElementById("queryInput");
    const sendBtn = document.getElementById("sendBtn");

    function updateStatus() {
      statusEl.textContent = `Active Session: ${roleEl.value} | ${classEl.value} | RBAC=${rbacEl.checked}`;
    }
    roleEl.addEventListener("change", updateStatus);
    classEl.addEventListener("change", updateStatus);
    rbacEl.addEventListener("change", updateStatus);

    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuery();
      }
    });

    function appendMessage(role, content) {
      const div = document.createElement("div");
      div.className = `message ${role}`;
      if (role === "assistant") {
        div.innerHTML = marked.parse(content);
      } else {
        div.textContent = content;
      }
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
      return div;
    }

    async function sendQuery() {
      const query = queryInput.value.trim();
      if (!query) return;

      queryInput.value = "";
      sendBtn.disabled = true;
      appendMessage("user", query);
      const loadingDiv = appendMessage("assistant", "_Analyzing Mastek knowledge base..._");

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query,
            authenticated_role: roleEl.value,
            authorized_classification: classEl.value,
            rbac_verified: rbacEl.checked
          })
        });
        const data = await res.json();
        if (data.error) {
          loadingDiv.innerHTML = `<span style="color:#dc2626; font-weight:600;">Error: ${data.error}</span>`;
        } else {
          loadingDiv.innerHTML = marked.parse(data.reply);
        }
      } catch (err) {
        loadingDiv.innerHTML = `<span style="color:#dc2626; font-weight:600;">Network Error: ${err.message}</span>`;
      } finally {
        sendBtn.disabled = false;
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    }

    async function clearChat() {
      await fetch("/clear", { method: "POST" });
      chatBox.innerHTML = "";
    }
  </script>
</body>
</html>
"""

# ------------------------------------------------------------------------------
# 7. FLASK ROUTES
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template_string(
        HTML_TEMPLATE,
        model=MODEL,
        chunk_count=len(KNOWLEDGE_CHUNKS),
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_query = (data.get("query") or "").strip()
    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        user_session = UserSession(
            authenticated_role=data.get("authenticated_role", "Project Managers"),
            authorized_classification=data.get("authorized_classification", "PUBLIC_EXTERNAL"),
            rbac_verified=bool(data.get("rbac_verified", False)),
        )

        chat_history = session.get("chat_history", [])
        reply = ask_mastek_bi_assistant(
            user_query=user_query,
            user_session=user_session,
            chat_history=chat_history,
        )

        chat_history.append({"role": "user", "content": user_query})
        chat_history.append({"role": "assistant", "content": reply})
        session["chat_history"] = chat_history[-10:]

        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    session["chat_history"] = []
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)