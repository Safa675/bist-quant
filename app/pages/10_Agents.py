"""Agents — Placeholder for future LLM-agent integration with session log display."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

st.set_page_config(
    page_title="Agents · BIST Quant", page_icon="🤖", layout="wide"
)

from app.layout import page_header, render_sidebar  # noqa: E402

render_sidebar()
page_header(
    "🤖 Agents",
    "LLM-powered research agents — coming soon",
)

# ===========================================================================
# Constants
# ===========================================================================

_PLANNED_AGENTS: list[dict[str, str]] = [
    {
        "name": "Market Research Agent",
        "icon": "🔎",
        "description": (
            "Autonomously gathers macro & sector intelligence, summarises news sentiment "
            "and cross-references with regime state to surface actionable insights."
        ),
        "status": "planned",
        "tags": "LLM · Sentiment · Macro",
    },
    {
        "name": "Backtesting Copilot",
        "icon": "📈",
        "description": (
            "Interprets backtest results in plain language, suggests signal parameter "
            "improvements and generates draft strategy code from natural-language descriptions."
        ),
        "status": "planned",
        "tags": "LLM · Backtest · Code-gen",
    },
    {
        "name": "Risk Monitor Agent",
        "icon": "🛡️",
        "description": (
            "Continuously watches portfolio risk metrics, escalates breaches via "
            "configurable channels and explains root causes in natural language."
        ),
        "status": "planned",
        "tags": "LLM · Risk · Alerts",
    },
    {
        "name": "Compliance Reviewer",
        "icon": "⚖️",
        "description": (
            "Reviews transaction batches against regulatory rule sets, drafts exception "
            "memos and maintains an audit trail with natural-language summaries."
        ),
        "status": "planned",
        "tags": "LLM · Compliance · Reporting",
    },
    {
        "name": "Fundamental Analyst",
        "icon": "📊",
        "description": (
            "Parses earnings reports, balance sheets and guidance calls, enriches them "
            "with sector context and outputs structured investment notes."
        ),
        "status": "planned",
        "tags": "LLM · Fundamentals · NLP",
    },
    {
        "name": "Execution Optimizer",
        "icon": "⚡",
        "description": (
            "Selects optimal TWAP/VWAP/Iceberg schedule for large orders, adapting in "
            "real-time to order-book liquidity and intraday regime signals."
        ),
        "status": "planned",
        "tags": "LLM · Execution · Microstructure",
    },
]

# ===========================================================================
# Agent cards
# ===========================================================================

st.markdown(
    """
    <div style="
        background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
        border:1px solid #30305a;
        border-radius:12px;
        padding:1.2rem 1.6rem;
        margin-bottom:1.5rem;
    ">
        <h3 style="margin:0 0 0.4rem 0;">🚧 &nbsp; LLM Agent Integration — Under Development</h3>
        <p style="margin:0;color:#aaa;font-size:0.9rem;">
            This page is a live placeholder that will host autonomous LLM-powered research agents.
            The session log below already captures simulated agent activity and will connect to
            real orchestration backends (LangChain / LangGraph / OpenAI Assistants API) in a
            future release.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(3)
for idx, agent in enumerate(_PLANNED_AGENTS):
    with cols[idx % 3]:
        st.markdown(
            f"""
            <div style="
                border:1px solid #2a2a4a;
                border-radius:10px;
                padding:1rem 1.1rem 1.1rem;
                margin-bottom:1rem;
                background:#12122a;
            ">
                <div style="font-size:1.8rem;margin-bottom:0.25rem;">{agent['icon']}</div>
                <div style="font-weight:700;font-size:1rem;margin-bottom:0.4rem;">{agent['name']}</div>
                <div style="color:#aaa;font-size:0.82rem;line-height:1.45;margin-bottom:0.6rem;">
                    {agent['description']}
                </div>
                <div style="font-size:0.72rem;color:#555;letter-spacing:0.5px;">{agent['tags']}</div>
                <div style="
                    display:inline-block;
                    margin-top:0.6rem;
                    background:#2a2a4a;
                    color:#9b9bff;
                    border-radius:6px;
                    padding:2px 8px;
                    font-size:0.72rem;
                    font-weight:600;
                    letter-spacing:0.8px;
                ">PLANNED</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===========================================================================
# Session log
# ===========================================================================
st.divider()
st.subheader("Agent Session Log")
st.caption(
    "Records all agent interactions in this session. "
    "Messages sent via the prompt box are acknowledged by a stub agent. "
    "Replace the stub handler with a real LLM call to activate the agent."
)

# ── session state ──────────────────────────────────────────────────────────
if "agent_log" not in st.session_state:
    st.session_state.agent_log: list[dict[str, Any]] = [
        {
            "id": str(uuid.uuid4())[:8],
            "timestamp": "2026-02-21T09:00:00Z",
            "role": "system",
            "agent": "System",
            "content": "Agent framework initialised. No active agents. Stub mode enabled.",
        }
    ]

if "agent_running" not in st.session_state:
    st.session_state.agent_running = False

# ── agent selector ─────────────────────────────────────────────────────────
selected_agent = st.selectbox(
    "Target Agent",
    [a["name"] for a in _PLANNED_AGENTS],
    help="Select the agent that will handle your prompt (all are stubs until implemented).",
)

# ── prompt input ───────────────────────────────────────────────────────────
with st.form("agent_prompt_form", clear_on_submit=True):
    prompt = st.text_area(
        "Prompt",
        placeholder=(
            "e.g. 'Summarise today's BIST macro regime and suggest top-3 momentum tickers' "
            "or 'Check my last backtest for overfitting signals'"
        ),
        height=100,
    )
    col_submit, col_clear = st.columns([2, 8])
    send = col_submit.form_submit_button("▶ Send", type="primary", use_container_width=True)

if send and prompt.strip():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # User message
    st.session_state.agent_log.append(
        {
            "id": str(uuid.uuid4())[:8],
            "timestamp": now,
            "role": "user",
            "agent": "User",
            "content": prompt.strip(),
        }
    )

    # Stub agent response
    stub_reply = (
        f"[STUB — {selected_agent}] I received your prompt and am ready to process it. "
        "This response will be replaced by a real LLM call once the agent backend is wired up. "
        f"Prompt preview: \"{prompt.strip()[:80]}{'...' if len(prompt.strip()) > 80 else ''}\""
    )
    st.session_state.agent_log.append(
        {
            "id": str(uuid.uuid4())[:8],
            "timestamp": now,
            "role": "agent",
            "agent": selected_agent,
            "content": stub_reply,
        }
    )
    st.rerun()

# ── render log ─────────────────────────────────────────────────────────────
log = st.session_state.agent_log

# show newest-first toggle
show_newest_first = st.checkbox("Newest first", value=True)
ordered_log = list(reversed(log)) if show_newest_first else log

for entry in ordered_log:
    role = entry["role"]
    if role == "system":
        st.markdown(
            f"<div style='color:#555;font-size:0.78rem;padding:2px 0;'>"
            f"⚙ <b>{entry['timestamp']}</b> — {entry['content']}"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif role == "user":
        with st.chat_message("user"):
            st.markdown(f"**{entry['timestamp']}**  \n{entry['content']}")
    else:
        with st.chat_message("assistant"):
            icon = next(
                (a["icon"] for a in _PLANNED_AGENTS if a["name"] == entry.get("agent")),
                "🤖",
            )
            st.markdown(
                f"**{icon} {entry['agent']}** — {entry['timestamp']}  \n{entry['content']}"
            )

# ── log management ─────────────────────────────────────────────────────────
st.divider()
col_export, col_clear_log = st.columns([2, 8])

with col_export:
    import json

    log_json = json.dumps(log, indent=2, ensure_ascii=False)
    st.download_button(
        "⬇ Export Log (JSON)",
        data=log_json,
        file_name=f"agent_session_log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

with col_clear_log:
    if st.button("🗑 Clear Session Log"):
        st.session_state.agent_log = [
            {
                "id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "role": "system",
                "agent": "System",
                "content": "Session log cleared. Agent framework ready.",
            }
        ]
        st.rerun()

# ===========================================================================
# Integration guide
# ===========================================================================
st.divider()
with st.expander("🔧 Integration Guide — How to activate a real agent"):
    st.markdown(
        """
### Connecting a Real LLM Agent

Replace the stub handler in this file with a real LLM call.  The session log
wiring is already in place — you only need to swap the `stub_reply` block.

#### Option A — OpenAI Assistants API
```python
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def call_agent(prompt: str, agent_name: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are {agent_name}, a professional quantitative research agent."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
```

#### Option B — LangChain / LangGraph
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

def call_agent(prompt: str, agent_name: str) -> str:
    return llm.invoke([HumanMessage(content=prompt)]).content
```

#### Option C — Local model via Ollama
```python
import requests

def call_agent(prompt: str, agent_name: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False},
    )
    return r.json()["response"]
```

#### Secrets / `.streamlit/secrets.toml`
```toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."
```

Once `call_agent` is defined, replace the stub block in `10_Agents.py`:
```python
# Before (stub):
stub_reply = f"[STUB — {selected_agent}] ..."

# After (live):
stub_reply = call_agent(prompt.strip(), selected_agent)
```
        """
    )
