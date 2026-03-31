import io
import json
import os
from datetime import datetime

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer


st.set_page_config(page_title="Grok Multi-Agent UI", layout="wide")

DEFAULT_TASK = """Build a PDF creator tool.
Requirements:
- Python
- Accept input text or file
- Custom title
- Basic page formatting
- Save as PDF
- Return runnable final code
"""

if "prev_ids" not in st.session_state:
    st.session_state.prev_ids = {"planner": None, "builder": None, "critic": None}

if "outputs" not in st.session_state:
    st.session_state.outputs = {"planner": "", "builder": "", "critic": ""}

if "citations" not in st.session_state:
    st.session_state.citations = {"planner": [], "builder": [], "critic": []}


def get_client() -> OpenAI | None:
    api_key = st.session_state.get("xai_api_key") or os.getenv("XAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=3600,
    )


def extract_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts = []
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for content in getattr(item, "content", []) or []:
                ctype = getattr(content, "type", None)
                if ctype in ("output_text", "text"):
                    value = getattr(content, "text", None)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                elif isinstance(content, dict):
                    if content.get("type") in ("output_text", "text") and content.get("text"):
                        parts.append(str(content["text"]).strip())

    return "\n\n".join([p for p in parts if p]).strip()


def extract_citations(response):
    try:
        if hasattr(response, "model_dump"):
            dumped = response.model_dump()
            return dumped.get("citations", []) or []
    except Exception:
        pass
    return getattr(response, "citations", []) or []


def call_agent(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    use_web_search: bool,
    previous_response_id: str | None = None,
):
    payload = {
        "model": model,
        "store": True,
    }

    if use_web_search:
        payload["tools"] = [{"type": "web_search"}]

    if previous_response_id:
        payload["previous_response_id"] = previous_response_id
        payload["input"] = [{"role": "user", "content": user_prompt}]
    else:
        payload["input"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    return client.responses.create(**payload)


def build_pdf(title: str, planner: str, builder: str, critic: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=32, rightMargin=32, topMargin=32, bottomMargin=32)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["BodyText"]))
    story.append(Spacer(1, 16))

    sections = [
        ("Planner", planner),
        ("Builder", builder),
        ("Critic", critic),
    ]

    for name, content in sections:
        story.append(Paragraph(name, styles["Heading2"]))
        story.append(Spacer(1, 6))
        safe = content or "(empty)"
        story.append(Preformatted(safe, styles["Code"]))
        story.append(Spacer(1, 14))

    doc.build(story)
    buf.seek(0)
    return buf.read()


PLANNER_SYSTEM = """You are Planner.
Return a sharp execution plan for the user's task.
Priorities:
- concrete architecture
- exact implementation steps
- note risks and assumptions
- optimize for a local runnable project
- when web search is available, use it if current docs/specs matter
"""

BUILDER_SYSTEM = """You are Builder.
Turn the task and planner output into the strongest practical implementation.
Priorities:
- complete runnable result
- direct answer before commentary
- if code is needed, return full code
- avoid legacy interfaces when a current recommended interface exists
"""

CRITIC_SYSTEM = """You are Critic.
Review the planner and builder outputs.
Return:
1) important gaps
2) important risks
3) a tightened final version
Be concrete and improve the result, not just comment on it.
"""

st.title("Grok Multi-Agent UI")
st.caption("Responses API-first • Planner / Builder / Critic • PDF export")

with st.sidebar:
    st.subheader("xAI")
    api_key_input = st.text_input(
        "XAI_API_KEY",
        value=os.getenv("XAI_API_KEY", ""),
        type="password",
        placeholder="Paste your xAI API key",
    )
    if api_key_input:
        st.session_state.xai_api_key = api_key_input

    st.subheader("Models")
    planner_model = st.text_input("Planner model", value="grok-4.20-multi-agent")
    builder_model = st.text_input("Builder model", value="grok-4.20-reasoning")
    critic_model = st.text_input("Critic model", value="grok-4.20-reasoning")

    st.subheader("Tools")
    planner_web = st.checkbox("Planner web_search", value=True)
    builder_web = st.checkbox("Builder web_search", value=False)
    critic_web = st.checkbox("Critic web_search", value=False)

    if st.button("Clear stored response IDs", use_container_width=True):
        st.session_state.prev_ids = {"planner": None, "builder": None, "critic": None}
        st.session_state.outputs = {"planner": "", "builder": "", "critic": ""}
        st.session_state.citations = {"planner": [], "builder": [], "critic": []}
        st.success("Cleared.")

client = get_client()
if not client:
    st.warning("Set XAI_API_KEY in the sidebar or environment to run.")
    st.stop()

task = st.text_area("Task", value=DEFAULT_TASK, height=180)
context = st.text_area(
    "Project context",
    value="Local Grok multi-agent UI project. Use Responses API first. Planner/Builder/Critic lanes. Prefer current xAI recommended APIs and tools.",
    height=120,
)
follow_up = st.text_input("Follow-up for current thread (optional)")

c1, c2 = st.columns(2)
run_fresh = c1.button("Run fresh", use_container_width=True)
run_continue = c2.button("Continue current thread", use_container_width=True)

if run_fresh:
    with st.spinner("Running Planner / Builder / Critic..."):
        planner_prompt = f"""Task:
{task}

Project context:
{context}

Return:
- goal
- architecture
- execution steps
- risks
- next implementation target
"""
        planner_resp = call_agent(
            client=client,
            model=planner_model,
            system_prompt=PLANNER_SYSTEM,
            user_prompt=planner_prompt,
            use_web_search=planner_web,
        )
        planner_text = extract_text(planner_resp)

        builder_prompt = f"""Task:
{task}

Project context:
{context}

Planner output:
{planner_text}

Produce the best practical final implementation for this task.
If code is needed, return complete runnable code.
"""
        builder_resp = call_agent(
            client=client,
            model=builder_model,
            system_prompt=BUILDER_SYSTEM,
            user_prompt=builder_prompt,
            use_web_search=builder_web,
        )
        builder_text = extract_text(builder_resp)

        critic_prompt = f"""Task:
{task}

Project context:
{context}

Planner output:
{planner_text}

Builder output:
{builder_text}

Improve the work and return the corrected final version.
"""
        critic_resp = call_agent(
            client=client,
            model=critic_model,
            system_prompt=CRITIC_SYSTEM,
            user_prompt=critic_prompt,
            use_web_search=critic_web,
        )
        critic_text = extract_text(critic_resp)

        st.session_state.prev_ids = {
            "planner": planner_resp.id,
            "builder": builder_resp.id,
            "critic": critic_resp.id,
        }
        st.session_state.outputs = {
            "planner": planner_text,
            "builder": builder_text,
            "critic": critic_text,
        }
        st.session_state.citations = {
            "planner": extract_citations(planner_resp),
            "builder": extract_citations(builder_resp),
            "critic": extract_citations(critic_resp),
        }

if run_continue:
    missing = [k for k, v in st.session_state.prev_ids.items() if not v]
    if missing:
        st.error("Run fresh first so response IDs exist.")
    elif not follow_up.strip():
        st.error("Enter a follow-up message first.")
    else:
        with st.spinner("Continuing stored threads..."):
            planner_resp = call_agent(
                client=client,
                model=planner_model,
                system_prompt=PLANNER_SYSTEM,
                user_prompt=follow_up,
                use_web_search=planner_web,
                previous_response_id=st.session_state.prev_ids["planner"],
            )
            planner_text = extract_text(planner_resp)

            builder_follow = f"""Follow-up request:
{follow_up}

Latest planner update:
{planner_text}

Update the implementation accordingly.
"""
            builder_resp = call_agent(
                client=client,
                model=builder_model,
                system_prompt=BUILDER_SYSTEM,
                user_prompt=builder_follow,
                use_web_search=builder_web,
                previous_response_id=st.session_state.prev_ids["builder"],
            )
            builder_text = extract_text(builder_resp)

            critic_follow = f"""Follow-up request:
{follow_up}

Latest planner update:
{planner_text}

Latest builder update:
{builder_text}

Tighten the result again and return the corrected final version.
"""
            critic_resp = call_agent(
                client=client,
                model=critic_model,
                system_prompt=CRITIC_SYSTEM,
                user_prompt=critic_follow,
                use_web_search=critic_web,
                previous_response_id=st.session_state.prev_ids["critic"],
            )
            critic_text = extract_text(critic_resp)

            st.session_state.prev_ids = {
                "planner": planner_resp.id,
                "builder": builder_resp.id,
                "critic": critic_resp.id,
            }
            st.session_state.outputs = {
                "planner": planner_text,
                "builder": builder_text,
                "critic": critic_text,
            }
            st.session_state.citations = {
                "planner": extract_citations(planner_resp),
                "builder": extract_citations(builder_resp),
                "critic": extract_citations(critic_resp),
            }

planner_out = st.session_state.outputs["planner"]
builder_out = st.session_state.outputs["builder"]
critic_out = st.session_state.outputs["critic"]

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Planner")
    st.markdown(planner_out or "_No output yet._")
    if st.session_state.citations["planner"]:
        with st.expander("Planner citations"):
            st.code(json.dumps(st.session_state.citations["planner"], indent=2, default=str), language="json")

with col2:
    st.subheader("Builder")
    st.markdown(builder_out or "_No output yet._")
    if st.session_state.citations["builder"]:
        with st.expander("Builder citations"):
            st.code(json.dumps(st.session_state.citations["builder"], indent=2, default=str), language="json")

with col3:
    st.subheader("Critic")
    st.markdown(critic_out or "_No output yet._")
    if st.session_state.citations["critic"]:
        with st.expander("Critic citations"):
            st.code(json.dumps(st.session_state.citations["critic"], indent=2, default=str), language="json")

if planner_out or builder_out or critic_out:
    pdf_bytes = build_pdf(
        title="Grok Multi-Agent Output",
        planner=planner_out,
        builder=builder_out,
        critic=critic_out,
    )
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"grok_multi_agent_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
