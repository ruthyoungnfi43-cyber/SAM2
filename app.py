import io
import os
import time
from datetime import datetime

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

st.set_page_config(page_title="Grok Multi-Agent SaaS", layout="wide")

APP_TITLE = "Grok Multi-Agent SaaS"
APP_CAPTION = "Secrets-only xAI key • session memory • rate limiting • PDF export"

CASUAL_PHRASES = [
    "hi", "hello", "hey", "yo", "sup", "what's up", "whats up",
    "how are you", "how are u", "how r u", "who are you", "who r u",
    "good morning", "good evening", "thanks", "thank you"
]

if "prev_ids" not in st.session_state:
    st.session_state.prev_ids = {"planner": None, "builder": None, "critic": None}

if "outputs" not in st.session_state:
    st.session_state.outputs = {"planner": "", "builder": "", "critic": ""}

if "history" not in st.session_state:
    st.session_state.history = []

if "run_times" not in st.session_state:
    st.session_state.run_times = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def get_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def require_login():
    app_password = get_secret("APP_PASSWORD", "")
    if not app_password:
        st.session_state.logged_in = True
        return

    if st.session_state.logged_in:
        return

    st.title(APP_TITLE)
    st.caption("Private access")
    entered = st.text_input("Enter access password", type="password")
    if st.button("Unlock", use_container_width=True):
        if entered == app_password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


def get_client() -> OpenAI | None:
    api_key = get_secret("XAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=60,
    )


def is_casual(text: str, context: str = "") -> bool:
    combined = f"{text} {context}".strip().lower()
    return any(p in combined for p in CASUAL_PHRASES) and len(combined) <= 140


def enforce_rate_limit():
    now = time.time()
    window_seconds = 60
    max_runs = 8
    st.session_state.run_times = [t for t in st.session_state.run_times if now - t < window_seconds]
    if len(st.session_state.run_times) >= max_runs:
        st.error("Rate limit hit. Please wait a minute and try again.")
        st.stop()
    st.session_state.run_times.append(now)


def extract_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
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


def call_agent(client, model, system_prompt, user_prompt, use_web_search=False, previous_response_id=None):
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

    for name, content in [("Planner", planner), ("Builder", builder), ("Critic", critic)]:
        story.append(Paragraph(name, styles["Heading2"]))
        story.append(Spacer(1, 6))
        story.append(Preformatted(content or "(empty)", styles["Code"]))
        story.append(Spacer(1, 14))

    doc.build(story)
    buf.seek(0)
    return buf.read()


PLANNER_SYSTEM = """You are Planner.
Detect intent first.
- Casual/social input: do not create a project plan. Return one short intent note only.
- Real task/request: return a concise plan with goal, architecture, steps, and risks.
Stay practical.
"""

BUILDER_SYSTEM = """You are Builder.
- Casual/social input: reply naturally and briefly.
- Real task/request: produce the strongest practical result.
No over-engineering.
"""

CRITIC_SYSTEM = """You are Critic.
- Casual/social input: briefly confirm whether the reply is natural and appropriate.
- Real task/request: identify gaps/risks and tighten the final version.
"""

require_login()
client = get_client()

if not client:
    st.error("Missing XAI_API_KEY in Streamlit Secrets / environment.")
    st.stop()

st.title(APP_TITLE)
st.caption(APP_CAPTION)

with st.sidebar:
    st.subheader("Status")
    st.success("xAI key loaded from Secrets")
    if get_secret("APP_PASSWORD", ""):
        st.info("Password gate enabled")
    else:
        st.warning("Password gate disabled")

    st.subheader("Models")
    planner_model = st.text_input("Planner model", value="grok-4.20-multi-agent")
    builder_model = st.text_input("Builder model", value="grok-4.20-reasoning")
    critic_model = st.text_input("Critic model", value="grok-4.20-reasoning")

    st.subheader("Tools")
    planner_web = st.checkbox("Planner web_search", value=False)
    builder_web = st.checkbox("Builder web_search", value=False)
    critic_web = st.checkbox("Critic web_search", value=False)

    if st.button("Clear session", use_container_width=True):
        st.session_state.prev_ids = {"planner": None, "builder": None, "critic": None}
        st.session_state.outputs = {"planner": "", "builder": "", "critic": ""}
        st.session_state.history = []
        st.success("Session cleared.")
        st.rerun()

task = st.text_area("Task", value="", height=180, placeholder="Ask a task, question, or casual prompt...")
context = st.text_area("Project context", value="", height=120, placeholder="Optional extra context...")
follow_up = st.text_input("Follow-up for current thread (optional)")

c1, c2 = st.columns(2)
run_fresh = c1.button("Run fresh", use_container_width=True)
run_continue = c2.button("Continue current thread", use_container_width=True)

if run_fresh:
    enforce_rate_limit()
    casual = is_casual(task, context)

    if casual:
        planner_text = "Casual conversation detected. Use a short natural reply."
        builder_text = "Hey! I'm doing great 😊 How about you?"
        critic_text = "Looks natural and appropriate for a casual greeting."
        st.session_state.prev_ids = {"planner": None, "builder": None, "critic": None}
        st.session_state.outputs = {
            "planner": planner_text,
            "builder": builder_text,
            "critic": critic_text,
        }
    else:
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

Produce the best practical final result.
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

Tighten the result.
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

    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "context": context,
        "planner": st.session_state.outputs["planner"],
        "builder": st.session_state.outputs["builder"],
        "critic": st.session_state.outputs["critic"],
    })

if run_continue:
    enforce_rate_limit()
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

            builder_resp = call_agent(
                client=client,
                model=builder_model,
                system_prompt=BUILDER_SYSTEM,
                user_prompt=f"Follow-up request:\n{follow_up}\n\nLatest planner update:\n{planner_text}",
                use_web_search=builder_web,
                previous_response_id=st.session_state.prev_ids["builder"],
            )
            builder_text = extract_text(builder_resp)

            critic_resp = call_agent(
                client=client,
                model=critic_model,
                system_prompt=CRITIC_SYSTEM,
                user_prompt=f"Follow-up request:\n{follow_up}\n\nLatest planner update:\n{planner_text}\n\nLatest builder update:\n{builder_text}",
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

        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": f"[FOLLOW-UP] {follow_up}",
            "context": "",
            "planner": planner_text,
            "builder": builder_text,
            "critic": critic_text,
        })

planner_out = st.session_state.outputs["planner"]
builder_out = st.session_state.outputs["builder"]
critic_out = st.session_state.outputs["critic"]

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Planner")
    st.markdown(planner_out or "_No output yet._")

with col2:
    st.subheader("Builder")
    st.markdown(builder_out or "_No output yet._")

with col3:
    st.subheader("Critic")
    st.markdown(critic_out or "_No output yet._")

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

st.divider()
st.subheader("Session history")
if not st.session_state.history:
    st.caption("No runs yet.")
else:
    for i, item in enumerate(st.session_state.history[:10], 1):
        with st.expander(f"{i}. {item['time']} — {item['task'][:80]}"):
            st.markdown(f"**Task:** {item['task']}")
            if item["context"]:
                st.markdown(f"**Context:** {item['context']}")
            st.markdown(f"**Planner:** {item['planner']}")
            st.markdown(f"**Builder:** {item['builder']}")
            st.markdown(f"**Critic:** {item['critic']}")
