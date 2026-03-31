import io
import os
import time
from datetime import datetime

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

st.set_page_config(page_title="Grok Hacker SaaS", page_icon="⚡", layout="wide")

CHAT_MODEL_OPTIONS = [
    "grok-4",
    "grok-4.20-reasoning",
    "grok-4.20-multi-agent",
]

IMAGE_MODEL_OPTIONS = [
    "grok-imagine-image",
]

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

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []


def inject_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg: #05080a;
            --panel: #0a0f12;
            --panel2: #0d1418;
            --text: #d8ffe7;
            --muted: #8ab79b;
            --green: #00ff88;
            --green2: #00cc6f;
            --border: rgba(0,255,136,.18);
            --glow: 0 0 18px rgba(0,255,136,.18);
        }
        .stApp {
            background:
              radial-gradient(circle at top right, rgba(0,255,136,.08), transparent 22%),
              radial-gradient(circle at bottom left, rgba(0,255,136,.06), transparent 18%),
              linear-gradient(180deg, #030507 0%, #071014 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1014 0%, #101720 100%);
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, .stMarkdown, label, p, div {
            color: var(--text);
        }
        .block-container {
            padding-top: 1.5rem;
        }
        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stSelectbox > div > div,
        .stNumberInput input {
            background: rgba(8,12,14,.9) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            box-shadow: var(--glow);
            border-radius: 12px !important;
        }
        .stButton > button,
        .stDownloadButton > button {
            background: linear-gradient(180deg, #0d1b14 0%, #0a140f 100%) !important;
            color: var(--green) !important;
            border: 1px solid rgba(0,255,136,.35) !important;
            border-radius: 14px !important;
            box-shadow: 0 0 20px rgba(0,255,136,.12);
            font-weight: 700 !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--green) !important;
            box-shadow: 0 0 28px rgba(0,255,136,.22);
        }
        .hack-card {
            background: linear-gradient(180deg, rgba(8,12,14,.96), rgba(9,15,18,.96));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: var(--glow);
            min-height: 220px;
        }
        .hack-title {
            color: var(--green);
            font-weight: 800;
            letter-spacing: .04em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .muted {
            color: var(--muted);
        }
        .hero {
            padding: 18px 22px;
            border: 1px solid var(--border);
            border-radius: 20px;
            background:
              linear-gradient(135deg, rgba(0,255,136,.08), rgba(0,0,0,0) 35%),
              linear-gradient(180deg, rgba(9,15,18,.94), rgba(8,12,14,.98));
            box-shadow: 0 0 24px rgba(0,255,136,.12);
            margin-bottom: 16px;
        }
        .hero h1 {
            color: #eafff3;
            font-size: 3rem;
            margin-bottom: .2rem;
        }
        .hero p {
            color: var(--muted);
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


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
    return any(p in combined for p in CASUAL_PHRASES) and len(combined) <= 160


def enforce_rate_limit():
    now = time.time()
    window_seconds = 60
    max_runs = 8
    st.session_state.run_times = [t for t in st.session_state.run_times if now - t < window_seconds]
    if len(st.session_state.run_times) >= max_runs:
        st.error("Rate limit hit. Wait a minute and try again.")
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
    payload = {"model": model, "store": True}

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


def generate_image(client, model: str, prompt: str, size: str, n: int = 1):
    return client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=n,
    )


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


def render_hack_card(title: str, content: str):
    safe_content = content if content else '<span class="muted">No output yet.</span>'
    html = f"""
    <div class="hack-card">
        <div class="hack-title">{title}</div>
        <div>{safe_content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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

inject_theme()
client = get_client()

if not client:
    st.error("Missing XAI_API_KEY in Streamlit Secrets or environment.")
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>Grok Hacker SaaS</h1>
        <p>Neon hacker UI • Select-menu controls • Chat + Imagine image generation • Secrets-only API</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("xAI")
    st.success("API loaded from Secrets")

    st.subheader("Models")
    planner_model = st.selectbox("Planner model", CHAT_MODEL_OPTIONS, index=2)
    builder_model = st.selectbox("Builder model", CHAT_MODEL_OPTIONS, index=1)
    critic_model = st.selectbox("Critic model", CHAT_MODEL_OPTIONS, index=1)
    image_model = st.selectbox("Imagine model", IMAGE_MODEL_OPTIONS, index=0)

    st.subheader("Mode")
    search_mode = st.selectbox("Web search", ["Off", "Planner only", "Builder only", "Critic only", "All"], index=0)

    planner_web = search_mode in ["Planner only", "All"]
    builder_web = search_mode in ["Builder only", "All"]
    critic_web = search_mode in ["Critic only", "All"]

    st.subheader("Image options")
    image_size = st.selectbox("Image size", ["1024x1024", "1536x1024", "1024x1536"], index=0)
    image_count = st.selectbox("Number of images", [1, 2, 3, 4], index=0)

    if st.button("Clear session", use_container_width=True):
        st.session_state.prev_ids = {"planner": None, "builder": None, "critic": None}
        st.session_state.outputs = {"planner": "", "builder": "", "critic": ""}
        st.session_state.history = []
        st.session_state.generated_images = []
        st.success("Session cleared.")
        st.rerun()

tab_chat, tab_imagine = st.tabs(["⚡ Chat", "🖼️ Imagine"])

with tab_chat:
    task = st.text_area("Task", value="", height=180, placeholder="Ask anything...")
    context = st.text_area("Project context", value="", height=120, placeholder="Optional context...")
    follow_up = st.text_input("Follow-up for current thread (optional)")

    c1, c2 = st.columns(2)
    run_fresh = c1.button("Run fresh", use_container_width=True)
    run_continue = c2.button("Continue current thread", use_container_width=True)

    if run_fresh:
        enforce_rate_limit()
        casual = is_casual(task, context)

        if casual:
            planner_text = "Casual conversation detected. Use a short natural reply."
            builder_text = "Hey! I'm your Grok-powered assistant ⚡ What do you need?"
            critic_text = "Natural and appropriate for casual chat."
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
                planner_resp = call_agent(client, planner_model, PLANNER_SYSTEM, planner_prompt, planner_web)
                planner_text = extract_text(planner_resp)

                builder_prompt = f"""Task:
{task}

Project context:
{context}

Planner output:
{planner_text}

Produce the best practical final result.
"""
                builder_resp = call_agent(client, builder_model, BUILDER_SYSTEM, builder_prompt, builder_web)
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
                critic_resp = call_agent(client, critic_model, CRITIC_SYSTEM, critic_prompt, critic_web)
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
                    client, planner_model, PLANNER_SYSTEM, follow_up, planner_web,
                    previous_response_id=st.session_state.prev_ids["planner"],
                )
                planner_text = extract_text(planner_resp)

                builder_resp = call_agent(
                    client, builder_model, BUILDER_SYSTEM,
                    f"Follow-up request:\n{follow_up}\n\nLatest planner update:\n{planner_text}",
                    builder_web,
                    previous_response_id=st.session_state.prev_ids["builder"],
                )
                builder_text = extract_text(builder_resp)

                critic_resp = call_agent(
                    client, critic_model, CRITIC_SYSTEM,
                    f"Follow-up request:\n{follow_up}\n\nLatest planner update:\n{planner_text}\n\nLatest builder update:\n{builder_text}",
                    critic_web,
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
        render_hack_card("Planner", planner_out)

    with col2:
        render_hack_card("Builder", builder_out)

    with col3:
        render_hack_card("Critic", critic_out)

    if planner_out or builder_out or critic_out:
        pdf_bytes = build_pdf(
            title="Grok Hacker SaaS Output",
            planner=planner_out,
            builder=builder_out,
            critic=critic_out,
        )
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"grok_hacker_saas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("### Session history")
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

with tab_imagine:
    st.markdown("### Grok Imagine")
    image_prompt = st.text_area("Image prompt", value="", height=180, placeholder="Describe the image you want...")
    image_run = st.button("Generate image", use_container_width=True)

    if image_run:
        enforce_rate_limit()
        if not image_prompt.strip():
            st.error("Enter an image prompt first.")
        else:
            with st.spinner("Generating image..."):
                resp = generate_image(client, image_model, image_prompt, image_size, image_count)
                urls = []
                for item in getattr(resp, "data", []) or []:
                    url = getattr(item, "url", None) if not isinstance(item, dict) else item.get("url")
                    if url:
                        urls.append(url)
                st.session_state.generated_images = urls

    if st.session_state.generated_images:
        cols = st.columns(2)
        for idx, url in enumerate(st.session_state.generated_images):
            with cols[idx % 2]:
                st.image(url, use_container_width=True)
                st.code(url, language="text")
