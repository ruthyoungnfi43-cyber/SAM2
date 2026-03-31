import base64
import io
import os
import time
from datetime import datetime

import requests
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

st.set_page_config(page_title="Custom Grok Hacker", page_icon="⚡", layout="wide")

CHAT_MODEL_OPTIONS = [
    "grok-4",
    "grok-4.20-reasoning",
    "grok-4.20-multi-agent",
]

IMAGE_MODEL_OPTIONS = [
    "grok-imagine-image",
]

VIDEO_MODEL_OPTIONS = [
    "grok-imagine-video",
]

IMAGE_SIZE_OPTIONS = [
    "1024x1024",
    "1536x1024",
    "1024x1536",
]

VIDEO_ASPECT_OPTIONS = [
    "16:9",
    "9:16",
    "1:1",
]

VIDEO_DURATION_OPTIONS = [
    5,
    10,
    15,
]

CASUAL_PHRASES = [
    "hi", "hello", "hey", "yo", "sup", "what's up", "whats up",
    "how are you", "how are u", "how r u", "who are you", "who r u",
    "good morning", "good evening", "thanks", "thank you"
]

if "chat_output" not in st.session_state:
    st.session_state.chat_output = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "run_times" not in st.session_state:
    st.session_state.run_times = []

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if "edited_images" not in st.session_state:
    st.session_state.edited_images = []

if "generated_videos" not in st.session_state:
    st.session_state.generated_videos = []


def inject_theme():
    st.markdown(
        """
        <style>
        :root {
            --text: #d8ffe7;
            --muted: #8ab79b;
            --green: #00ff88;
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
        .hack-card {
            background: linear-gradient(180deg, rgba(8,12,14,.96), rgba(9,15,18,.96));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: var(--glow);
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


def get_api_key():
    return get_secret("XAI_API_KEY")


def get_client() -> OpenAI | None:
    api_key = get_api_key()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=120,
    )


def enforce_rate_limit():
    now = time.time()
    window_seconds = 60
    max_runs = 8
    st.session_state.run_times = [t for t in st.session_state.run_times if now - t < window_seconds]
    if len(st.session_state.run_times) >= max_runs:
        st.error("Rate limit hit. Wait a minute and try again.")
        st.stop()
    st.session_state.run_times.append(now)


def file_to_data_uri(uploaded_file):
    if uploaded_file is None:
        return None
    mime = uploaded_file.type or "application/octet-stream"
    raw = uploaded_file.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


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


def call_chat(client, model, system_prompt, user_prompt):
    resp = client.responses.create(
        model=model,
        store=True,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return extract_text(resp)


def build_pdf(title: str, content: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=32, rightMargin=32, topMargin=32, bottomMargin=32)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
        Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["BodyText"]),
        Spacer(1, 16),
        Preformatted(content or "(empty)", styles["Code"]),
    ]
    doc.build(story)
    buf.seek(0)
    return buf.read()


def render_card(title: str, content: str):
    safe_content = content if content else '<span class="muted">No output yet.</span>'
    html = f"""
    <div class="hack-card">
        <div class="hack-title">{title}</div>
        <div>{safe_content}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_image(client, model: str, prompt: str, size: str, n: int):
    return client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=n,
    )


def edit_image_raw(prompt: str, image_data_uri: str, model: str):
    api_key = get_api_key()
    url = "https://api.x.ai/v1/images/edits"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {
        "prompt": (None, prompt),
        "model": (None, model),
        "image": (None, image_data_uri),
    }
    r = requests.post(url, headers=headers, files=files, timeout=180)
    r.raise_for_status()
    return r.json()


def create_video_raw(prompt: str, image_data_uri: str, model: str, aspect_ratio: str, duration: int):
    api_key = get_api_key()
    url = "https://api.x.ai/v1/videos/generations"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "image": image_data_uri,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


CUSTOM_GROK_SYSTEM = """You are Custom Grok.
Reply as a direct one-way assistant.
Rules:
- No planner / builder / critic structure.
- No self-referential fluff.
- Be sharp, useful, and concise.
- Match casual tone naturally.
- If the user asks for creation, return the finished result.
"""

inject_theme()
client = get_client()

if not client:
    st.error("Missing XAI_API_KEY in Streamlit Secrets or environment.")
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>Custom Grok Hacker</h1>
        <p>One-way chat • Imagine image generation • Image edit • Video from image • Secrets-only API</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("xAI")
    st.success("API loaded from Secrets")

    st.subheader("Chat")
    chat_model = st.selectbox("Chat model", CHAT_MODEL_OPTIONS, index=1)

    st.subheader("Image")
    image_model = st.selectbox("Image model", IMAGE_MODEL_OPTIONS, index=0)
    image_size = st.selectbox("Image size", IMAGE_SIZE_OPTIONS, index=0)
    image_count = st.selectbox("Image count", [1, 2, 3, 4], index=0)

    st.subheader("Video")
    video_model = st.selectbox("Video model", VIDEO_MODEL_OPTIONS, index=0)
    video_aspect = st.selectbox("Video aspect ratio", VIDEO_ASPECT_OPTIONS, index=0)
    video_duration = st.selectbox("Video seconds", VIDEO_DURATION_OPTIONS, index=0)

    if st.button("Clear session", use_container_width=True):
        st.session_state.chat_output = ""
        st.session_state.chat_history = []
        st.session_state.generated_images = []
        st.session_state.edited_images = []
        st.session_state.generated_videos = []
        st.success("Session cleared.")
        st.rerun()

tab_chat, tab_imagine, tab_edit, tab_video = st.tabs(["⚡ Chat", "🖼️ Imagine", "🛠️ Edit Image", "🎬 Video"])

with tab_chat:
    user_prompt = st.text_area("Message", value="", height=220, placeholder="Talk to Custom Grok...")
    send_chat = st.button("Send", use_container_width=True)

    if send_chat:
        enforce_rate_limit()
        if not user_prompt.strip():
            st.error("Enter a message first.")
        else:
            with st.spinner("Thinking..."):
                st.session_state.chat_output = call_chat(
                    client=client,
                    model=chat_model,
                    system_prompt=CUSTOM_GROK_SYSTEM,
                    user_prompt=user_prompt,
                )
            st.session_state.chat_history.insert(0, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": user_prompt,
                "assistant": st.session_state.chat_output,
            })

    render_card("Custom Grok", st.session_state.chat_output)

    if st.session_state.chat_output:
        pdf_bytes = build_pdf("Custom Grok Chat Output", st.session_state.chat_output)
        st.download_button(
            "Download Chat PDF",
            data=pdf_bytes,
            file_name=f"custom_grok_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("### Chat history")
    if not st.session_state.chat_history:
        st.caption("No messages yet.")
    else:
        for i, item in enumerate(st.session_state.chat_history[:10], 1):
            with st.expander(f"{i}. {item['time']}"):
                st.markdown(f"**You:** {item['user']}")
                st.markdown(f"**Custom Grok:** {item['assistant']}")

with tab_imagine:
    image_prompt = st.text_area("Image prompt", value="", height=180, placeholder="Describe the image you want...")
    run_image = st.button("Generate image", use_container_width=True)

    if run_image:
        enforce_rate_limit()
        if not image_prompt.strip():
            st.error("Enter an image prompt first.")
        else:
            with st.spinner("Generating image..."):
                resp = create_image(client, image_model, image_prompt, image_size, image_count)
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

with tab_edit:
    edit_source = st.file_uploader("Upload image to edit", type=["png", "jpg", "jpeg", "webp"], key="edit_upload")
    edit_prompt = st.text_area("Edit prompt", value="", height=160, placeholder="Describe exactly what to change...")
    run_edit = st.button("Edit image", use_container_width=True)

    if run_edit:
        enforce_rate_limit()
        if edit_source is None:
            st.error("Upload an image first.")
        elif not edit_prompt.strip():
            st.error("Enter an edit prompt first.")
        else:
            with st.spinner("Editing image..."):
                image_data_uri = file_to_data_uri(edit_source)
                data = edit_image_raw(edit_prompt, image_data_uri, image_model)
                urls = [item.get("url") for item in data.get("data", []) if item.get("url")]
                st.session_state.edited_images = urls

    if st.session_state.edited_images:
        cols = st.columns(2)
        for idx, url in enumerate(st.session_state.edited_images):
            with cols[idx % 2]:
                st.image(url, use_container_width=True)
                st.code(url, language="text")

with tab_video:
    video_source = st.file_uploader("Upload source image for video", type=["png", "jpg", "jpeg", "webp"], key="video_upload")
    video_prompt = st.text_area("Video prompt", value="", height=160, placeholder="Describe the motion, camera move, and style...")
    run_video = st.button("Generate video", use_container_width=True)

    if run_video:
        enforce_rate_limit()
        if video_source is None:
            st.error("Upload a source image first.")
        elif not video_prompt.strip():
            st.error("Enter a video prompt first.")
        else:
            with st.spinner("Generating video..."):
                image_data_uri = file_to_data_uri(video_source)
                data = create_video_raw(video_prompt, image_data_uri, video_model, video_aspect, video_duration)
                urls = []
                if "data" in data:
                    for item in data["data"]:
                        if isinstance(item, dict):
                            if item.get("url"):
                                urls.append(item["url"])
                            elif item.get("video_url"):
                                urls.append(item["video_url"])
                st.session_state.generated_videos = urls

    if st.session_state.generated_videos:
        for url in st.session_state.generated_videos:
            st.video(url)
            st.code(url, language="text")
