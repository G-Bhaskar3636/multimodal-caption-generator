import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from PIL import Image
from groq import Groq
from dotenv import dotenv_values

from agent.ai_agent import run_agent

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Chat")
st.caption(
    "Chat normally, or optionally upload an image to unlock image-analysis tools."
)

# =========================================================
# ENV / GROQ
# =========================================================

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
env_vars = dotenv_values(ENV_PATH)
groq_api_key = env_vars.get("GROQ_API_KEY")

if not groq_api_key:
    st.error(f"GROQ_API_KEY not found. Checked at: {ENV_PATH}")
    st.stop()


@st.cache_resource
def load_groq():
    return Groq(api_key=groq_api_key)


groq_client = load_groq()


# =========================================================
# SESSION ID + CHAT HISTORY
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# =========================================================
# OPTIONAL IMAGE UPLOAD — image tools are a bonus, not required
# =========================================================

current_image = None
blip_processor = blip_model = object_detector = None

with st.expander("📁 Optional: upload an image to unlock image tools"):
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_image is not None:
        current_image = Image.open(uploaded_image).convert("RGB")
        st.image(current_image, use_container_width=True)

        # Models only load if an image is actually provided — no reason
        # to pay the load cost for a plain-text chat.
        @st.cache_resource
        def load_image_models():
            import torch
            from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration

            device = 0 if torch.cuda.is_available() else -1
            detector_path = (
                "models/models--facebook--detr-resnet-101-dc5/"
                "snapshots/"
                "96317ca979e231bd960cb3cac31328e0165a3e94"
            )
            object_detector = pipeline("object-detection", model=detector_path, device=device)

            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            model.to("cuda" if device == 0 else "cpu")
            model.eval()

            return object_detector, processor, model

        object_detector, blip_processor, blip_model = load_image_models()
        st.success("Image tools enabled for this conversation.")


# =========================================================
# CHAT DISPLAY
# =========================================================

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# =========================================================
# CHAT INPUT — always available, image or not
# =========================================================

user_question = st.chat_input("Type a message...")

if user_question:
    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            final_answer = run_agent(
                groq_client=groq_client,
                user_question=user_question,
                session_id=st.session_state.session_id,
                current_image=current_image,
                blip_processor=blip_processor,
                blip_model=blip_model,
                object_detector=object_detector,
            )
        st.write(final_answer)

    st.session_state.chat_messages.append({"role": "assistant", "content": final_answer})

st.info("⚠️ **AI can make mistakes. Please verify important information.**")