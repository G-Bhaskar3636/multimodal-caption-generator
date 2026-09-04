import streamlit as st

st.set_page_config(page_title="Multimodal Caption Generator", page_icon="🖼️")

st.title("🖼️ Multimodal Caption Generator")

st.write(
    "Turn any image into a caption and a short story — in the language of "
    "your choice — then search, chat, and explore what you've created."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🖼️ Image Captioning")
    st.write(
        "Upload an image, pick a language, and get an AI-generated "
        "caption and short story built from what's actually in the picture."
    )

with col2:
    st.subheader("🔍 Search History")
    st.write(
        "Every caption you generate is remembered. Search past uploads "
        "by meaning — \"photos with dogs\" finds them even without that "
        "exact word."
    )

with col3:
    st.subheader("🤖 AI Chatbot")
    st.write(
        "Chat freely, or upload an image mid-conversation to unlock "
        "tools like captioning and object detection — the agent decides "
        "what to use and remembers what you've discussed."
    )

st.markdown("---")

st.info("👈 Use the sidebar to get started with any of the sections above.")