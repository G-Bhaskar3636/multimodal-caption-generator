from pathlib import Path
import sys

import streamlit as st
import torch
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from google import genai
from dotenv import dotenv_values

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rag.rag_manager import store_caption



ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
env_vars = dotenv_values(ENV_PATH)
gemini_api_key = env_vars.get("GEMINI_API_KEY")

if not gemini_api_key:
    st.error(
        f"GEMINI_API_KEY not found. Checked for a .env file at: {ENV_PATH}\n\n"
        "Make sure the file exists there and contains a line like:\n"
        "GEMINI_API_KEY=your_actual_key_here"
    )
    st.stop()

# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Image Story Generator",
    page_icon="🖼️"
)

st.title("🖼️ Image Story Generator")

# =========================================================
# DEVICE
# =========================================================

device = 0 if torch.cuda.is_available() else -1


# =========================================================
# LOAD HUGGING FACE MODELS
# =========================================================

@st.cache_resource
def load_models():
    detector_path = (
        "models/models--facebook--detr-resnet-101-dc5/"
        "snapshots/"
        "96317ca979e231bd960cb3cac31328e0165a3e94"
    )

    object_detector = pipeline(
        "object-detection",
        model=detector_path,
        device=device
    )

    narrator_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    narrator_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    narrator_model.to("cuda" if device == 0 else "cpu")
    narrator_model.eval()

    return object_detector, narrator_processor, narrator_model


object_detector, narrator_processor, narrator_model = load_models()


# =========================================================
# GEMINI
# =========================================================

@st.cache_resource
def load_gemini():
    client = genai.Client(
        api_key=gemini_api_key
    )

    return client


gemini_client = load_gemini()

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_image = st.file_uploader(
    "📁 Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_image is not None:

    st.markdown("---")

    image = Image.open(uploaded_image).convert("RGB")
    st.image(uploaded_image)

    # =========================================================
    # LANGUAGE SELECTION
    # =========================================================

    LANGUAGES = ["English", "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada",
                 "Kashmiri", "Konkani", "Maithili", "Malayalam", "Manipuri", "Marathi", "Nepali",
                 "Odia", "Punjabi", "Sanskrit", "Santali", "Sindhi", "Tamil", "Telugu", "Urdu"
                 ]
    language = st.selectbox(
        "🌐 Select Language",
        LANGUAGES,
        index=0  # "English" is first in the list, so it's the default selection
    )

    st.markdown("---")

    if st.button(
            "🚀 Generate Story",
            use_container_width=True
    ):
        # =================================================
        # STEP 1 — OBJECT DETECTION (no output shown, feeds the prompt only)
        # =================================================


        with st.spinner("Generating story..."):
            detections = object_detector(
                image,
                threshold=0.30
            )

            detected_objects = [
                result["label"]
                for result in detections
            ]
            detected_objects = list(
                dict.fromkeys(detected_objects)
            )

            # =================================================
            # STEP 2 — IMAGE CAPTION (no output shown, feeds the prompt only)
            # =================================================

            caption_result = narrator_processor(images=image, return_tensors="pt")
            caption_result = {k: v.to(narrator_model.device) for k, v in caption_result.items()}

            with torch.no_grad():
                output_ids = narrator_model.generate(**caption_result, max_new_tokens=40)

            image_caption = narrator_processor.decode(output_ids[0], skip_special_tokens=True)

            st.write(image_caption)


            # =================================================
            # STEP 3 — STORY GENERATION
            # =================================================

            objects_text = ", ".join(
                detected_objects
            )

            prompt = f"""
                You are a creative storyteller.

                Analyze the following image information.

                Detected objects:
                {objects_text}

                Image description:
                {image_caption}

                Create a short, engaging story based on the image.

                The story should:
                - Connect the detected objects naturally.
                - Describe what may be happening in the scene.
                - Be creative but consistent with the image.
                - Be around 100-150 words.
                - Do not invent completely unrelated objects.

                Write the story in {language}.
                """

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            story = response.text

        # =================================================
        # STEP 4 — DISPLAY FINAL OUTPUT ONLY
        # =================================================

        st.markdown("---")
        st.write(story)

        store_caption(
            english_caption=image_caption,
            translated_caption=story,
            language=language,
        )

st.markdown("---")

st.info("⚠️ **AI can make mistakes. Please verify important information.**")