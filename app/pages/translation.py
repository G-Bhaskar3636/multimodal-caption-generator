import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB language codes for the languages you're using. Add more here if
# needed — full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANG_CODES = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "telugu": "tel_Telu",
}

# Cached at module level so the model only loads once, the first time
# translate_text() is actually called (lazy loading) — not at import time.
_tokenizer = None
_model = None
_device = None


def _get_model():
    global _tokenizer, _model, _device

    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME).to(_device)
        _model.eval()

    return _tokenizer, _model, _device


def Translate_text(text: str, target_language: str, max_new_tokens: int = 60) -> str:
    """
    Translate `text` (assumed English) into `target_language`.

    target_language should be one of: "english", "hindi", "telugu"
    (case-insensitive). "english" is a no-op — returns text unchanged.
    """
    target_language = target_language.strip().lower()

    if target_language not in NLLB_LANG_CODES:
        raise ValueError(
            f"Unsupported language '{target_language}'. "
            f"Expected one of: {list(NLLB_LANG_CODES.keys())}"
        )

    if target_language == "english":
        return text

    tokenizer, model, device = _get_model()
    target_code = NLLB_LANG_CODES[target_language]

    inputs = tokenizer(text, return_tensors="pt").to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)