"""
Tools available to the agent. Image-analysis tools (caption_image,
detect_objects) are only built and offered to the LLM when an image is
actually present — the chatbot works fine with zero tools too, it just
answers from its own knowledge in that case.
"""

import torch


def make_tools(current_image=None, blip_processor=None, blip_model=None, object_detector=None):
    """
    Returns (tool_functions, tool_schemas). Both are empty if no image
    was provided — the agent then just chats normally, with no tools
    offered to the LLM.
    """

    tool_functions = {}
    tool_schemas = []

    if current_image is None:
        return tool_functions, tool_schemas

    def caption_image() -> str:
        inputs = blip_processor(images=current_image, return_tensors="pt").to(blip_model.device)
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs, max_new_tokens=40)
        return blip_processor.decode(output_ids[0], skip_special_tokens=True)

    def detect_objects() -> str:
        detections = object_detector(current_image, threshold=0.30)
        labels = [d["label"] for d in detections]
        if not labels:
            return "No objects were detected with reasonable confidence."
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return ", ".join(f"{count}x {label}" for label, count in counts.items())

    tool_functions["caption_image"] = caption_image
    tool_functions["detect_objects"] = detect_objects

    tool_schemas.extend([
        {
            "type": "function",
            "function": {
                "name": "caption_image",
                "description": (
                    "Generate a general English caption describing the overall "
                    "content of the currently uploaded image. Use this for broad "
                    "questions like 'what is this?' or 'describe this image'. "
                    "Only relevant if the user has uploaded an image."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_objects",
                "description": (
                    "Detect and list distinct objects present in the currently "
                    "uploaded image, with counts. Use this for questions like "
                    "'what objects are in this?', 'is there a X in this image?', "
                    "or 'how many Y are there?'. Only relevant if the user has "
                    "uploaded an image."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ])

    return tool_functions, tool_schemas