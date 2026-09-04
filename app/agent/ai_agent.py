from agent.tools import make_tools
from agent.memory import store_turn, recall_relevant_turns
from rag.rag_manager import search_history

GROQ_MODEL = "openai/gpt-oss-20b"


def run_agent(
    groq_client,
    user_question: str,
    session_id: str,
    current_image=None,
    blip_processor=None,
    blip_model=None,
    object_detector=None,
) -> str:

    # ---- Tools (empty if no image provided) ----
    tool_functions, tool_schemas = make_tools(
        current_image, blip_processor, blip_model, object_detector
    )

    # ---- RAG ----
    rag_matches = search_history(user_question, top_k=3)
    rag_context = (
        "\n".join(f"- {m['caption']}" for m in rag_matches)
        if rag_matches
        else "No relevant past captions found."
    )

    # ---- Memory ----
    past_turns = recall_relevant_turns(user_question, session_id=session_id, top_k=3)
    memory_context = "\n\n".join(past_turns) if past_turns else "No relevant past conversation."

    image_note = (
        "An image has been uploaded and tools are available to analyze it."
        if current_image is not None
        else "No image has been uploaded — answer as a general chatbot; "
             "no image tools are available right now."
    )

    system_prompt = f"""You are a helpful assistant. {image_note}
You are also given retrieved context below — use it only if relevant to
the question; ignore it otherwise.

Relevant past caption history (RAG):
{rag_context}

Relevant past conversation (Memory):
{memory_context}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    # ---- First call ----
    # Only pass tools/tool_choice if there actually are tools — Groq
    # rejects an empty tools list with tool_choice="auto" in some cases,
    # and it's just unnecessary overhead when chatting with no image.
    create_kwargs = {"model": GROQ_MODEL, "messages": messages}
    if tool_schemas:
        create_kwargs["tools"] = tool_schemas
        create_kwargs["tool_choice"] = "auto"

    response = groq_client.chat.completions.create(**create_kwargs)
    response_message = response.choices[0].message

    if getattr(response_message, "tool_calls", None):
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func = tool_functions.get(func_name)
            result = func() if func else f"Unknown tool: {func_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": result,
            })

        second_response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
        )
        final_answer = second_response.choices[0].message.content
    else:
        final_answer = response_message.content

    store_turn(session_id=session_id, user_question=user_question, final_answer=final_answer)

    return final_answer