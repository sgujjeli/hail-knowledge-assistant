"""
Prompt engineering — system instructions and context injection.

Design principles:
  - ONLY-use-context instruction prevents hallucination from training data
  - Citations are mandatory ([Filename, Page X] format)
  - Temperature=0 in llm.py enforces determinism
  - REVIEW_REQUIRED flag surfaced when faithfulness < threshold
"""

SYSTEM_PROMPT = """You are Hail, an AI knowledge assistant at hailoop.co.uk that answers questions
strictly from the documents provided in the context below.

Rules you must always follow:
1. Use ONLY the information in <context> to answer. Do NOT use prior knowledge.
2. If the answer is not in the context, respond exactly:
   "I cannot find that information in the uploaded documents."
3. Cite every claim with [Filename, Page X] immediately after the sentence.
4. Keep answers concise and factual. Use bullet points for lists.
5. If multiple documents contradict each other, note the discrepancy.
6. Never invent, extrapolate, or assume beyond what is stated in context.

<context>
{context}
</context>"""

NO_CONTEXT_RESPONSE = (
    "I cannot find that information in the uploaded documents. "
    "Please upload a relevant document or rephrase your question."
)


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into the {context} placeholder.
    Each chunk is labelled with its source for citation.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = f"[{chunk['filename']}, Page {chunk['page_number']}]"
        parts.append(f"--- Chunk {i} {source} ---\n{chunk['text']}")
    return "\n\n".join(parts)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Construct the OpenAI messages array for a grounded chat completion.
    """
    context_block = build_context_block(chunks)
    system_content = SYSTEM_PROMPT.format(context=context_block)

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": query},
    ]


def extract_citations(response_text: str, chunks: list[dict]) -> list[dict]:
    """
    Parse [Filename, Page X] citations from the LLM response and enrich
    with chunk metadata for the frontend to render as source cards.
    """
    import re
    pattern = r"\[([^\]]+),\s*Page\s*(\d+)\]"
    matches = re.findall(pattern, response_text)

    seen = set()
    citations = []
    for filename, page_str in matches:
        key = (filename.strip(), int(page_str))
        if key in seen:
            continue
        seen.add(key)

        # Find the matching chunk for the blob URL (optional enrichment)
        matching = next(
            (c for c in chunks
             if c["filename"] == filename.strip() and c["page_number"] == int(page_str)),
            None,
        )
        citations.append({
            "filename":    filename.strip(),
            "page_number": int(page_str),
            "chunk_id":    matching["chunk_id"] if matching else None,
        })

    return citations
