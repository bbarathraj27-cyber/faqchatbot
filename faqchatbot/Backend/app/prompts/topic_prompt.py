"""
Prompt for extracting topic keywords from a chunk of text, used to
populate the Neo4j knowledge graph during ingestion.
"""

TOPIC_EXTRACTION_PROMPT = """Extract 2 to 4 short topic keywords (1-3 words each) that best represent
the subject matter of the text below. Return ONLY a comma-separated list,
nothing else. No explanations, no numbering.

Text:
{text}
"""


def build_topic_extraction_prompt(text: str) -> list[dict]:
    return [
        {"role": "user", "content": TOPIC_EXTRACTION_PROMPT.format(text=text)},
    ]


# Batch version: extracts topics for many chunks in a single LLM call instead
# of one call per chunk. This is the difference between ~1 Groq call per
# chunk (thousands of calls for a full site ingest) and ~1 call per 15-20
# chunks. Chunks are numbered so the model's numbered JSON output can be
# matched back to the right chunk even if one entry is skipped/malformed.
BATCH_TOPIC_EXTRACTION_PROMPT = """You will be given {count} numbered text chunks. For EACH chunk, extract 2 to 4
short topic keywords (1-3 words each) that best represent its subject matter.

Return ONLY a JSON array with exactly {count} elements, one per chunk, in the
same order as the chunks below. Each element must be a JSON array of strings
(the topics for that chunk). No explanations, no markdown, no extra text —
just the raw JSON array.

Example output format for 2 chunks:
[["neural networks", "backpropagation"], ["data pipelines", "etl"]]

Chunks:
{chunks}
"""


def build_batch_topic_extraction_prompt(texts: list[str]) -> list[dict]:
    numbered = "\n\n".join(f"[{i+1}] {text}" for i, text in enumerate(texts))
    content = BATCH_TOPIC_EXTRACTION_PROMPT.format(count=len(texts), chunks=numbered)
    return [
        {"role": "user", "content": content},
    ]