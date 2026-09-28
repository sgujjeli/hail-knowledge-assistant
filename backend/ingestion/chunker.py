"""
Text chunking — RecursiveCharacterTextSplitter.

Why these settings:
  chunk_size=800   — fits comfortably inside GPT-4o's context while giving
                     enough content for the retriever to find signal.
  chunk_overlap=150 — ~18% overlap so sentences that straddle a boundary
                      are represented in both chunks. Prevents mid-sentence cuts
                      from losing key context.
  separators        — tries paragraph breaks first, then newlines, then
                      sentences, then words — always prefers semantic splits
                      over hard character limits.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(pages: list[dict], doc_id: str, filename: str) -> list[dict]:
    """
    Split page text into overlapping chunks.

    Args:
        pages:    list of page dicts from parser.parse_pdf / parse_docx
        doc_id:   UUID of the parent document (for filtering in AI Search)
        filename: original filename (shown in citations)

    Returns:
        list of chunk dicts ready for embedding and indexing:
        {
            chunk_id, doc_id, filename, page_number,
            text, chunk_index, char_count
        }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    for page in pages:
        if not page["text"].strip():
            continue  # skip blank pages

        splits = splitter.split_text(page["text"])

        for chunk_index, split_text in enumerate(splits):
            chunks.append({
                "chunk_id": f"{doc_id}_p{page['page_number']}_c{chunk_index}",
                "doc_id": doc_id,
                "filename": filename,
                "page_number": page["page_number"],
                "text": split_text.strip(),
                "chunk_index": chunk_index,
                "char_count": len(split_text),
            })

    return chunks
