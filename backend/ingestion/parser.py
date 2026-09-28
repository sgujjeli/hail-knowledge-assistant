"""
Document parser — PDF and DOCX.

PDF:  PyMuPDF (fitz) extracts text per page + embedded images as base64.
      Images are passed to GPT-4o Vision for alt-text / caption generation
      so their content becomes searchable text in the vector index.

DOCX: python-docx extracts paragraph text; paragraphs are grouped into
      synthetic "pages" of ~1000 words to match the PDF page abstraction.
"""
import pymupdf as fitz   # PyMuPDF >= 1.24 — suppresses fitz deprecation warning
import base64
import io
from docx import Document as DocxDocument   # python-docx


# ── PDF ─────────────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> dict:
    """
    Parse a PDF into structured pages.

    Returns:
        {
            "pages": [
                {
                    "page_number": int,
                    "text": str,
                    "images": [{"index": int, "ext": str, "data": str (base64)}],
                    "char_count": int,
                }
            ],
            "total_pages": int,
        }
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc):
        # 1. Extract text — "text" mode preserves reading order
        text = page.get_text("text").strip()

        # 2. Extract images embedded in the page
        images = []
        for img_index, img_ref in enumerate(page.get_images(full=True)):
            xref = img_ref[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"]          # png, jpeg, etc.
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append({
                "index": img_index,
                "ext": img_ext,
                "data": img_b64,                 # pass to GPT-4o Vision for captioning
                "width": base_image.get("width"),
                "height": base_image.get("height"),
            })

        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "images": images,
            "char_count": len(text),
        })

    doc.close()
    return {"pages": pages, "total_pages": len(pages)}


async def caption_image_with_gpt4o(img_b64: str, img_ext: str) -> str:
    """
    Send an image to GPT-4o Vision and get a descriptive caption.
    This caption is appended to the page text so diagrams become searchable.

    Endpoint used:
        POST https://{resource}.openai.azure.com/openai/deployments/gpt-4o/chat/completions
             ?api-version=2024-02-01
    """
    from openai import AsyncAzureOpenAI
    from core.config import settings

    client = AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )

    response = await client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_ext};base64,{img_b64}",
                            "detail": "low",    # "low" = faster, cheaper; "high" for diagrams
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this image concisely for a search index. "
                            "Include any text, numbers, labels, or key concepts visible. "
                            "Max 100 words."
                        ),
                    },
                ],
            }
        ],
        max_tokens=150,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── DOCX ────────────────────────────────────────────────────────────────────

def parse_docx(file_bytes: bytes) -> dict:
    """
    Parse a DOCX file into synthetic pages of ~1000 words each.

    Returns same structure as parse_pdf for a uniform downstream pipeline.
    """
    doc = DocxDocument(io.BytesIO(file_bytes))

    # Collect all non-empty paragraph text
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    words = full_text.split()

    # Group into ~1000-word pages
    page_size_words = 1000
    pages = []
    for i in range(0, len(words), page_size_words):
        page_words = words[i : i + page_size_words]
        page_text = " ".join(page_words)
        pages.append({
            "page_number": (i // page_size_words) + 1,
            "text": page_text,
            "images": [],          # DOCX image extraction skipped for brevity
            "char_count": len(page_text),
        })

    return {"pages": pages, "total_pages": len(pages)}
