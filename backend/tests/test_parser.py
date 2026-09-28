"""
Unit tests — ingestion/parser.py

Tests:
  - parse_pdf: extracts text and images from a minimal PDF (mocked fitz)
  - parse_pdf: returns correct total_pages count
  - parse_pdf: blank pages are included but text is empty string
  - parse_docx: splits text into synthetic ~1000-word pages
  - caption_image_with_gpt4o: calls the Azure OpenAI client with correct payload
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import base64


class TestParsePDF:
    def test_returns_pages_and_total_pages(self):
        """parse_pdf returns the correct structure and page count."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Clinical trial results."
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
        mock_doc.__len__ = MagicMock(return_value=2)

        with patch("fitz.open", return_value=mock_doc):
            from ingestion.parser import parse_pdf
            result = parse_pdf(b"fake-pdf-bytes")

        assert result["total_pages"] == 2
        assert len(result["pages"]) == 2
        assert result["pages"][0]["page_number"] == 1
        assert result["pages"][1]["page_number"] == 2

    def test_page_text_is_stripped(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "   Some text with whitespace.   "
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        with patch("fitz.open", return_value=mock_doc):
            from ingestion.parser import parse_pdf
            result = parse_pdf(b"fake")

        assert result["pages"][0]["text"] == "Some text with whitespace."

    def test_images_extracted(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Text"
        mock_page.get_images.return_value = [(1, 0, 0, 0, 0, 0, 0)]   # xref=1

        fake_img_bytes = b"\x89PNG\r\n"
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.extract_image.return_value = {
            "image": fake_img_bytes,
            "ext": "png",
            "width": 100,
            "height": 100,
        }

        with patch("fitz.open", return_value=mock_doc):
            from ingestion.parser import parse_pdf
            result = parse_pdf(b"fake")

        images = result["pages"][0]["images"]
        assert len(images) == 1
        assert images[0]["ext"] == "png"
        # data should be base64-encoded
        assert base64.b64decode(images[0]["data"]) == fake_img_bytes

    def test_char_count_matches_text(self):
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Hello world"
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

        with patch("fitz.open", return_value=mock_doc):
            from ingestion.parser import parse_pdf
            result = parse_pdf(b"fake")

        assert result["pages"][0]["char_count"] == len("Hello world")


class TestParseDocx:
    def test_splits_into_pages(self):
        """1000-word text should produce at least one page; 2000 words → 2 pages."""
        from unittest.mock import MagicMock, patch

        fake_para = MagicMock()
        # 1100 words = 2 pages (1000 per page)
        fake_para.text = "word " * 1100
        fake_doc = MagicMock()
        fake_doc.paragraphs = [fake_para]

        with patch("docx.Document", return_value=fake_doc):
            from ingestion.parser import parse_docx
            result = parse_docx(b"fake-docx")

        assert result["total_pages"] == 2
        assert result["pages"][0]["page_number"] == 1
        assert result["pages"][1]["page_number"] == 2

    def test_empty_paragraphs_skipped(self):
        fake_para_empty = MagicMock()
        fake_para_empty.text = "   "
        fake_para_real = MagicMock()
        fake_para_real.text = "Actual text here."

        fake_doc = MagicMock()
        fake_doc.paragraphs = [fake_para_empty, fake_para_real]

        with patch("docx.Document", return_value=fake_doc):
            from ingestion.parser import parse_docx
            result = parse_docx(b"fake-docx")

        assert result["total_pages"] == 1
        assert "Actual text here" in result["pages"][0]["text"]


class TestCaptionImageWithGPT4o:
    @pytest.mark.asyncio
    async def test_calls_openai_with_base64_image(self):
        """caption_image_with_gpt4o should send the image as a data URL to GPT-4o."""
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "A bar chart showing Q3 revenue."

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)

        with patch("openai.AsyncAzureOpenAI", return_value=mock_client):
            from ingestion.parser import caption_image_with_gpt4o
            result = await caption_image_with_gpt4o("base64data==", "png")

        assert result == "A bar chart showing Q3 revenue."

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        user_content = call_kwargs["messages"][0]["content"]
        image_part = next(p for p in user_content if p["type"] == "image_url")
        assert "data:image/png;base64,base64data==" in image_part["image_url"]["url"]
