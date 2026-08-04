import tempfile
import unittest
from pathlib import Path


class LoaderErrorTests(unittest.TestCase):
    def test_load_document_rejects_empty_text_file(self):
        from modules.rag.loader import DocumentLoadError, load_document

        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_path = Path(tmp_dir) / "empty.txt"
            empty_path.write_text("   \n\n", encoding="utf-8")

            with self.assertRaises(DocumentLoadError) as context:
                load_document(empty_path)

            self.assertIn("没有提取到可用文字", str(context.exception))

    def test_load_document_reports_broken_pdf_file(self):
        from modules.rag.loader import DocumentLoadError, load_document

        with tempfile.TemporaryDirectory() as tmp_dir:
            broken_pdf = Path(tmp_dir) / "broken.pdf"
            broken_pdf.write_bytes(b"not a valid pdf")

            with self.assertRaises(DocumentLoadError) as context:
                load_document(broken_pdf)

            self.assertIn("PDF 文件解析失败", str(context.exception))

    def test_load_document_reports_broken_pptx_file(self):
        from modules.rag.loader import DocumentLoadError, load_document

        with tempfile.TemporaryDirectory() as tmp_dir:
            broken_pptx = Path(tmp_dir) / "broken.pptx"
            broken_pptx.write_bytes(b"not a valid pptx")

            with self.assertRaises(DocumentLoadError) as context:
                load_document(broken_pptx)

            self.assertIn("PPTX 文件解析失败", str(context.exception))


if __name__ == "__main__":
    unittest.main()
