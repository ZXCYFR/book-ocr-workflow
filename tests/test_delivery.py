"""使用合成 PDF 检查发布与失败路径，不代表真实成书视觉核验。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from pypdf import PdfWriter
from reportlab.pdfgen.canvas import Canvas

MODULE = Path(__file__).resolve().parents[1] / '源码/book_ocr/delivery.py'
spec = importlib.util.spec_from_file_location('delivery', MODULE)
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


class DeliveryTests(unittest.TestCase):
    """验证交付函数的真实序列化与无覆盖行为。"""

    def setUp(self) -> None:
        """为每个测试建立独立临时目录。"""
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        """清理本测试创建的文件。"""
        self.temp.cleanup()

    def make_pdf(self, name: str = 'input.pdf', pages: int = 1,
                 image_format: str | None = None) -> Path:
        """生成带文字及可选栅格图的 PDF。

        Args:
            name: 输出文件名。
            pages: 页数。
            image_format: 可选 JPEG 或 PNG。
        Returns:
            PDF 路径。
        """
        target = self.root / name
        canvas = Canvas(str(target))
        for index in range(pages):
            canvas.drawString(10, 20, f'page {index + 1}')
            if image_format:
                image_path = self.root / f'image.{image_format.lower()}'
                Image.new('RGB', (20, 20), (170, 170, 170)).save(image_path, image_format)
                canvas.drawImage(str(image_path), 20, 40, 100, 100)
            canvas.showPage()
        canvas.save()
        return target

    def test_export_and_fingerprint(self) -> None:
        """检验导出内容、旧原印页字段及指纹。"""
        source = self.make_pdf(pages=2)
        target = self.root / 'export'
        manifest = delivery.export_text(source, target, [{'uid': 'a', '原印页': 1}, {'uid': 'b', 'printed_page': 2}])
        rows = [json.loads(line) for line in (target / 'pages.jsonl').read_text().splitlines()]
        self.assertEqual(rows[0]['printed_page'], 1)
        self.assertEqual(manifest['source_sha256'], delivery.sha256(source))
        self.assertIn('page 2', (target / 'text.txt').read_text())
        self.assertFalse(delivery.verify_pdf(source, expected_pages=1)['ok'])
        with self.assertRaises(FileExistsError):
            delivery.export_text(source, target)

    def test_invalid_order_is_not_published(self) -> None:
        """拒绝页数错误和重复 uid，且不留下输出。"""
        source = self.make_pdf(pages=2)
        target = self.root / 'export'
        for order in ([{'uid': 'a'}], [{'uid': 'a'}, {'uid': 'a'}]):
            with self.assertRaises(ValueError):
                delivery.export_text(source, target, order)
        self.assertFalse(target.exists())

    def test_default_labels_and_failed_write_cleanup(self) -> None:
        """验证默认页标识及写入失败不会留下目标或临时数据。"""
        source = self.make_pdf()
        target = self.root / 'export'
        with patch.object(Path, 'write_text', side_effect=OSError('模拟磁盘写入失败')):
            with self.assertRaises(OSError):
                delivery.export_text(source, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob('.export.*')), [])
        delivery.export_text(source, target)
        row = json.loads((target / 'pages.jsonl').read_text())
        self.assertEqual(row['uid'], 'page_0001')

    def test_failed_pdf_write_cleanup(self) -> None:
        """验证 PDF 写出失败不发布文件且清理本次暂存。"""
        source = self.make_pdf()
        target = self.root / 'merged.pdf'
        with patch.object(PdfWriter, 'write', side_effect=OSError('模拟磁盘写入失败')):
            with self.assertRaises(OSError):
                delivery.merge_pages([source], target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob('.merged.pdf.*')), [])

    def test_merge_strict_inputs(self) -> None:
        """检验缺页、多页、重复输入及禁止覆盖。"""
        first, second = self.make_pdf('a.pdf'), self.make_pdf('b.pdf')
        target = self.root / 'merged.pdf'
        self.assertEqual(delivery.merge_pages([first, second], target)['pages'], 2)
        old_hash = delivery.sha256(target)
        with self.assertRaises(FileExistsError):
            delivery.merge_pages([first], target)
        self.assertEqual(delivery.sha256(target), old_hash)
        with self.assertRaises(ValueError):
            delivery.merge_pages([first, first], self.root / 'duplicate.pdf')
        with self.assertRaises(ValueError):
            delivery.merge_pages([target], self.root / 'multi.pdf')
        with self.assertRaises(FileNotFoundError):
            delivery.merge_pages([self.root / 'missing.pdf'], self.root / 'bad.pdf')
        self.assertFalse((self.root / 'bad.pdf').exists())

    def test_jpeg_conversion_preserves_text(self) -> None:
        """检验支持的直接 JPEG 与文字层保持。"""
        # Pillow 生成不含 ASCII85 外层编码的直接 JPEG PDF。
        image = Image.new('RGB', (20, 20), (170, 170, 170))
        image_pdf = self.root / 'image.pdf'
        image.save(image_pdf, 'PDF')
        writer = PdfWriter(clone_from=str(image_pdf))
        from pypdf import PdfReader
        writer.pages[0].merge_page(PdfReader(self.make_pdf()).pages[0])
        source = self.root / 'source.pdf'
        writer.write(source)
        result = delivery.convert_monochrome(source, self.root / 'mono.pdf')
        self.assertTrue(result['ok'])
        self.assertEqual(result['converted_images'], 1)
        with self.assertRaises(FileExistsError):
            delivery.convert_monochrome(source, self.root / 'mono.pdf')

    def test_unsupported_image_is_not_published(self) -> None:
        """PNG 与无图页面明确失败，不留下半成品。"""
        target = self.root / 'mono.pdf'
        for source in (self.make_pdf('png.pdf', image_format='PNG'), self.make_pdf('text.pdf')):
            with self.assertRaises(ValueError):
                delivery.convert_monochrome(source, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.glob('.mono.pdf.*')), [])


if __name__ == '__main__':
    unittest.main()
