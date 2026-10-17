"""验证完整归档核验能发现缺失和篡改的成员。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from book_ocr import cli, delivery


class ArchiveTests(unittest.TestCase):
    """用一页合成PDF检查默认项目入口。"""

    def setUp(self) -> None:
        """建立独立、带完整清单的小项目。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / '源码/book_ocr').mkdir(parents=True)
        self.pdf = self.root / 'book.pdf'
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.write(self.pdf)
        self.spec = {'pdf': '../book.pdf', 'manifest': '../文件清单.json',
                     'sha256': delivery.sha256(self.pdf), 'pages': 1}
        (self.root / '源码/项目配置.json').write_text(json.dumps(self.spec))
        (self.root / '文件清单.json').write_text(json.dumps({'files': [
            {'path': 'book.pdf', 'sha256': self.spec['sha256'], 'bytes': self.pdf.stat().st_size}]}))
        self.location = str(self.root / '源码/book_ocr/cli.py')

    def test_complete_then_missing(self) -> None:
        """完整归档通过，成员缺失则失败。"""
        with patch.object(cli, '__file__', self.location):
            self.assertTrue(cli.verify_project()['ok'])
            self.pdf.unlink()
            self.assertFalse(cli.verify_project()['ok'])

    def test_changed_bytes_fail(self) -> None:
        """即使还是可读PDF，文件字节改变也不能当作已采用版本。"""
        with self.pdf.open('ab') as stream:
            stream.write(b'\n% changed\n')
        with patch.object(cli, '__file__', self.location):
            self.assertFalse(cli.verify_project()['ok'])


if __name__ == '__main__':
    unittest.main()
