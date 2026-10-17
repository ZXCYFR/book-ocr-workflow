"""针对页序、OCR边界和审核失效的回归测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from book_ocr import workflow


class WorkflowTests(unittest.TestCase):
    """使用独立临时工作区，不触碰真实书籍。"""

    def setUp(self) -> None:
        """初始化含一页的最小配置。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'book'
        self.config = workflow.init_workspace(self.root, 'example')
        workflow.write_json(self.root / 'page_order.json', [
            {'id': 'page_001', 'scan': 1, 'crop': [0, 0, 20, 30],
             'source_image': 'scans/scan_001.png'}])

    def test_duplicate_and_escape_rejected(self) -> None:
        """页序重复与路径穿越必须明确失败。"""
        rows = workflow.load_order(self.config)
        workflow.write_json(self.root / 'page_order.json', rows * 2)
        with self.assertRaises(ValueError):
            workflow.load_order(self.config)
        with self.assertRaises(ValueError):
            workflow.safe_path(self.root, '../outside')
        with self.assertRaises(ValueError):
            workflow.check_uid('../page')

    def test_import_ocr_and_no_overwrite(self) -> None:
        """接入真实结构并保留数学符号，不覆盖已有OCR。"""
        source = self.root / 'blocks.json'
        workflow.write_json(source, {'engine': 'fixture', 'blocks': [
            {'label': 'Text', 'bbox': [10, 20, 900, 700], 'text': 'δ φ alpha'}]})
        workflow.import_ocr(self.config, 'page_001', source)
        result = json.loads((self.root / 'ocr/page_001.json').read_text())
        self.assertIn('δ φ alpha', result['raw'])
        with self.assertRaises(FileExistsError):
            workflow.import_ocr(self.config, 'page_001', source)

    def test_review_binding_invalidates_after_edit(self) -> None:
        """修改实际文稿后，旧审核不可用于正式合并。"""
        folder = self.root / 'pages/page_001'
        folder.mkdir(parents=True)
        for name in workflow.BINDING_FILES:
            (folder / name).write_text('fixture')
        workflow.write_json(folder / '技术预筛绑定.json',
                            {'uid': 'page_001', 'hashes': {name: workflow.digest(folder / name)
                             for name in workflow.BINDING_FILES}})
        workflow.review(self.config, 'page_001', 'tester', '仅测试绑定，不是实际审核')
        self.assertEqual(len(workflow.checked_pages(self.config)), 1)
        (folder / '行文本.txt').write_text('changed')
        with self.assertRaises(ValueError):
            workflow.checked_pages(self.config)
        with self.assertRaises(ValueError):
            workflow.review(self.config, 'page_001', 'tester', '未重渲染不能通过')
        self.assertEqual(len(workflow.checked_pages(self.config, proof=True)), 1)
        (folder / '原位排印校样.pdf').unlink()
        with self.assertRaises(FileNotFoundError):
            workflow.checked_pages(self.config, proof=True)


if __name__ == '__main__':
    unittest.main()
