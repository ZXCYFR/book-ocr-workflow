"""验证真实单页渲染、路径迁移和书籍规则隔离。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

from book_ocr.rendering import load_engine, render, validate_lines


class RenderingTests(unittest.TestCase):
    """使用本机字体的小样，不读取原书或运行模型。"""

    def setUp(self) -> None:
        """准备一个独立目录和单行源图。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')
        if not self.font.exists():
            self.skipTest('需要 DejaVu Serif 字体完成真实渲染')
        self.config = {'schema_version': 1, 'book_id': 'sample',
                       'paths': {'pages': 'pages', 'scans': 'scans',
                                 'order': 'page_order.json', 'ocr': 'ocr'},
                       'render': {'scale': 0.5, 'fonts': {k: str(self.font) for k in ('R', 'I', 'B', 'BI')}}}
        self.config_path = self.root / 'book.json'
        self.config_path.write_text(json.dumps(self.config))
        self.page = self.root / 'pages' / 'sample'
        self.page.mkdir(parents=True)
        source = Image.new('RGB', (600, 180), 'white')
        ImageDraw.Draw(source).text((30, 35), 'Alpha delta', fill='black',
                                    font=ImageFont.truetype(str(self.font), 30))
        source.save(self.page / '源页.png')
        spec = {'uid': 'sample', 'rules': [], 'blocks': [
            {'kind': 'text', 'label': 'Text', 'index': 0, 'bbox': [20, 30, 300, 90],
             'columns': [{'size_pt': 15, 'guess_slope': 0,
                          'bands': [{'x0': 30, 'x1': 210, 'y0': 40, 'y1': 65,
                                     'baseline_px': 65}]}]}]}
        (self.page / '页面配置.json').write_text(json.dumps(spec))
        (self.page / '行文本.txt').write_text('# 块0 Text\nAlpha delta\n')

    def test_ampersand_preserves_inline_styles(self) -> None:
        """裸与号及XML实体都不应触发整行样式丢失。"""
        engine = load_engine()
        spans = engine.markup_spans("<i>A & B</i> &amp; <b>C</b> &apos;", 12)
        self.assertEqual(''.join(span['text'] for span in spans), "A & B & C '")
        self.assertEqual(spans[0]['font'], 'I')
        self.assertEqual(next(span['font'] for span in spans if span['text'] == 'C'), 'B')

    def test_calibration_registers_fonts_in_fresh_process(self) -> None:
        """全新进程中直接标定字号，不能依赖此前渲染注册字体。"""
        script = """
import json
import sys
from pathlib import Path
from book_ocr.rendering import load_engine
engine = load_engine()
config = Path(sys.argv[1])
engine.configure(json.loads(config.read_text()), config.parent)
words = engine.word_tokens('Alpha delta')
bands = [[{'x0': 30, 'x1': 210, 'y0': 40, 'y1': 65, 'h_core': 25}]] * 2
size, detail = engine.calibrate_size([words, words], bands, 15)
assert detail['method'] == 'dp_fit', detail
assert size > 0
"""
        result = subprocess.run([sys.executable, '-B', '-c', script, str(self.config_path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_render_and_overwrite_guard(self) -> None:
        """真实输出可提取文字，并拒绝隐式覆盖。"""
        result = render(self.config_path, 'sample')
        pdf = PdfReader(result['pdf'])
        self.assertEqual(len(pdf.pages), 1)
        self.assertIn('Alpha', pdf.pages[0].extract_text())
        self.assertEqual(float(pdf.pages[0].mediabox.width), 300)
        self.assertIn('font_sha256', result['binding'])
        with self.assertRaises(FileExistsError):
            render(self.config_path, 'sample')
        review = self.page / '复核记录.json'
        review.write_text('{"note":"old review"}')
        render(self.config_path, 'sample', rerender=True)
        self.assertEqual(review.read_text(), '{"note":"old review"}')

    def test_bad_font_and_uid(self) -> None:
        """缺字体和路径穿越在生成输出前失败。"""
        with self.assertRaises(ValueError):
            render(self.config_path, '../other')
        self.config['render']['fonts']['R'] = 'missing.ttf'
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaises(FileNotFoundError):
            render(self.config_path, 'sample')
        self.assertFalse((self.page / '原位排印校样.pdf').exists())

    def test_extra_text_is_not_silently_dropped(self) -> None:
        """拒绝文字行多于墨带时历史引擎的静默截断。"""
        (self.page / '行文本.txt').write_text('# 块0 Text\nAlpha delta\nExtra line\n')
        with self.assertRaisesRegex(ValueError, '拒绝截断文字'):
            render(self.config_path, 'sample')

    def test_generic_draft_preserves_symbols_and_source_path(self) -> None:
        """新书不套用原书希腊字母转换和书面替换。"""
        (self.root / 'scan.png').write_bytes((self.page / '源页.png').read_bytes())
        (self.root / 'page_order.json').write_text(json.dumps([
            {'id': 'new', 'scan': 1, 'crop': [0, 0, 600, 180], 'source_image': 'scan.png'}]))
        (self.root / 'ocr').mkdir()
        markup = r'<div data-bbox="10 150 800 600" data-label="Text"><p>Alpha \delta \phi</p></div>'
        (self.root / 'ocr' / 'new.json').write_text(json.dumps({'raw': markup}))
        engine = load_engine()
        engine.configure(self.config, self.root)
        engine.register_fonts()
        result = render(self.config_path, 'new', draft=True)
        self.assertEqual(len(PdfReader(result['pdf']).pages), 1)
        spec = (self.root / 'pages' / 'new' / '页面配置.json').read_text()
        self.assertIn('delta', spec)
        self.assertNotIn('♂', spec)
        self.assertNotIn('♀', spec)
        self.assertEqual(engine.hyphen_cuts('щетинками'), [])
        other = self.root / 'other'
        engine.configure(self.config, other)
        self.assertEqual(engine.page_dir('a'), other / 'pages' / 'a')

    def test_fill_warning_blocks_pdf_even_when_line_counts_match(self) -> None:
        """行数匹配也不能掩盖填充丢词，须保留报告且禁止构建 PDF。"""
        engine = load_engine()
        target = self.root / 'pages' / 'truncated'
        flags = ['块0: 最后一行文字截断，剩余词未耗尽']

        def prepare_draft(uid: str) -> None:
            """生成行数与行带一致的文稿，模拟词级截断而非行数缺失。"""
            target.mkdir(parents=True)
            spec = json.loads((self.page / '页面配置.json').read_text())
            spec['uid'] = uid
            (target / '页面配置.json').write_text(json.dumps(spec))
            (target / '行文本.txt').write_text('# 块0 Text\nAlpha\n')
            validate_lines(target, spec)

        with patch('book_ocr.rendering.preflight', return_value=(engine, target)), \
                patch.object(engine, 'draft', side_effect=prepare_draft), \
                patch.object(engine, 'fill_all', return_value=flags), \
                patch.object(engine, 'build') as build:
            with self.assertRaisesRegex(ValueError, '存在未排入文字'):
                render(self.config_path, 'truncated', draft=True)
            build.assert_not_called()
        report = json.loads((target / '草稿填充报告.json').read_text())
        self.assertEqual(report, {'uid': 'truncated', 'flags': flags})
        self.assertTrue((target / '行文本.txt').is_file())
        self.assertFalse((target / '原位排印校样.pdf').exists())


if __name__ == '__main__':
    unittest.main()
