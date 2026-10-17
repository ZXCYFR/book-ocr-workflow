"""配置驱动的单页排印入口；不运行 OCR，不授予内容复核结论。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from types import ModuleType

from .workflow import check_uid, load_config, load_order

ENGINE_PATH = Path(__file__).resolve().parents[1] / '排印核心' / 'inplace_lib.py'


def load_engine() -> ModuleType:
    """独立加载历史算法引擎，避免修改模块搜索路径。"""
    spec = importlib.util.spec_from_file_location('book_ocr_inplace_engine', ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载排印引擎：{ENGINE_PATH}')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def preflight(config_path: Path, uid: str, draft: bool = False) -> tuple[ModuleType, Path]:
    """检查配置、依赖与单页输入，不生成或重写文件。

    Args:
        config_path: 工作区配置 JSON。
        uid: 单页标识，不允许目录分隔符。
        draft: 是否准备从 Surya HTML 起草。

    Returns:
        配置后的引擎及工作页目录。
    """
    check_uid(uid)
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    engine = load_engine()
    engine.configure(config, config_path.parent)
    engine.check_runtime()
    if shutil.which('pdftoppm') is None:
        raise RuntimeError('缺少 Poppler 的 pdftoppm，无法完成渲染核验')
    engine.register_fonts()
    if engine.HYPHEN_LANGUAGE:
        import pyphen
        if engine.HYPHEN_LANGUAGE not in pyphen.LANGUAGES:
            raise ValueError(f'不支持的断词语言：{engine.HYPHEN_LANGUAGE}')
    page_dir = engine.page_dir(uid)
    if draft:
        units = load_order(config_path)
        ids = [unit['id'] for unit in units]
        if len(ids) != len(set(ids)) or ids.count(uid) != 1:
            raise ValueError('页清单 ID 重复或缺少指定页面')
        unit = units[ids.index(uid)]
        engine.Page(unit)  # 核对源图和裁剪框。
        raw = engine.load_raw(uid)
        blocks = engine.parse_blocks(raw.get('raw', ''))
        if not blocks:
            raise ValueError('OCR 输入须为含 data-bbox/data-label 块的 Surya HTML')
        for block in blocks:
            box = block['bbox_n']
            if len(box) != 4 or not (0 <= box[0] < box[2] <= 1000
                                    and 0 <= box[1] < box[3] <= 1000):
                raise ValueError('OCR 坐标必须为 0–1000 范围内的有效矩形')
    else:
        for name in ('源页.png', '页面配置.json', '行文本.txt'):
            if not (page_dir / name).is_file():
                raise FileNotFoundError(f'缺少排印输入：{page_dir / name}')
        page_config = json.loads((page_dir / '页面配置.json').read_text(encoding='utf-8'))
        if page_config.get('uid') != uid or not isinstance(page_config.get('blocks'), list):
            raise ValueError('页面配置的 uid 或 blocks 无效')
        validate_lines(page_dir, page_config)
    return engine, page_dir


def validate_lines(page_dir: Path, spec: dict) -> None:
    """拒绝会被历史渲染器截短或忽略的文稿行，防止生成缺字校样。"""
    counts: dict[int, int] = {}
    current = None
    for line in (page_dir / '行文本.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        if line.startswith('#'):
            match = re.match(r'# 块(\d+)', line)
            if match is None:
                raise ValueError(f'无法解析块标题：{line}')
            current = int(match.group(1))
            if current in counts:
                raise ValueError(f'文稿块标题重复：{current}')
            counts[current] = 0
        else:
            if current is None:
                raise ValueError('行文本必须先声明 # 块N 标题')
            counts[current] += 1
    expected = {block['index']: sum(len(col['bands']) for col in block['columns'])
                for block in spec['blocks'] if block['kind'] != 'image'}
    if counts != expected:
        raise ValueError(f'文稿行数与页面行带不一致，拒绝截断文字：{counts} / {expected}')


def render(config_path: Path, uid: str, draft: bool = False,
           rerender: bool = False) -> dict:
    """生成单页校样和技术绑定，保留现有人工复核记录。

    Args:
        config_path: 配置文件路径。
        uid: 页标识。
        draft: 从已有 OCR HTML 起草，仅允许全新工作页。
        rerender: 明确允许重建已有 PDF，不允许重新起草覆盖文稿。

    Returns:
        输出路径、技术检查及输入输出文件校验绑定。
    """
    engine, page_dir = preflight(config_path, uid, draft=draft)
    output = page_dir / '原位排印校样.pdf'
    if output.exists() and not rerender:
        raise FileExistsError(f'校样已存在；明确指定 rerender 才能重建：{output}')
    if draft and page_dir.exists() and any(page_dir.iterdir()):
        raise FileExistsError('起草仅接受全新工作页，禁止覆盖已有文稿或复核记录')
    fill_flags = []
    if draft:
        engine.draft(uid)
        fill_flags = engine.fill_all(uid)
        engine.atomic_write(page_dir / '草稿填充报告.json',
                            json.dumps({'uid': uid, 'flags': fill_flags}, ensure_ascii=False, indent=2))
        if any(any(marker in flag for marker in ('截断', '未耗尽', '墨带不足'))
               for flag in fill_flags):
            raise ValueError('OCR草稿存在未排入文字，已保留文稿与填充报告；请校正后再渲染')
        validate_lines(page_dir, json.loads((page_dir / '页面配置.json').read_text()))
    checks = engine.build(uid)
    binding_path = page_dir / '技术预筛绑定.json'
    binding = json.loads(binding_path.read_text(encoding='utf-8'))
    binding['config_sha256'] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    binding['font_sha256'] = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                              for key, path in engine.FONT_FILES.items()}
    engine.atomic_write(binding_path, json.dumps(binding, ensure_ascii=False, indent=2))
    return {'uid': uid, 'pdf': str(output), 'checks': checks, 'binding': binding,
            'fill_flags': fill_flags,
            'scope': '单页技术渲染；内容和版式仍需独立复核'}
