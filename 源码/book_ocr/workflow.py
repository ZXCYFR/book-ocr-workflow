"""管理新书页序、OCR输入及绑定实际文件的复核记录。"""

import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader

from .delivery import sha256 as digest

BINDING_FILES = ('源页.png', '页面配置.json', '行文本.txt', '原位排印校样.pdf',
                 '程序验收.json', '逐行坐标样式.json')


def write_json(path: Path, value: Any) -> None:
    """在同一目录原子更新JSON；调用方负责避免覆盖已采用成果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False, suffix='.tmp') as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_path(base: Path, value: str) -> Path:
    """解析工作区内相对路径，拒绝路径穿越和绝对路径。"""
    path = Path(value)
    resolved = (base / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(base.resolve()):
        raise ValueError(f'工作区文件必须使用区内相对路径：{value}')
    return resolved


def check_uid(uid: str) -> str:
    """校验页标识，避免页名被解释成目录或路径。"""
    if not isinstance(uid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', uid):
        raise ValueError(f'无效页ID：{uid}')
    return uid


def load_config(path: Path) -> dict:
    """读取版本1工作区配置，并验证必需目录。"""
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('不支持的配置版本')
    for name in ('pages', 'scans', 'order', 'ocr'):
        safe_path(path.resolve().parent, config['paths'][name])
    return config


def load_order(config_path: Path) -> list[dict]:
    """读取全部有序物理页，拒绝重复UID、空清单或越界路径。"""
    config = load_config(config_path)
    base = config_path.resolve().parent
    order = json.loads(safe_path(base, config['paths']['order']).read_text(encoding='utf-8'))
    if not isinstance(order, list) or not order:
        raise ValueError('页序必须是非空列表')
    ids = [check_uid(row['id']) for row in order]
    if len(ids) != len(set(ids)):
        raise ValueError('页序含重复ID')
    for row in order:
        safe_path(base, row['source_image'])
        box = row['crop']
        if (len(box) != 4 or any(type(x) is not int for x in box)
                or min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError(f'裁剪框无效：{row["id"]}')
    return order


def init_workspace(workspace: Path, book_id: str) -> Path:
    """初始化一本新书；拒绝使用已有目录，避免混入历史结果。"""
    check_uid(book_id)
    workspace.mkdir(parents=True, exist_ok=False)
    config = {'schema_version': 1, 'book_id': book_id,
              'paths': {'scans': 'scans', 'order': 'page_order.json',
                        'ocr': 'ocr', 'pages': 'pages'},
              'render': {'scale': 0.24, 'fonts': {}, 'hyphen_language': None,
                         'book_specific_rules': False}}
    path = workspace / 'book.json'
    write_json(path, config)
    return path


def prepare(config_path: Path, pdf: Path, dpi: int = 150, layout: str = 'single',
            first_page_single: bool = False) -> dict:
    """将PDF渲染成扫描图并建立物理页清单；不识别文字。

    Args:
        config_path: 工作区配置。
        pdf: 明确指定的源PDF。
        dpi: 扫描渲染分辨率，决定像素到PDF点的换算。
        layout: single为一张一页，spread按中线左右拆页。
        first_page_single: 对开扫描时保留首张封面整页。

    Returns:
        扫描张数、物理页数与源文件指纹。
    """
    config = load_config(config_path)
    base = config_path.resolve().parent
    scans = safe_path(base, config['paths']['scans'])
    order_file = safe_path(base, config['paths']['order'])
    if scans.exists() or order_file.exists():
        raise FileExistsError('扫描目录或页序已存在，请新建工作区')
    if not 72 <= dpi <= 600 or layout not in ('single', 'spread'):
        raise ValueError('dpi须为72—600，layout须为single或spread')
    if not shutil.which('pdftoppm'):
        raise RuntimeError('缺少Poppler的pdftoppm，请在项目环境中安装')
    count = len(PdfReader(pdf).pages)
    if count == 0:
        raise ValueError('源PDF没有页面')
    source_sha = digest(pdf)
    order = []
    # 全部扫描完成后再发布目录；失败不遗留可误认为完整的页序。
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=base) as temporary:
        stage = Path(temporary)
        for number in range(1, count + 1):
            prefix = stage / f'scan_{number:03d}'
            subprocess.run(['pdftoppm', '-f', str(number), '-l', str(number),
                            '-singlefile', '-r', str(dpi), '-png', str(pdf.resolve()),
                            str(prefix)], check=True, capture_output=True, text=True)
            with Image.open(prefix.with_suffix('.png')) as image:
                width, height = image.size
            halves = [('whole', [0, 0, width, height])]
            if layout == 'spread' and not (number == 1 and first_page_single):
                halves = [('left', [0, 0, width // 2, height]),
                          ('right', [width // 2, 0, width, height])]
            for side, crop in halves:
                order.append({'id': f'scan_{number:03d}_{side}', 'scan': number,
                              'crop': crop, 'printed_page': None,
                              'source_image': str(Path(config['paths']['scans']) / prefix.with_suffix('.png').name)})
        if digest(pdf) != source_sha:
            raise ValueError('渲染期间源PDF发生变化，请重新准备')
        scans.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(scans)
    write_json(order_file, order)
    config['render']['scale'] = 72 / dpi
    config['source'] = {'filename': pdf.name, 'sha256': source_sha, 'dpi': dpi,
                        'layout': layout, 'first_page_single': first_page_single}
    write_json(config_path, config)
    return {'scans': count, 'pages': len(order), 'source_sha256': source_sha}


def import_ocr(config_path: Path, uid: str, source: Path) -> dict:
    """接入规范化OCR块；不执行模型推理，不把文本自动认定为校正稿。

    Args:
        config_path: 工作区配置。
        uid: 页序中已有的物理页ID。
        source: JSON对象，含blocks列表，每块含label、bbox、text；bbox为0—1000。

    Returns:
        导入页ID、块数和输入指纹。
    """
    check_uid(uid)
    config = load_config(config_path)
    base = config_path.resolve().parent
    if uid not in {row['id'] for row in load_order(config_path)}:
        raise ValueError('页ID不在页序清单中')
    target = safe_path(base, config['paths']['ocr']) / f'{uid}.json'
    if target.exists():
        raise FileExistsError('该页OCR已存在；请保留原始记录并在页面文稿中校正')
    data = json.loads(source.read_text(encoding='utf-8'))
    blocks = data['blocks']
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('OCR blocks必须为非空列表')
    labels = {'Text', 'Caption', 'Section-Header', 'Page-Header', 'Page-Footer',
              'Footnote', 'List-Item', 'Page-Number', 'Image'}
    parts = []
    for block in blocks:
        label, box, text = block['label'], block['bbox'], block.get('text', '')
        if label not in labels or not isinstance(text, str):
            raise ValueError('不支持的OCR块类型或文本格式')
        if (len(box) != 4 or any(type(v) is not int or not 0 <= v <= 1000 for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('OCR bbox必须为有效的0—1000整数归一化坐标')
        # 文本作为文字处理，避免OCR字符串被当成任意HTML结构。
        paragraphs = ''.join(f'<p>{html.escape(line)}</p>' for line in text.splitlines())
        parts.append(f'<div data-bbox="{" ".join(map(str, box))}" data-label="{label}">{paragraphs}</div>')
    record = {'raw': ''.join(parts), 'uid': uid, 'input_sha256': digest(source),
              'engine': data.get('engine', '未提供'), 'status': 'ocr_candidate'}
    write_json(target, record)
    return {'uid': uid, 'blocks': len(blocks), 'input_sha256': record['input_sha256']}


def review(config_path: Path, uid: str, reviewer: str, note: str) -> dict:
    """记录操作者明确作出的审核结论并绑定六项文件；不会代替实际目视审核。"""
    if not reviewer.strip() or not note.strip():
        raise ValueError('必须记录复核人和具体复核说明')
    config = load_config(config_path)
    if check_uid(uid) not in {row['id'] for row in load_order(config_path)}:
        raise ValueError('页ID不在清单中')
    folder = safe_path(config_path.resolve().parent, config['paths']['pages']) / uid
    hashes = {name: digest(folder / name) for name in BINDING_FILES}
    technical = json.loads((folder / '技术预筛绑定.json').read_text(encoding='utf-8'))
    if technical.get('uid') != uid or technical.get('hashes') != hashes:
        raise ValueError('页面输入与实际PDF的技术绑定不一致，请先重新渲染再复核')
    record = {'uid': uid, 'status': 'approved', 'reviewer': reviewer,
              'note': note, 'bindings': hashes}
    write_json(folder / '复核记录.json', record)
    return record


def checked_pages(config_path: Path, proof: bool = False) -> list[Path]:
    """按完整页序检查单页PDF，正式汇编须通过当前版本复核绑定。"""
    config = load_config(config_path)
    base = config_path.resolve().parent
    folder = safe_path(base, config['paths']['pages'])
    paths = []
    for row in load_order(config_path):
        uid = row['id']
        page = folder / uid
        pdf = page / '原位排印校样.pdf'
        if not pdf.is_file():
            raise FileNotFoundError(f'缺页，拒绝部分汇编：{uid}')
        if not proof:
            record = json.loads((page / '复核记录.json').read_text(encoding='utf-8'))
            actual = {name: digest(page / name) for name in BINDING_FILES}
            if (record.get('uid') != uid or record.get('status') != 'approved'
                    or record.get('bindings') != actual):
                raise ValueError(f'未通过当前版本复核：{uid}')
        paths.append(pdf)
    return paths
