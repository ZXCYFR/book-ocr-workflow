"""可移植的 PDF 交付工具；不执行 OCR，也不恢复缺失的文字层。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import PIL
import pypdf
from pypdf import PdfReader, PdfWriter

PathLike = str | Path


def sha256(path: PathLike) -> str:
    """分块计算文件指纹。

    Args:
        path: 输入文件路径。
    Returns:
        SHA256 十六进制字符串。
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _reader(path: PathLike) -> PdfReader:
    """读取未加密 PDF，拒绝空文档。

    Args:
        path: 输入路径。
    Returns:
        已完成页树解析的读取器。
    Raises:
        ValueError: 文档加密或没有页面。
    """
    reader = PdfReader(str(path), strict=True)
    if reader.is_encrypted or not reader.pages:
        raise ValueError('不支持加密或零页 PDF')
    return reader


def verify_pdf(pdf: PathLike, expected_sha256: str | None = None,
               expected_pages: int | None = None) -> dict[str, Any]:
    """核对 PDF 可读性、页数和可选指纹；不判定视觉正确性。

    Args:
        pdf: PDF 文件路径。
        expected_sha256: 可选预期指纹。
        expected_pages: 可选预期页数。
    Returns:
        含 ok、sha256、pages、bytes 和 errors 的核验记录。
    """
    path = Path(pdf)
    reader = _reader(path)
    fingerprint = sha256(path)
    errors: list[str] = []
    if expected_sha256 is not None and fingerprint != expected_sha256.lower():
        errors.append('SHA256 不匹配')
    if expected_pages is not None and len(reader.pages) != expected_pages:
        errors.append('页数不匹配')
    return {'ok': not errors, 'sha256': fingerprint, 'pages': len(reader.pages),
            'bytes': path.stat().st_size, 'errors': errors}


def export_text(pdf: PathLike, output_dir: PathLike,
                page_order: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """原子导出现有文字层；不调用 OCR、不保证阅读顺序或排印忠实度。

    Args:
        pdf: 输入 PDF。
        output_dir: 尚不存在的目标目录；拒绝覆盖已有目录。
        page_order: 与 PDF 同序的逐页 uid 和 printed_page；兼容原印页字段。
    Returns:
        写入 manifest.json 的交付记录。
    Raises:
        ValueError: 页序长度、uid 或原印页字段无效。
        FileExistsError: 输出目录已经存在。
    """
    source, target = Path(pdf), Path(output_dir)
    if target.exists():
        raise FileExistsError(f'输出目录已存在：{target}')
    reader = _reader(source)
    if page_order is not None and len(page_order) != len(reader.pages):
        raise ValueError('页序记录数必须等于 PDF 页数')
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, page in enumerate(reader.pages, 1):
        item = page_order[index - 1] if page_order is not None else {}
        uid = item.get('uid', f'page_{index:04d}')
        if not isinstance(uid, str) or not uid.strip() or uid in seen:
            raise ValueError('uid 必须为非空且唯一的字符串')
        seen.add(uid)
        printed = item.get('printed_page', item.get('原印页'))
        if printed is not None and not isinstance(printed, (str, int)):
            raise ValueError('原印页必须为字符串、整数或 null')
        rows.append({'pdf_page': index, 'uid': uid, 'printed_page': printed,
                     'text': page.extract_text() or ''})
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        (staging / 'text.txt').write_text('\n\f\n'.join(row['text'] for row in rows), encoding='utf-8')
        (staging / 'pages.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        manifest = {'schema_version': 1, 'source_name': source.name,
                    'source_sha256': sha256(source), 'pages': len(rows),
                    'empty_text_pages': [row['pdf_page'] for row in rows if not row['text'].strip()],
                    'software': {'pypdf': pypdf.__version__, 'Pillow': PIL.__version__},
                    'extraction_boundary': '仅提取已有文字层；未执行 OCR；空文字页不代表空白页；不保证阅读顺序或视觉一致性。',
                    'files': {name: sha256(staging / name) for name in ('text.txt', 'pages.jsonl')}}
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if target.exists():
            raise FileExistsError(f'输出目录已存在：{target}')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def _publish_pdf(writer: PdfWriter, output: PathLike, expected_pages: int,
                 expected_text: list[str] | None = None) -> dict[str, Any]:
    """在同目录暂存并回读验证，通过硬链接无覆盖发布。

    Args:
        writer: 待写 PDF。
        output: 尚不存在的目标路径。
        expected_pages: 预期页数。
        expected_text: 可选逐页文字层结果。
    Returns:
        核验记录。
    Raises:
        ValueError: 回读页数或文字层不一致。
        FileExistsError: 目标已经存在。
    """
    target = Path(output)
    if target.exists():
        raise FileExistsError(f'输出文件已存在：{target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.pdf', dir=target.parent)
    temp_path = Path(filename)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer.write(stream)
        result = verify_pdf(temp_path, expected_pages=expected_pages)
        if not result['ok']:
            raise ValueError('回读页数与输入不一致')
        if expected_text is not None:
            actual = [page.extract_text() or '' for page in _reader(temp_path).pages]
            if actual != expected_text:
                raise ValueError('转换后文字层发生变化')
        os.link(temp_path, target)
        return result
    finally:
        temp_path.unlink(missing_ok=True)


def merge_pages(page_paths: Sequence[PathLike], output: PathLike) -> dict[str, Any]:
    """按显式顺序合并单页 PDF，不允许缺页、重复路径或覆盖。

    Args:
        page_paths: 按成书顺序排列的单页文件。
        output: 新建输出 PDF 路径。
    Returns:
        合并结果核验记录。
    Raises:
        ValueError: 输入为空、路径重复或输入不是单页。
    """
    paths = [Path(path).resolve(strict=True) for path in page_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('输入必须非空且不能含重复路径')
    writer = PdfWriter()
    for path in paths:
        reader = _reader(path)
        if len(reader.pages) != 1:
            raise ValueError(f'输入必须恰好一页：{path.name}')
        writer.add_page(reader.pages[0])
    return _publish_pdf(writer, output, len(paths))


def convert_monochrome(source: PathLike, output: PathLike,
                       threshold: int = 180) -> dict[str, Any]:
    """将直接嵌入的 JPEG 栅格图转为二值图，保留已有文字层。

    仅处理无透明度、无嵌套 Form、无注释的简单 PDF。矢量图形及文字
    颜色不转换，因此本接口不保证任意彩色 PDF 全页变为黑白。

    Args:
        source: 输入 PDF。
        output: 尚不存在的目标 PDF。
        threshold: 灰度小于此值变黑，其余变白，范围 0 到 255。
    Returns:
        核验记录，含实际替换的唯一图像数及处理边界。
    Raises:
        ValueError: 阈值无效或遇到不支持的 PDF 结构。
    """
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError('阈值必须为 0 到 255 的整数')
    reader = _reader(source)
    original_text = [page.extract_text() or '' for page in reader.pages]
    writer = PdfWriter(clone_from=reader)
    visited: set[tuple[int, int]] = set()
    for page in writer.pages:
        if page.get('/Annots') or page.get('/Group'):
            raise ValueError('不支持含注释或透明组的页面')
        resources_ref = page.get('/Resources')
        resources = resources_ref.get_object() if resources_ref else {}
        states = resources.get('/ExtGState', {}).get_object() if '/ExtGState' in resources else {}
        for reference in states.values():
            state = reference.get_object()
            if state.get('/SMask', '/None') != '/None' or state.get('/ca', 1) != 1 or state.get('/CA', 1) != 1 or state.get('/BM', '/Normal') != '/Normal':
                raise ValueError('不支持透明度或混合模式')
        objects = resources.get('/XObject', {}).get_object() if '/XObject' in resources else {}
        if any(obj.get_object().get('/Subtype') == '/Form' for obj in objects.values()):
            raise ValueError('不支持嵌套 Form 图像')
        for image in page.images:
            ref = image.indirect_reference
            if ref is None:
                raise ValueError('不支持内联图像')
            key = (ref.idnum, ref.generation)
            if key in visited:
                continue
            obj = ref.get_object()
            filters = obj.get('/Filter')
            if filters != '/DCTDecode' and filters != ['/DCTDecode']:
                raise ValueError('仅支持直接嵌入的 JPEG 图像')
            if '/SMask' in obj or '/Mask' in obj or obj.get('/ImageMask') or '/Decode' in obj:
                raise ValueError('不支持透明蒙版、图像蒙版或自定义 Decode')
            if image.image is None or image.image.mode not in ('RGB', 'L'):
                raise ValueError('仅支持 RGB 或灰度 JPEG 图像')
            gray = image.image.convert('L')
            binary = gray.point([0 if value < threshold else 255 for value in range(256)], mode='1')
            image.replace(binary)
            visited.add(key)
    if not visited:
        raise ValueError('输入没有可转换的 JPEG 图像')
    result = _publish_pdf(writer, output, len(reader.pages), original_text)
    result.update({'converted_images': len(visited), 'threshold': threshold,
                   'boundary': '仅将直接嵌入的 JPEG 栅格图二值化；文字和矢量颜色保持原状；未进行视觉核验。'})
    return result
