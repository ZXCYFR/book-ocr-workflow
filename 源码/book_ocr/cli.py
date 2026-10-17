"""可移植书籍工作流的命令行入口。"""

import argparse
import json
import subprocess
from pathlib import Path

from . import delivery, workflow


def verify_project() -> dict:
    """核对归档清单的全部文件，并核对本书黑白成品的指纹和页数。"""
    config_path = Path(__file__).resolve().parents[1] / '项目配置.json'
    spec = json.loads(config_path.read_text(encoding='utf-8'))
    manifest_path = (config_path.parent / spec['manifest']).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors = []
    names = set()
    for entry in manifest['files']:
        name = entry['path']
        if name in names:
            raise ValueError(f'文件清单重复：{name}')
        names.add(name)
        path = workflow.safe_path(manifest_path.parent, name)
        if not path.is_file():
            errors.append(f'缺少文件：{name}；请下载完整项目包')
        elif path.stat().st_size != entry['bytes'] or delivery.sha256(path) != entry['sha256']:
            errors.append(f'文件与归档版本不符：{name}')
    if errors:
        return {'ok': False, 'errors': errors, 'files_expected': len(names)}
    result = delivery.verify_pdf(config_path.parent / spec['pdf'], spec['sha256'], spec['pages'])
    result['files_verified'] = len(names)
    result['scope'] = '归档文件身份及PDF页数；不代替内容审核'
    return result


def main() -> None:
    """解析子命令并明确传播失败；不自动下载模型或覆盖成果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='创建新书工作区')
    init.add_argument('--workspace', type=Path, required=True)
    init.add_argument('--book-id', required=True)
    prepare = commands.add_parser('prepare', help='从PDF准备扫描和页序，不运行OCR')
    prepare.add_argument('--config', type=Path, required=True)
    prepare.add_argument('--pdf', type=Path, required=True)
    prepare.add_argument('--dpi', type=int, default=150)
    prepare.add_argument('--layout', choices=['single', 'spread'], default='single')
    prepare.add_argument('--first-page-single', action='store_true')
    ocr = commands.add_parser('import-ocr', help='接入归一化文字块JSON')
    ocr.add_argument('--config', type=Path, required=True)
    ocr.add_argument('--uid', required=True)
    ocr.add_argument('--input', type=Path, required=True)
    render = commands.add_parser('render', help='生成单页草稿，或渲染校正文稿')
    render.add_argument('--config', type=Path, required=True)
    render.add_argument('--uid', required=True)
    render.add_argument('--draft', action='store_true', help='从OCR生成初稿；仍需逐页校正')
    render.add_argument('--rerender', action='store_true', help='明确允许重渲染已有草稿，旧复核自动失效')
    review = commands.add_parser('review', help='记录实际完成的逐页审核，绑定当前文件')
    review.add_argument('--config', type=Path, required=True)
    review.add_argument('--uid', required=True)
    review.add_argument('--reviewer', required=True)
    review.add_argument('--note', required=True)
    merge = commands.add_parser('merge', help='按全部物理页汇编，缺页拒绝')
    merge.add_argument('--config', type=Path, required=True)
    merge.add_argument('--output', type=Path, required=True)
    merge.add_argument('--proof', action='store_true', help='仅生成完整校样，允许未审核页')
    export = commands.add_parser('export', help='从任意PDF文字层导出文本')
    export.add_argument('--pdf', type=Path, help='省略时导出本项目采用的黑白成品')
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--order', type=Path)
    verify = commands.add_parser('verify', help='校验PDF可读性、指纹及页数')
    verify.add_argument('--pdf', type=Path, help='省略时核验完整项目文件及采用成品')
    verify.add_argument('--sha256')
    verify.add_argument('--pages', type=int)
    mono = commands.add_parser('monochrome', help='将受支持的JPEG插图二值化；保持文字层')
    mono.add_argument('--pdf', type=Path, required=True)
    mono.add_argument('--output', type=Path, required=True)
    mono.add_argument('--threshold', type=int, default=180)
    args = parser.parse_args()
    try:
        if args.command == 'init':
            result = {'config': str(workflow.init_workspace(args.workspace, args.book_id))}
        elif args.command == 'prepare':
            result = workflow.prepare(args.config, args.pdf, args.dpi,
                                      args.layout, args.first_page_single)
        elif args.command == 'import-ocr':
            result = workflow.import_ocr(args.config, args.uid, args.input)
        elif args.command == 'render':
            from .rendering import render
            result = render(args.config, args.uid, draft=args.draft, rerender=args.rerender)
        elif args.command == 'review':
            result = workflow.review(args.config, args.uid, args.reviewer, args.note)
        elif args.command == 'merge':
            result = delivery.merge_pages(workflow.checked_pages(args.config, args.proof), args.output)
            result['review_scope'] = '未审核完整校样' if args.proof else '全部页面当前版本复核通过'
        elif args.command == 'export':
            if args.pdf is None:
                config_path = Path(__file__).resolve().parents[1] / '项目配置.json'
                spec = json.loads(config_path.read_text(encoding='utf-8'))
                args.pdf = config_path.parent / spec['pdf']
                args.order = args.order or config_path.parent / spec['page_order']
                verification = delivery.verify_pdf(args.pdf, spec['sha256'], spec['pages'])
                if not verification['ok']:
                    raise ValueError('项目成品与采用版本不一致')
            order = json.loads(args.order.read_text(encoding='utf-8')) if args.order else None
            if order is not None:
                order = [dict(row, uid=row.get('uid', row.get('id'))) for row in order]
            result = delivery.export_text(args.pdf, args.output, order)
        elif args.command == 'verify':
            if args.pdf is None:
                result = verify_project()
            else:
                result = delivery.verify_pdf(args.pdf, args.sha256, args.pages)
            if not result['ok']:
                parser.exit(1, json.dumps(result, ensure_ascii=False) + '\n')
        else:
            result = delivery.convert_monochrome(args.pdf, args.output, args.threshold)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'执行失败：{error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
