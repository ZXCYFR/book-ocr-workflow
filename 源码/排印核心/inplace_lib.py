# -*- coding: utf-8 -*-
"""配置驱动的逐页排印核心。

输入为扫描裁页、规范化OCR块，以及可编辑的行文本和版式配置；
输出单页PDF、技术检查与对照图。OCR推理及内容复核由外部完成。
书籍专用文字裁定仅在显式启用book_specific_rules时应用。
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

LIB_DIR = Path(__file__).resolve().parent
ROOT = LIB_DIR
SCAN_DIR = ROOT / 'scans'
ORDER_PATH = ROOT / 'page_order.json'
RAW_DIR = ROOT / 'ocr'
REPAIR_DIR = ROOT / 'ocr_repairs'
BATCH_ROOT = ROOT
PAGES_ROOT = ROOT / 'pages'
REGISTRY = ROOT / '失败登记.jsonl'
FONT_FILES: dict[str, Path] = {}
SCALE = 0.24
INK_THR = 150
PAGE_PT = (433.2, 638.4)
EXCLUDE: set[str] = set()
BOOK_SPECIFIC_RULES = False
HYPHEN_LANGUAGE: str | None = None


def configure(config: dict, base_dir: Path) -> None:
    """按工作区配置解析路径和排印参数，不沿用其他书籍的规则。

    Args:
        config: 含 paths 和 render 的书籍配置。
        base_dir: 配置文件所在目录。
    """
    global ROOT, SCAN_DIR, ORDER_PATH, RAW_DIR, REPAIR_DIR, BATCH_ROOT
    global PAGES_ROOT, REGISTRY, FONT_FILES, SCALE, BOOK_SPECIFIC_RULES
    global HYPHEN_LANGUAGE, _PYRU
    ROOT = base_dir.resolve()
    paths = config.get('paths', {})
    settings = config.get('render', {})
    SCAN_DIR = (ROOT / paths.get('scans', 'scans')).resolve()
    ORDER_PATH = (ROOT / paths.get('order', 'page_order.json')).resolve()
    RAW_DIR = (ROOT / paths.get('ocr', 'ocr')).resolve()
    REPAIR_DIR = (ROOT / paths.get('ocr_repairs', 'ocr_repairs')).resolve()
    BATCH_ROOT = ROOT
    PAGES_ROOT = (ROOT / paths.get('pages', 'pages')).resolve()
    REGISTRY = ROOT / '失败登记.jsonl'
    SCALE = float(settings.get('scale', 0.24))
    if not math.isfinite(SCALE) or SCALE <= 0:
        raise ValueError('render.scale 必须为有限正数')
    FONT_FILES = {key: (ROOT / value).resolve()
                  for key, value in settings.get('fonts', {}).items()}
    BOOK_SPECIFIC_RULES = settings.get('book_specific_rules', False)
    if not isinstance(BOOK_SPECIFIC_RULES, bool):
        raise ValueError('book_specific_rules 必须为布尔值')
    HYPHEN_LANGUAGE = settings.get('hyphen_language')
    _PYRU = None
    _PAGE_CACHE.clear()


TEXT_LABELS = {'Text', 'Caption', 'Section-Header', 'List-Group', 'Page-Footer',
               'Footnote', 'Header', 'Subheader', 'List-Item', 'Page-Number'}
DIV_RE = re.compile(r'<div\s+data-bbox="([^"]+)"\s+data-label="([^"]+)">(.*?)</div>', re.S)
P_RE = re.compile(r'<p>(.*?)</p>', re.S)
KEEP_TAGS = ('i', 'b', 'sub', 'sup')

# ---------------------------------------------------------------- 基础数据


def load_units() -> list[dict]:
    return json.loads(ORDER_PATH.read_text())


def queue_ids() -> list[str]:
    """剩余待制作物理页（按页序，剔除已完成与进行中）。"""
    return [u['id'] for u in load_units() if u['id'] not in EXCLUDE]


def load_raw(uid: str) -> dict:
    return json.loads(raw_path(uid).read_text())


def raw_path(uid: str) -> Path:
    p = REPAIR_DIR / f'{uid}.json'
    return p if p.exists() else RAW_DIR / f'{uid}.json'


def load_corrections(uid: str) -> list[dict]:
    out = []
    if not BOOK_SPECIFIC_RULES:
        return out
    for name in ('最终自动校正', '第二轮校正', '第三轮校正'):
        f = LIB_DIR / f'{name}.json'
        if not f.exists():
            continue
        for r in json.loads(f.read_text()):
            if r.get('物理页') == uid and r.get('校正') and r.get('原文') \
                    and str(r.get('判定', '采纳')) == '采纳':
                out.append(r)
    return out


_PAGE_CACHE: dict[int, Image.Image] = {}


class Page:
    """一物理页的图像与灰度缓存。"""

    def __init__(self, unit: dict):
        self.unit = unit
        self.uid = unit['id']
        x0, y0, x1, y1 = unit['crop']
        source = ((ROOT / unit['source_image']).resolve() if unit.get('source_image')
                  else SCAN_DIR / f"scan_{unit['scan']:03d}.png")
        if source not in _PAGE_CACHE:
            with Image.open(source) as image:
                _PAGE_CACHE[source] = image.copy()
        img = _PAGE_CACHE[source]
        if not (0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height):
            raise ValueError(f'{self.uid}: 裁剪框超出源图范围')
        self.rgb = img.convert('RGB').crop((x0, y0, x1, y1))
        self.gray = np.asarray(self.rgb.convert('L'))
        self.size = self.rgb.size  # (w,h)


def parse_blocks(raw_html: str) -> list[dict]:
    """Surya HTML → 块列表：label、bbox（像素）、段落（保留 i/b/sub/sup 的行内标记）。"""
    blocks = []
    for bbox, label, body in DIV_RE.findall(raw_html):
        n = [int(v) for v in bbox.split()]
        ps = P_RE.findall(body)
        texts = [clean_markup(t) for t in ps] if ps else [clean_markup(body)]
        blocks.append({'label': label, 'bbox_n': n,
                       'paragraphs': [t for t in texts if t]})
    return blocks


def clean_markup(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r'<(?!/?(?:i|b|sub|sup)\b)[^>]*>', '', s)
    return re.sub(r'[ \t]+', ' ', s).strip()


def block_px(b: dict, size: tuple[int, int]) -> list[int]:
    w, h = size
    n = b['bbox_n']
    return [n[0] * w // 1000, n[1] * h // 1000, n[2] * w // 1000, n[3] * h // 1000]

# ---------------------------------------------------------------- 行带与测量


def shear_profile(gray: np.ndarray, box: list[int], slope: float) -> np.ndarray:
    """批8质检修正：按斜率剪切后的行投影（倾斜页水平投影会粘行）。"""
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1] < INK_THR
    h, w = sub.shape
    xc = w / 2.0
    shifts = np.rint(slope * (np.arange(w) - xc)).astype(int)
    prof = np.zeros(h + w, dtype=np.int64)
    for x in range(0, w, 8):  # 8px 列步进足够稳
        col = sub[:, x]
        sh = shifts[x]
        if sh >= 0:
            prof[sh:sh + h] += col
        else:
            prof[0:h + sh] += col[-sh:] if sh < 0 else col
    return prof


def _merge_slices(out: list[dict]) -> list[dict]:
    """批11质检修正：单物理行被纵切成上下两片时合并（间隙<=8px 且 x 界重合>=0.9），
    块尾小片（高<0.5×前带）并入前带。"""
    merged = []
    for bd in out:
        pv = merged[-1] if merged else None
        if pv is not None and (bd['y0'] - pv['y1']) <= 8:
            ov = (min(pv['x1'], bd['x1']) - max(pv['x0'], bd['x0'])) \
                 / max(1, min(pv['x1'] - pv['x0'], bd['x1'] - bd['x0']))
            if ov >= 0.9:
                pv['y1'] = bd['y1']
                pv['x0'] = min(pv['x0'], bd['x0'])
                pv['x1'] = max(pv['x1'], bd['x1'])
                pv['h_core'] = max(pv['h_core'], bd['h_core'])
                continue
            if (bd['y1'] - bd['y0']) < 0.5 * (pv['y1'] - pv['y0']):
                pv['y1'] = bd['y1']
                continue
        merged.append(bd)
    return merged


def _band_from_rows(sub: np.ndarray, counts_raw: np.ndarray, a: int, b: int,
                    x0: int, y0: int) -> dict | None:
    """批13共性1：按行区间 [a,b) 重建墨带（x 界簇剔除 + 点线豁免 + 核心墨高）。"""
    if b - a < 4:
        return None
    colsum = sub[a:b].sum(0)
    on = np.flatnonzero(colsum >= 1)
    if not len(on):
        return None
    clusters, cs = [], [on[0]]
    for c0, c1 in zip(on[:-1], on[1:]):
        if c1 - c0 > 30:
            clusters.append(cs)
            cs = [c1]
        else:
            cs.append(c1)
    clusters.append(cs)
    big = max(clusters, key=len)
    keep = [cl for cl in clusters
            if len(cl) >= 15 or abs(np.mean(cl) - np.mean(big)) <= 80]
    if len(clusters) >= 5:
        starts = np.array([cl[0] for cl in clusters])
        gaps = np.diff(starts)
        if float(np.mean(gaps)) > 25 and float(np.std(gaps)) < 0.25 * float(np.mean(gaps)):
            keep = clusters
    kept = np.concatenate(keep) if keep else big
    seg = counts_raw[a:b]
    core = int((seg >= max(float(np.max(seg)) * 0.35, 3.0)).sum()) or (b - a)
    return {'y0': a + y0, 'y1': b + y0, 'x0': int(kept.min()) + x0,
            'x1': int(kept.max()) + 1 + x0, 'h_core': core}


def _drop_fragments(bands: list[dict]) -> list[dict]:
    """批13共性1c：碎带甄别——h_core<8px 且与邻带叠印/紧贴的碎带删除
    （上行降部/杂墨碎片把一行劈成两行）；点线块（中位核心墨高<10px）豁免；
    带<4 个的块统计不可靠不动（scan_032_right 块8 误删教训）。
    批15共性2：紧贴阈值放宽到 0.35×行距（原6px漏网降部墨尾），并加 y 叠印判定。"""
    if len(bands) < 4:
        return bands
    med_c = float(np.median([bd['h_core'] for bd in bands]))
    if med_c < 10:
        return bands
    pitch = _pitch_of([bd['y0'] for bd in bands])
    near_px = max(6, min(int(0.35 * pitch), 10)) if pitch else 6
    out = []
    for i, bd in enumerate(bands):
        prev = out[-1] if out else None
        nxt = bands[i + 1] if i + 1 < len(bands) else None
        if bd['h_core'] < 8:
            overlap_prev = prev and bd['y0'] < prev['y1']          # 批15：y 叠印
            overlap_next = nxt and nxt['y0'] < bd['y1']
            near_prev = prev and (bd['y0'] - prev['y1']) <= near_px
            near_next = nxt and (nxt['y0'] - bd['y1']) <= near_px
            # 批17共性3：不叠印型降部/尘点碎带——距上带<0.35×行距
            if overlap_prev or overlap_next or near_prev or near_next:
                continue
        # 批19共性1：降部假带——起点落入上带下 30% 段（基线之下）且无 x 高核墨
        # （100_right 2151-2166 15px 降部带、100_left h_core=2/3 型漏检）
        if prev and bd['h_core'] < 10 \
                and bd['y0'] > prev['y0'] + 0.70 * (prev['y1'] - prev['y0']):
            continue
        # 批20共性7：伪带剔除——y 紧贴上带(叠印或距上带底<0.4×行距)且核墨<12
        # 且(核墨宽<上带50% 或 x 完全落入上带区间)。首版②无 y 邻近门，误删
        # 真实窄行致 079_left 块10/019_right 块4 回归变差——真实窄行距上带整行距。
        if prev and pitch and bd['h_core'] < 12 \
                and (bd['y0'] - prev['y1']) < 0.4 * pitch:
            w_prev = prev['x1'] - prev['x0']
            narrow = w_prev > 0 and (bd['x1'] - bd['x0']) < 0.5 * w_prev
            inside = bd['x0'] >= prev['x0'] and bd['x1'] <= prev['x1']
            if narrow or inside:
                continue
        out.append(bd)
    return out


def _pitch_of(tops: list[int]) -> float:
    """批13共性1：行距=带顶差最小值，需代表性（>=2 个且 >=20% 顶差 <=1.25×估计），
    不具代表性返回 0（页级先验回退由 _resplit_merged 按字核门槛受控使用）。"""
    diffs = np.diff(sorted(tops))
    if not len(diffs):
        return 0.0
    p = float(np.min(diffs))
    if p >= 14:
        n_near = int((diffs <= 1.25 * p).sum())
        if n_near >= 2 and n_near >= 0.2 * len(diffs):
            return p
    return 0.0


def _valley_cuts(counts_raw: np.ndarray, a: int, b2: int,
                 min_piece: int = 18, calm_frac: float = 0.15) -> list[int]:
    """批13共性1a：无行距先验时的谷点切位——带内行墨<12%峰值的连续>=3行平静段中点；
    切后每段须>=min_piece（半个行高），防 разрядка/大字单行被内部稀疏行劈碎。"""
    seg = counts_raw[a:b2]
    if (b2 - a) < 60 or float(seg.max()) < 5:
        return []
    calm = seg < max(1.0, calm_frac * float(seg.max()))
    cuts, i = [], 4
    while i < len(seg) - 4:
        if calm[i]:
            j = i
            while j < len(seg) - 4 and calm[j]:
                j += 1
            if j - i >= 3:
                cuts.append(a + (i + j) // 2)
            i = j
        else:
            i += 1
    cuts = cuts[:7]
    # 段高过滤：任何一侧段 < min_piece 的切点回退删除
    kept, prev = [], a
    for c in cuts:
        if c - prev >= min_piece and b2 - c >= min_piece:
            kept.append(c)
            prev = c
    return kept


def _resplit_merged_pass(gray: np.ndarray, box: list[int], bands: list[dict],
                        page_pitch: float = 0.0, core_gate: float = 0.0) -> list[dict]:
    """批13共性1a：并带拆分——带高>1.4×行距(带顶差最小值)时按 y 投影谷点强制拆分；
    行距先验不可得（单带块/无代表性顶差）时用 _valley_cuts 平静段谷点拆。"""
    if not bands:
        return bands
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1] < INK_THR
    counts_raw = sub.sum(1).astype(float)
    pitch = _pitch_of([bd['y0'] for bd in bands]) if len(bands) >= 2 else 0.0
    prior_used = False
    if pitch < 14 and 14 <= page_pitch <= 120:
        # 批15共性1：页级行距先验回退——整块并带时局部估计失效；字核门槛
        # （中位 h_core >= 0.35×先验）把脚注/图注等小字块排除（032_right 块8/23/33 教训）
        med_c = float(np.median([bd['h_core'] for bd in bands]))
        if med_c >= 0.35 * page_pitch:
            pitch = page_pitch
            prior_used = True
    out = []
    heights_all = [bd['y1'] - bd['y0'] for bd in bands]
    for bd in bands:
        # 批21回归：二趟起加字核门——单行带 h_core 不超行距1.5倍，复检只针对
        # 稠密并带（109_right 块9 残段 h_core=2.3×行距）；合法高行不再被网格误拆
        _p_eff = max(pitch, page_pitch, 14.0)
        if core_gate > 0:
            # 批21回归：点线签名门——峰值行墨<0.25×带宽=点线引导行（每行零星墨点；
            # 正文行墨≈0.5-0.7×带宽）。080_right 块5栏3 点线巨型带由此跳过二趟
            _seg2 = counts_raw[bd['y0'] - y0:bd['y1'] - y0]
            _dotted = len(_seg2) and float(_seg2.max()) < 0.25 * (bd['x1'] - bd['x0'])
            if (bd.get('h_core', 0) < core_gate * _p_eff
                    or (bd['y1'] - bd['y0']) < 2.8 * _p_eff or _dotted):
                out.append(bd)
                continue
        a, b2 = bd['y0'] - y0, bd['y1'] - y0
        cuts: list[int] = []
        mega = pitch >= 14 and bd['h_core'] > 2.5 * pitch  # 批16共性1③：并生带（多行检索/点线）
        if pitch >= 14:
            n_expect = min(12, int(round((b2 - a) / pitch)))  # 批17: 6格截8行带(089_left块4)；批18质检发现曾静默丢失，已验证落盘
            if prior_used and n_expect >= 2 and not mega:
                # 批15回归：页先验隐含行距校验——带高/(0.75n+0.25) 与先验差>18% 即弃拆
                # （018_left 块4 实距37px 被先验51 按 n=2 劈成两半的教训）
                p_impl = (b2 - a) / (0.75 * n_expect + 0.25)
                tol_impl = 0.25 if n_expect == 2 else 0.18  # 批17: 2行带含升降部余量，p_impl 系统性偏大
                if abs(p_impl - pitch) > tol_impl * pitch:
                    n_expect = 0
            if (b2 - a) > 1.4 * pitch and n_expect >= 2:
                seg = counts_raw[a:b2]
                seg_max = float(seg.max())
                ink_thr2 = 0.62 if mega else 0.45  # 并生带点线行墨低，谷阈值放宽
                cuts = [a]
                for i in range(1, n_expect):
                    center = a + int((b2 - a) * i / n_expect)
                    lo = max(a + 6, center - int(pitch * 0.3))
                    hi = min(b2 - 6, center + int(pitch * 0.3))
                    if hi <= lo:
                        continue
                    v = lo + int(np.argmin(seg[lo - a:hi - a]))
                    # 批15回归：切点须落在墨量谷（<45%带峰值；行间降部桥接≈15-40%，字身高≈60-90%）——高字单行被网格
                    # 强切两半的 14 个 +1 块（043_left 块3 等）由此杜绝
                    if seg[v - a] >= ink_thr2 * seg_max:
                        continue
                    if v - cuts[-1] > 8 and b2 - v > 8:
                        cuts.append(v)
        if len(cuts) < 2:
            # 谷点兜底须带高远超同块单行带（1.6×；单带块用绝对110px）——
            # 无约束时121个金页块被过拆（scan_002_left 块2 等），教训。
            sibs = [h for h2, h in zip(bands, heights_all) if h2 is not bd]
            sib_h = max(sibs) if sibs else 0
            thr_v = max(110.0, 1.6 * sib_h) if not sibs else 1.6 * sib_h
            if sib_h >= 20 or not sibs or pitch >= 14:  # 批17: 行距先验成立时放行（087_right块8被sib stub带闸死）
                if (b2 - a) > thr_v:
                    vc = _valley_cuts(counts_raw, a, b2,
                                      max(18, int(0.5 * sib_h)) if sibs else 18)
                    if vc:
                        cuts = [a] + vc
        if len(cuts) >= 2:
            made = [_band_from_rows(sub, counts_raw, cv,
                                    cuts[i + 1] if i + 1 < len(cuts) else b2, x0, y0)
                    for i, cv in enumerate(cuts)]
            # 批15回归：切片有无效（<4行/无墨）时放弃拆分保留原带——静默丢带曾致
            # 深欠检块计数再降（008_right 块14 8→6 类）
            if all(made):
                if mega:
                    for m2 in made:
                        m2['split_note'] = '并生带栅格强拆，行数待复核'
                out.extend(made)
                continue
            bd['unsplit_note'] = '切片无效'
        elif pitch >= 14 and (b2 - a) > 2.0 * pitch:
            # 批16共性1④：显著多行带未拆——逐带记原因供复核与回归统计
            bd['unsplit_note'] = '先验校验拒' if prior_used else '无墨谷'
        elif pitch < 14 and (b2 - a) > 110:
            bd['unsplit_note'] = '无行距先验'
        out.append(bd)
    return out


def _resplit_merged(gray: np.ndarray, box: list[int], bands: list[dict],
                    page_pitch: float = 0.0) -> list[dict]:
    """批21共性1：迭代包装至不动点（≤3趟）——首趟谷点兜底拆出的干净兄弟带让残余
    稠密段在二趟获得局部行距先验（109_right 块9 整块565px：首趟拆5行+269px残余、
    二趟拆尽实证；原单趟非递归致稠密段永不复检）。"""
    # 批21回归：稠密/点线块豁免——中位核墨<10（碎带甄别同判据）或中位 核墨/行距>1.0
    # （并生点线行自身核墨即超行距；080_right 块5 四栏 9→23 实证）。二趟迭代只对
    # 『兄弟带皆干净单行(核墨≈0.6×行距)+个别稠密残余』的正文块有意义
    if len(bands) >= 2:
        # 批21回归：med-ratio 豁免须 len>=2 即查——080_right 块5 栏以 2 条巨型并生带
        # 进入（核墨91/先验37=2.46），曾因 len>=4 前置门漏网被迭代炸成 23 带
        _mc = float(np.median([bd.get('h_core', 0) for bd in bands]))
        _pl = _pitch_of([bd['y0'] for bd in bands]) or page_pitch
        if _mc < 10 or (_pl and float(np.median([bd.get('h_core', 0) / max(_pl, 1.0)
                                                 for bd in bands])) > 0.85):
            return _resplit_merged_pass(gray, box, bands, page_pitch)
    for _pass in range(3):
        out = _resplit_merged_pass(gray, box, bands, page_pitch,
                                   core_gate=0.0 if _pass == 0 else 1.8)
        if len(out) == len(bands):
            return out
        bands = out
    return bands


def _fill_missing_bands(gray: np.ndarray, box: list[int], bands: list[dict],
                         page_pitch: float = 0.0) -> list[dict]:
    """批13共性1b：漏带补检——相邻带顶差>1.6×行距且空档内确有墨迹(行墨≥3px)
    时按空档墨迹补带（短行/淡行被阈值滤掉的典型场景）。"""
    if len(bands) < 3:
        return bands
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1] < INK_THR
    counts_raw = sub.sum(1).astype(float)
    pitch = _pitch_of([bd['y0'] for bd in bands])
    if pitch < 14:
        # 批17共性7：2带块唯一顶差无代表性——页先验兜底（字核门槛同 _resplit_merged）
        med_c7 = float(np.median([bd['h_core'] for bd in bands]))
        if 14 <= page_pitch <= 120 and med_c7 >= 0.35 * page_pitch:
            pitch = page_pitch
        else:
            return bands
    trig_gap = 1.3 * pitch if pitch < 40 else 1.6 * pitch  # 批17: 图注类短行距块收紧触发
    out = []
    for i, bd in enumerate(bands):
        out.append(bd)
        if i + 1 >= len(bands):
            continue
        gap_top, gap_bot = bd['y1'] - y0, bands[i + 1]['y0'] - y0
        if (bands[i + 1]['y0'] - bd['y0']) <= trig_gap or gap_bot <= gap_top:
            continue
        seg = counts_raw[gap_top:gap_bot]
        ink_rows = np.flatnonzero(seg >= 2)  # 批17: 图注小字行墨门槛 3→2px
        if not len(ink_rows):
            continue
        a2, b2 = int(ink_rows.min()) + gap_top, int(ink_rows.max()) + gap_top + 1
        if (b2 - a2) < max(4, 0.3 * pitch):
            continue
        nb = _band_from_rows(sub, counts_raw, a2, b2, x0, y0)
        if nb and 4 <= (nb['y1'] - nb['y0']):
            out.append(nb)
    # 批15共性3：末带以下至块底扫描——段末短行/窄行整段漏检（原逻辑只查带间空档）
    last = out[-1] if out else bands[-1]
    a_end = last['y1'] - y0
    b_end = min(y1 - y0, a_end + int(2.2 * pitch))
    if b_end - a_end > 0.8 * pitch:
        seg2 = counts_raw[a_end:b_end]
        ink2 = np.flatnonzero(seg2 >= 3)
        if len(ink2):
            a3, b3 = int(ink2.min()) + a_end, int(ink2.max()) + a_end + 1
            if b3 - a3 >= max(4, 0.3 * pitch):
                nb2 = _band_from_rows(sub, counts_raw, a3, b3, x0, y0)
                if nb2:
                    out.append(nb2)
    out.sort(key=lambda bd2: bd2['y0'])
    return out


def detect_bands(gray: np.ndarray, box: list[int], ink: int = INK_THR,
                 gap: int = 4, min_h: int = 7, page_pitch: float = 0.0,
                 iterate: bool = True) -> list[dict]:
    """块内水平投影 → 行带（含每带墨迹左右界与核心墨高，用于字号估计）。

    批1质检修正（2026-10-13）：
    - 行投影先平滑（[1,2,1]），抑制孤立尘点行；
    - 明显高于常见行高的带在内部墨量谷值处再拆分（紧排行/降部粘连）；
    - 高度 <8px 且紧贴上带的伪带（降部/杂墨）并入上带；
    - x 界剔除离主墨团 >80px 且宽 <15px 的孤立墨点簇（尘点/邻块墨迹）。
    """
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1] < ink
    # 批8质检修正：倾斜块先搜剪切斜率（谷值锐度最大者），行集合按剪切投影切
    best_slope, best_score = 0.0, -1.0
    if (y1 - y0) > 150 and (x1 - x0) > 400:
        for sl in [round(v, 4) for v in np.arange(-0.05, 0.0501, 0.002)]:
            prof = shear_profile(gray, box, sl)[:(y1 - y0) + 100]
            if prof.max() <= 0:
                continue
            nz = prof[prof > 0]
            score = float((prof.max()) / (np.mean(nz) + 1e-6))
            if score > best_score:
                best_score, best_slope = score, sl
    counts_raw = sub.sum(1).astype(float)
    if abs(best_slope) >= 0.002:
        # 用剪切投影替代水平投影做行集合（谷值判据：<=25% 局部峰）
        prof = shear_profile(gray, box, best_slope)
        thr_line = max(2.0, 0.25 * float(prof.max()))
        rows = prof >= thr_line
        bands, start, clean = [], None, 0
        for i, v in enumerate(rows):
            if v:
                if start is None:
                    start = i
                clean = 0
            elif start is not None:
                clean += 1
                if clean >= gap:
                    bands.append((start, i - clean + 1))
                    start, clean = None, 0
        if start is not None:
            bands.append((start, len(rows)))
        # 批9质检修正：shear_profile 以中心列 shift=0 对齐，回映偏移恒为 0
        bands = [(a, b) for a, b in bands]
        if not all(0 <= a < b <= (y1 - y0) and (b - a) >= 4 for a, b in bands):
            bands = []  # 带域无效 → 回落水平投影路径
        if len(bands) >= 2:
            out_bands = bands
            # 走与原路径相同的收尾（伪带并入/等分窗口/杂墨簇），复用下方逻辑
            prelim = [b2 - a2 for a2, b2 in out_bands]
            med_h0 = float(np.median(prelim)) if prelim else 0
            merged = []
            for a, b in out_bands:
                if merged and (b - a) < max(8, 0.35 * med_h0) and (a - merged[-1][1]) <= 6:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            bands = merged
            out2 = []
            thr = max(6, int((x1 - x0) * 0.02))
            for a, b in bands:
                if b - a < min_h:
                    continue
                colsum = sub[a:b].sum(0)
                on = np.flatnonzero(colsum >= 1)
                if not len(on):
                    continue
                clusters, cs = [], [on[0]]
                for c0, c1 in zip(on[:-1], on[1:]):
                    if c1 - c0 > 30:
                        clusters.append(cs)
                        cs = [c1]
                    else:
                        cs.append(c1)
                clusters.append(cs)
                big = max(clusters, key=len)
                keep = [cl for cl in clusters
                        if len(cl) >= 15 or abs(np.mean(cl) - np.mean(big)) <= 80]
                if len(clusters) >= 5:
                    starts2 = np.array([cl[0] for cl in clusters])
                    gaps2 = np.diff(starts2)
                    if float(np.mean(gaps2)) > 25 and float(np.std(gaps2)) < 0.25 * float(np.mean(gaps2)):
                        keep = clusters
                kept = np.concatenate(keep) if keep else big
                seg = counts_raw[a:b]
                core = int((seg >= max(float(np.max(seg)) * 0.35, 3.0)).sum()) or (b - a)
                out2.append({'y0': a + y0, 'y1': b + y0,
                             'x0': int(kept.min()) + x0, 'x1': int(kept.max()) + 1 + x0,
                             'h_core': core})
            if len(out2) >= 2:
                # 批11质检修正：纵切片合并——间隙<=8px 且 x 界重合>=0.9（切片同源）
                # 批13共性1：剪切路径原先无拆分/补检收尾，统一过三道后处理
                out2 = _drop_fragments(_merge_slices(out2))
                _rm = _resplit_merged if iterate else _resplit_merged_pass
                return _fill_missing_bands(gray, box, _rm(gray, box, out2, page_pitch),
                                           page_pitch)
    if len(counts_raw) >= 3:
        counts = np.convolve(counts_raw, [1, 2, 1], 'same') / 4.0
    else:
        counts = counts_raw
    rows = counts >= 2.0
    bands, start, clean = [], None, 0
    for i, v in enumerate(rows):
        if v:
            if start is None:
                start = i
            clean = 0
        elif start is not None:
            clean += 1
            if clean >= gap:
                bands.append((start, i - clean + 1))
                start, clean = None, 0
    if start is not None:
        bands.append((start, len(rows)))
    bands = [(a, b) for a, b in bands if b - a >= min_h]
    if not bands:
        return []
    # 批4质检修正：伪带并入放宽（高<0.35×行距中位、间隙<=6px 即并入上带）
    prelim = [b - a for a, b in bands]
    med_h0 = float(np.median(prelim)) if prelim else 0
    merged = []
    for a, b in bands:
        small = (b - a) < max(8, 0.35 * med_h0)
        if merged and small and (a - merged[-1][1]) <= 6:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    bands = merged
    # 批5质检修正：行距估计改下分位（P15），并带不再推高阈值；两轮迭代拆分
    def split_once(bands_in, guard=0):
        heights = [b - a for a, b in bands_in]
        if not heights:
            return bands_in
        unit = float(np.percentile(heights, 15))
        if guard > 0:  # 批10：第二遍护栏——若首遍未拆出更多带，unit 减半再试
            unit = max(8.0, unit * 0.5)
        med_pitch = unit * 1.45
        result = []
        for a, b in bands_in:
            if unit > 7 and (b - a) > med_pitch * 1.3:
                seg = counts[a:b]
                n_expect = max(2, int(round((b - a) / med_pitch)))
                cuts = [a]
                for i in range(1, n_expect):
                    center = a + int((b - a) * i / n_expect)
                    lo = max(a + 8, center - int(unit * 0.4))
                    hi = min(b - 8, center + int(unit * 0.4))
                    if hi <= lo:
                        continue
                    v = lo + int(np.argmin(seg[lo - a:hi - a]))
                    vmin = float(seg[v - a])
                    # 批10：谷点不过检时仍强制等分（吸附窗内极小值），不再整带放弃
                    if v - cuts[-1] > 8 and b - v > 8:
                        cuts.append(v)
                for i2, cv in enumerate(cuts):
                    result.append((cv, cuts[i2 + 1] if i2 + 1 < len(cuts) else b))
            else:
                result.append((a, b))
        return result

    bands = split_once(bands)
    bands2 = split_once(bands)
    if len(bands2) > len(bands):  # 第二遍正常路径
        bands = bands2
    else:  # 批10：自指先验失效（过合并占优）→ 护栏遍：单位减半强拆
        bands3 = split_once(bands, guard=1)
        bands = bands3 if len(bands3) > len(bands) else bands2
    # 批9质检修正：核心墨高 <0.3×中位的碎带（散点/下伸碎片）并入上带或丢弃
    # 批10：去 len>=4 门槛；另加间隙规则（相邻带间隙≤8px 且合并高≤1.4×单位即并）
    if len(bands) >= 2:
        unit_g = float(np.percentile([b2 - a2 for a2, b2 in bands], 15)) or 10.0
        gap_merged = []
        for a, b2 in bands:
            if gap_merged and (a - gap_merged[-1][1]) <= 8 and (b2 - gap_merged[-1][0]) <= 1.4 * unit_g * 1.45:
                gap_merged[-1] = (gap_merged[-1][0], b2)
            else:
                gap_merged.append((a, b2))
        bands = gap_merged
        if len(bands) >= 2 and (bands[0][1] - bands[0][0]) < 0.5 * unit_g \
                and (bands[1][0] - bands[0][1]) > 0.6 * unit_g * 1.45:
            bands = bands[1:]  # 块首孤立小带（降部碎片）丢弃
    if len(bands) >= 2:
        def _core(a, b2):
            seg2 = counts_raw[a:b2]
            return int((seg2 >= max(float(np.max(seg2)) * 0.35, 3.0)).sum()) or 1
        cores = [_core(a, b2) for a, b2 in bands]
        med_c = float(np.median(cores))
        kept2 = []
        for (a, b2), c2 in zip(bands, cores):
            if c2 < 0.3 * med_c and kept2 and (a - kept2[-1][1]) <= 10:
                kept2[-1] = (kept2[-1][0], b2)
            elif c2 >= 0.3 * med_c:
                kept2.append((a, b2))
        bands = kept2 or bands
    unit_f = float(np.percentile([b - a for a, b in bands], 15)) if bands else 0
    bands = [(a, b) for a, b in bands if b - a >= max(6, int(unit_f * 0.4))]
    out = []
    thr = max(6, int((x1 - x0) * 0.02))
    for a, b in bands:
        if b - a < min_h:
            continue
        colsum = sub[a:b].sum(0)
        on = np.flatnonzero(colsum >= 1)
        if not len(on):
            continue
        # 孤立墨点簇剔除：按 >=30px 空隙分组，丢掉离最大簇 >80px 且宽 <15px 的小簇
        clusters, cs = [], [on[0]]
        for c0, c1 in zip(on[:-1], on[1:]):
            if c1 - c0 > 30:
                clusters.append(cs)
                cs = [c1]
            else:
                cs.append(c1)
        clusters.append(cs)
        big = max(clusters, key=len)
        keep = [cl for cl in clusters
                if len(cl) >= 15 or abs(np.mean(cl) - np.mean(big)) <= 80]
        # 批7质检修正：等周期点簇（点线导引行，点距40-46px）豁免剔除
        if len(clusters) >= 5:
            starts = np.array([cl[0] for cl in clusters])
            gaps = np.diff(starts)
            if float(np.mean(gaps)) > 25 and float(np.std(gaps)) < 0.25 * float(np.mean(gaps)):
                keep = clusters
        kept = np.concatenate(keep) if keep else big
        # 批2质检修正：核心墨行=达到本带最大行墨量35%的行（窄栏降部/括号行不再虚增字高）
        seg = counts_raw[a:b]
        core = int((seg >= max(float(np.max(seg)) * 0.35, 3.0)).sum()) or (b - a)
        out.append({'y0': a + y0, 'y1': b + y0,
                    'x0': int(kept.min()) + x0, 'x1': int(kept.max()) + 1 + x0,
                    'h_core': core})
    if len(out) >= 2:
        out = _merge_slices(out)
    # 批13共性1：水平路径同样过三道后处理（碎带删/并带拆/漏带补）
    out = _drop_fragments(out)
    out = (_resplit_merged if iterate else _resplit_merged_pass)(gray, box, out, page_pitch)
    out = _fill_missing_bands(gray, box, out, page_pitch)
    return out


def column_ranges(gray: np.ndarray, box: list[int], bands: list[dict]) -> list[list[int]]:
    """检测块是否多栏：多数行带在同一 x 处出现 >=40px 竖直空隙则分栏。"""
    x0, y0, x1, y1 = box
    n_split = 0
    cut_votes: dict[int, int] = {}
    for bd in bands:
        seg = gray[bd['y0']:bd['y1'], x0:x1] < INK_THR
        cols = seg.sum(0) >= 1
        gaps, s = [], None
        for i, v in enumerate(cols):
            if not v and s is None:
                s = i
            elif v and s is not None:
                if i - s >= 40:
                    gaps.append((s + x0, i + x0))
                s = None
        if s is not None and x1 - (s + x0) >= 40:
            gaps.append((s + x0, x1))
        if gaps:
            n_split += 1
            for a, b in gaps:
                m = (a + b) // 2 // 20 * 20
                cut_votes[m] = cut_votes.get(m, 0) + 1
    if not cut_votes or n_split < max(2, len(bands) * 0.5):
        return [[x0, x1]]
    cuts = sorted(c for c, v in cut_votes.items() if v >= n_split * 0.6)
    ranges, left = [], x0
    for c in cuts:
        if c - left > 80:
            ranges.append([left, c])
            left = c
    if x1 - left > 80:
        ranges.append([left, x1])
    return ranges or [[x0, x1]]


def measure_line(a: np.ndarray, top: int, bottom: int, bounds: list[int],
                 guess: float, cap_px: int) -> tuple[float, float, float, float]:
    """按倾斜窗口测行左右界，多字块墨迹主峰拟合基线。返回 x0,x1,左端基线,斜率。"""
    xs = np.arange(max(0, bounds[0]), min(a.shape[1], bounds[1]))
    if not len(xs):
        return float(bounds[0]), float(bounds[0] + 1), float(top + cap_px), guess
    ref = float((bounds[0] + bounds[1]) / 2)
    rows = np.arange(max(0, top - 4), max(top + 1, bottom))[:, None]
    ys = np.clip(np.rint(rows + guess * (xs - ref)).astype(int), 0, a.shape[0] - 1)
    mask = a[ys, xs] < 125
    colsum = mask.sum(0)
    cols = xs[colsum > 3]
    if not len(cols):
        return float(bounds[0]), float(bounds[1]), float(top + cap_px), guess
    x0, x1 = float(cols.min()), float(cols.max() + 1)
    # 批5质检修正：行首长破折号等细横杠（每列墨1-3行）被剔除致起点右移——左侧连续细墨段>=30px 则延伸
    thin = xs[colsum >= 1]
    left_thin = thin[thin < cols.min()]
    if len(left_thin):
        run_end = cols.min()
        seg = left_thin[left_thin >= run_end - 220]
        if len(seg) >= 30 and seg.max() - seg.min() + 1 == len(seg) and run_end - seg.min() <= 4:
            x0 = float(seg.min())
    points = []
    for left in range(int(x0), int(x1) - 40, 150):
        bottoms = []
        for x in range(left, min(left + 150, int(x1))):
            expected = top + cap_px + guess * (x - ref)
            lo = max(0, int(expected - 35))
            hi = min(a.shape[0], int(expected + 9))
            ink = np.flatnonzero(a[lo:hi, x] < 125)
            if len(ink) >= 3:
                bottoms.append(int(ink[-1]) + lo)
        if len(bottoms) > 15:
            peak = int(np.argmax(np.convolve(
                np.bincount(bottoms, minlength=a.shape[0]), np.ones(3), 'same')))
            points.append([left + min(150, int(x1) - left) / 2, peak])
    if len(points) >= 2:
        slope, itc = np.polyfit(np.asarray(points)[:, 0], np.asarray(points)[:, 1], 1)
        # 批4质检修正：拟合点>=4 时信任实测斜率（原 0.018 否决窗会误杀反向倾斜页）
        if abs(slope - guess) > 0.018 and len(points) < 4:
            slope = guess
            itc = float(np.median([y - slope * x for x, y in points]))
        return x0, x1, float(slope * x0 + itc + 1), float(slope)
    return x0, x1, float(guess * (x0 - ref) + top + cap_px + 1), guess


def find_rules(gray: np.ndarray, box: list[int], y_body_top: int | None = None
               ) -> list[list[int]]:
    """全页长横线/装饰线（页眉下划线、出版者上方装饰线）。

    批1质检修正：不再只扫页顶 400px，整页扫描；厚度 <=24px 的长墨条
    （最长连续墨行 >55% 宽）视为规则线，多条全返回；>24px 的成片区域
    是插图，不算。返回 [x0,y0,x1,y1] 列表。
    """
    x0, y0, x1, y1 = box
    # 批2质检修正：断续线桥接<=8px 断口；斜线允许行间 run 中心平移
    hit_rows = []  # (y, center)
    top_lim = y0 + (y1 - y0) // 10
    for y in range(y0, y1):
        row = gray[y, x0:x1] < (160 if y < top_lim else 125)  # 批7：页顶淡墨横线放宽
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        splits = np.flatnonzero(np.diff(idx) > 24)  # 批11：桥接放宽（断口53-131px 的页眉线）
        runs = np.split(idx, splits + 1)
        best = max(runs, key=len)
        cov_thr = 0.28 if y < top_lim else 0.55  # 批11：页顶横线门槛 55%→28%
        if len(best) > (x1 - x0) * cov_thr:
            hit_rows.append((y, float(best[0] + best[-1]) / 2))
    rules, cur = [], []
    for y, ctr in hit_rows:
        if cur and y - cur[-1][0] <= 3 and abs(ctr - cur[-1][1]) <= 8:
            cur.append((y, ctr))
        else:
            if cur and len(cur) <= 48:
                rules.append(cur)
            cur = [(y, ctr)]
    if cur and len(cur) <= 48:
        rules.append(cur)
    out = []
    for ys in rules:  # ys 为 (y, center) 列表；批3质检修正：切片取 [0]
        band = gray[ys[0][0]:ys[-1][0] + 1, x0:x1] < 125
        cols = np.flatnonzero(band.any(0))
        if len(cols) < (x1 - x0) * 0.5:
            continue
        out.append([int(cols.min()) + x0, ys[0][0], int(cols.max()) + 1 + x0, ys[-1][0] + 1,
                    max(2, len(ys))])  # 批20共性2①：4元水平带自动转5元（断言只认5元）
    # 批4质检修正：斜细线（页眉下划线常见）——单行 run>=25% 宽、中心随 y 线性漂移
    drift_rows = []
    for y in range(y0, y1):
        row = gray[y, x0:x1] < (160 if y < top_lim else 125)  # 批8：斜线分支页顶同样放宽
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        splits = np.flatnonzero(np.diff(idx) > 8)
        runs = np.split(idx, splits + 1)
        best = max(runs, key=len)
        if len(best) >= max(60, (x1 - x0) * 0.03):
            drift_rows.append((y, float(best[0] + best[-1]) / 2, len(best)))
    # 批5质检修正：k 为 dx/dy（每行中心平移），换算 rise=dy/dx 判定；门槛放宽到薄近水平线
    # 批9质检修正：按 y 间隔聚簇、逐簇迭代剔离群拟合（全页单次拟合被正文行污染）
    clusters, cur_c = [], []
    for r in drift_rows:
        if cur_c and r[0] - cur_c[-1][0] <= 4:
            cur_c.append(r)
        else:
            if len(cur_c) >= 8:
                clusters.append(cur_c)
            cur_c = [r]
    if len(cur_c) >= 8:
        clusters.append(cur_c)
    for cl in clusters:
        pts = list(cl)
        for _ in range(3):
            ys_d = np.array([r[0] for r in pts])
            cs_d = np.array([r[1] for r in pts])
            k, b0 = np.polyfit(ys_d, cs_d, 1)
            resid = np.abs(cs_d - (k * ys_d + b0))
            keep_i = resid <= max(12.0, 2.0 * float(np.median(resid)))
            if keep_i.all() or len(keep_i) < 8:
                break
            pts = [pt for pt, kk in zip(pts, keep_i) if kk]
        ys_d = np.array([r[0] for r in pts])
        cs_d = np.array([r[1] for r in pts])
        if len(pts) < 8:
            continue
        k, b0 = np.polyfit(ys_d, cs_d, 1)
        resid = float(np.abs(cs_d - (k * ys_d + b0)).max())
        span = float(cs_d.max() - cs_d.min())
        rise = abs(1.0 / k) if abs(k) > 1e-6 else 0.0
        near_horiz = rise <= 0.0015 and len(pts) <= 30
        if (0.0015 <= rise <= 0.06 or near_horiz) and resid <= 12 and span >= 0.15 * (x1 - x0):
            out.append([int(b0 + k * ys_d[0]), int(ys_d[0]),
                        int(b0 + k * ys_d[-1]), int(ys_d[-1]),
                        max(2, int(np.median([r[2] for r in pts]) * 0.12))])
            break
    # 批13共性2：页眉下细斜线专扫（人工实测：厚2-3px、微斜0.012-0.024、几乎整页宽；
    # 9/10 页此前靠人工补测）。斜描边每行墨段=厚/斜率≈60-200px 宽的独立 run，
    # 与页眉文字行(词桥接成>30%宽 run)、双栏文本行(每行两个达标簇)可分。
    # y_body_top=首个正文块顶缘（draft() 传入）：线必须位于其上方，防正文区误报。
    y_acc = int(y_body_top) - 2 if y_body_top else y0 + 420
    cands = []  # (y, run_x0, run_x1)：每行恰有一个达标 run 才取（页眉词簇/双栏行/边缘残块弃）
    for y in range(y0, min(y1, y0 + 420)):
        idx = np.flatnonzero(gray[y, x0 + 15:x1 - 15] < 230)
        if not len(idx):
            continue
        splits = np.flatnonzero(np.diff(idx) > 24)
        runs = np.split(idx, splits + 1)
        ok = [(int(r[0]), int(r[-1])) for r in runs
              if 30 <= len(r) <= (x1 - x0) * 0.30]
        if len(ok) == 1:
            rx0, rx1 = ok[0]
            cands.append((y, rx0 + x0 + 15, rx1 + x0 + 15))
    chains, cur_ch = [], []
    for r in cands:
        # 宽容成链：漂移预算随行距伸缩（斜0.012的浅线每行漂85px；淡线缺行5行≈425px）
        if cur_ch:
            dy_g = r[0] - cur_ch[-1][0]
            drift = abs((r[1] + r[2]) / 2 - (cur_ch[-1][1] + cur_ch[-1][2]) / 2)
            if dy_g <= 8 and drift <= 110 * dy_g + 80:
                cur_ch.append(r)
                continue
        if cur_ch and len(cur_ch) >= 8:
            chains.append(cur_ch)
        cur_ch = [r]
    if len(cur_ch) >= 8:
        chains.append(cur_ch)
    new_lines = []
    for cl in chains:
        pts_l = list(cl)

        def _fit(pts):
            ys_f = np.array([r[0] for r in pts], dtype=float)
            cs_f = np.array([(r[1] + r[2]) / 2 for r in pts])
            # 批13调试：影印页细线有平滑弯曲（端部偏离直线>70px），>=12点改二次拟合
            if len(pts) >= 12:
                co = np.polyfit(ys_f, cs_f, 2)
                return lambda yy, co=co: np.polyval(co, yy)
            k_f, b_f = np.polyfit(ys_f, cs_f, 1)
            return lambda yy, k_f=k_f, b_f=b_f: k_f * yy + b_f

        for _ in range(3):
            if len(pts_l) < 8:
                break
            fx = _fit(pts_l)
            pts2 = [r for r in pts_l
                    if abs(float(fx(r[0])) - (r[1] + r[2]) / 2)
                    <= (r[2] - r[1]) / 2 + 14]
            if len(pts2) == len(pts_l) or len(pts2) < 8:
                pts_l = pts2 if len(pts2) >= 8 else pts_l
                break
            pts_l = pts2
        if len(pts_l) < 8:
            continue
        # 批13调试：影印线端头强弯曲（局部斜率75→123px/行），全局拟合外推不够——
        # 两轮回捞：每轮重拟合（含已回捞点）后按 ±(w/2+80) 从原链端部外扩
        for _round in range(2):
            fx_end = _fit(pts_l)
            i_lo, i_hi = cl.index(pts_l[0]), cl.index(pts_l[-1])
            moved = False
            while i_lo > 0:
                p = cl[i_lo - 1]
                if abs(float(fx_end(p[0])) - (p[1] + p[2]) / 2) > (p[2] - p[1]) / 2 + 80:
                    break
                i_lo -= 1
                moved = True
            while i_hi < len(cl) - 1:
                p = cl[i_hi + 1]
                if abs(float(fx_end(p[0])) - (p[1] + p[2]) / 2) > (p[2] - p[1]) / 2 + 80:
                    break
                i_hi += 1
                moved = True
            pts_l = cl[i_lo:i_hi + 1]
            if not moved:
                break
        if pts_l[0][0] > y_acc:  # 线顶必须低于首个正文块（防正文区误报）
            continue
        r_top, r_bot = pts_l[0], pts_l[-1]
        dy_line = max(1, r_bot[0] - r_top[0])
        dx_line = abs((r_bot[1] + r_bot[2] - r_top[1] - r_top[2]) / 2)
        rise = dy_line / max(1.0, dx_line)  # dy/dx（斜率倒数取端点差，免受中段弯曲影响）
        span = float(max(r[2] for r in pts_l) - min(r[1] for r in pts_l))
        if span < 400 or not (0.0015 <= rise <= 0.06):
            continue
        if (r_top[1] + r_top[2]) <= (r_bot[1] + r_bot[2]):  # 线往右下走
            new_lines.append([r_top[1], r_top[0], r_bot[2], r_bot[0] + 1, 2])
        else:
            new_lines.append([r_bot[1], r_bot[0], r_top[2], r_top[0] + 1, 2])
    acc_top2 = int(y_body_top) if y_body_top else y0 + 420
    for ln_i, ln in enumerate(new_lines):  # 批20共性2③：装订影假起点校验
        if ln[0] < 60:
            y_a, y_b = ln[1], ln[3]
            seg_bind = gray[max(0, y_a - 2):y_b + 2, 0:150] < 150
            cols_bind = np.flatnonzero(seg_bind.any(0))
            if not len(cols_bind):  # 该端150px内无浓墨=假起点，剔至首个浓墨列
                seg_far = gray[max(0, y_a - 2):y_b + 2, 150:ln[0] + 320] < 150
                cf = np.flatnonzero(seg_far.any(0))
                if len(cf):
                    ln[0] = int(150 + cf[0])
    if len(new_lines) > 1:  # 批18共性2：多线时短线降级（<主线70% 系误检，093L [205,140,1066,122] 教训）
        mx_sp = max(ln[2] - ln[0] for ln in new_lines)
        new_lines = [ln for ln in new_lines if (ln[2] - ln[0]) >= 0.7 * mx_sp]
    for ln in new_lines:
        out.append(ln)
        # 细斜线检出后，剔除与它 y 重叠的旧短残段（069_left [324,92,707,101,11] 类；
        # 真横线(>50%宽)与页外装饰线不受影响）
        out = [r for r in out
               if not (r is not ln and (r[2] - r[0]) < 0.5 * (x1 - x0)
                       and r[1] <= ln[3] + 15 and r[3] >= ln[1] - 15)]
    # 批17共性2③：页眉区厚>4px『规则』系鬼影（089_left 23px/090_left 10px），生成端即弃
    out = [r for r in out if not (len(r) >= 5 and r[4] > 4 and r[1] < acc_top2)]
    return out

# ---------------------------------------------------------------- 文本行填充


def markup_spans(markup: str, size: float) -> list[dict]:
    """行内标记 → span 列表（字体/字号/上下标/疏排），可提取纯文本。

    批1质检修正：递归解析，支持 <i><b> 嵌套（→Times Bold Italic）；
    <sp>…</sp> 为疏排（разрядка）标记，tracking='auto' 由渲染时按行宽反算。
    """
    from xml.etree import ElementTree as ET
    # 裸 & 会让 ET 抛 ParseError 走兜底分支、静默丢掉全部行内标记，先转义成实体再解析
    safe = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)', '&amp;', markup)
    try:
        root = ET.fromstring('<line>' + safe + '</line>')
    except ET.ParseError:
        return [{'font': 'R', 'text': re.sub(r'<[^>]+>', '', markup),
                 'size': size, 'rise': 0, 'tracking': 0}]
    spans: list[dict] = []

    def walk(node, fonts: tuple, tracking) -> None:
        font = {'i': 'I', 'b': 'B'}.get(node.tag)
        new_fonts = fonts + (font,) if font else fonts
        fname = ''.join(sorted(set(new_fonts), key=lambda c: 'IB'.index(c))) if new_fonts else 'R'
        if fname == 'IB':
            fname = 'BI'
        tr = 'auto' if node.tag == 'sp' else tracking
        sub = node.tag == 'sub'
        sup = node.tag == 'sup'
        if node.text:
            spans.append({'font': fname, 'text': node.text,
                          'size': size * .73 if sub else size,
                          'rise': -size * .22 if sub else (size * .33 if sup else 0),
                          'tracking': tr})
        for ch in node:
            walk(ch, new_fonts, tr)
            if ch.tail:
                spans.append({'font': fname, 'text': ch.tail, 'size': size,
                              'rise': 0, 'tracking': tr})

    walk(root, (), 0)
    merged: list[dict] = []
    for sp in spans:
        if sp['text'] == '':
            continue
        if merged and merged[-1]['font'] == sp['font'] and merged[-1]['size'] == sp['size'] \
                and merged[-1]['rise'] == sp['rise'] and merged[-1]['tracking'] == sp['tracking']:
            merged[-1]['text'] += sp['text']
        else:
            merged.append(sp)
    return merged


def word_tokens(markup: str) -> list[list[dict]]:
    """标记文本 → 词 token 序列（每词为 span 片段列表，保留字体）。"""
    toks, cur = [], []
    for s in markup_spans(markup, 10):
        parts = re.split(r'(\s+)', s['text'])
        for p in parts:
            if not p:
                continue
            if p.isspace():
                if cur:
                    toks.append(cur)
                    cur = []
            else:
                cur.append({'font': s['font'], 'text': p,
                            'size': s['size'] / 10, 'rise': s['rise'] / 10})
    if cur:
        toks.append(cur)
    return toks


def token_width(tok: list[dict], size: float) -> float:
    from reportlab.pdfbase import pdfmetrics
    w = 0.0
    for f in tok:
        w += pdfmetrics.stringWidth(f['text'], f['font'], f['size'] * size)
    return w


def greedy_fill_bands(toks: list[list[dict]], bands: list[dict], size: float,
                      space_pt: float) -> tuple[list[list[list[dict]]], list[list[dict]]]:
    """批14共性4c：逐带按各自墨宽贪心装词（DP 失败回退用），返回（每带词表, 剩余词表）。"""
    lines, wi = [], 0
    for bd in bands:
        bw = (bd['x1'] - bd['x0']) * SCALE - 1.0
        cur, wsum = [], 0.0
        while wi < len(toks):
            tw = token_width(toks[wi], size)
            add = tw if not cur else tw + space_pt
            if cur and wsum + add > bw + 2:
                break
            cur.append(toks[wi])
            wsum += add
            wi += 1
        lines.append(cur)
    return lines, toks[wi:]


def _presplit_failed_seg(gray: np.ndarray, seg: list[dict], pitch: float) -> list[dict]:
    """批19共性1b：DP 失败路径专用谷点预拆——超高带(>1.7×行距)放宽谷阈剖开。
    仅在必截断的失败分支调用，不触金页通过路径与并生带判据（回归否决项不动）。"""
    out = []
    for bd in seg:
        h_f = bd['y1'] - bd['y0']
        if h_f > 1.7 * pitch:
            x0b = max(0, bd['x0'] - 10)
            x1b = min(gray.shape[1], bd['x1'] + 10)
            sub_f = gray[bd['y0']:bd['y1'], x0b:x1b] < INK_THR
            counts_f = sub_f.sum(1).astype(float)
            cuts_f = _valley_cuts(counts_f, 0, h_f, max(10, int(0.4 * pitch)), calm_frac=0.35)
            made_ok = False
            if cuts_f:
                edges = [0] + cuts_f + [h_f]
                made_f = [_band_from_rows(sub_f, counts_f, edges[i], edges[i + 1], x0b, bd['y0'])
                          for i in range(len(edges) - 1)]
                if all(made_f):
                    out.extend(made_f)
                    made_ok = True
            if not made_ok and h_f > 2.0 * pitch:
                # 批21共性1：谷点无效时比例剖分兜底（网格±0.3行距窗内argmin，无墨阈）——
                # 111_left 块15 型（148px/3行、行间墨桥无平静段）；仅DP必截断路径介入
                n_p = max(2, min(12, int(round(h_f / pitch))))
                cuts_p = [0]
                for i_p in range(1, n_p):
                    c_p = int(h_f * i_p / n_p)
                    lo_p = max(2, c_p - int(0.3 * pitch))
                    hi_p = min(h_f - 2, c_p + int(0.3 * pitch))
                    if hi_p <= lo_p:
                        continue
                    v_p = lo_p + int(np.argmin(counts_f[lo_p:hi_p]))
                    if v_p - cuts_p[-1] > 8 and h_f - v_p > 8:
                        cuts_p.append(v_p)
                if len(cuts_p) >= 2:
                    edges = cuts_p + [h_f]
                    made_p = [_band_from_rows(sub_f, counts_f, edges[i], edges[i + 1], x0b, bd['y0'])
                              for i in range(len(edges) - 1)]
                    if all(made_p):
                        for m3 in made_p:
                            m3['split_note'] = '比例剖分，行数待复核'
                        out.extend(made_p)
                        continue
        out.append(bd)
    return out


def fill_lines(toks: list[list[dict]], width_pt: float, size: float,
               indent_pt: float, space_pt: float) -> list[list[list[dict]]]:
    """贪心断行：非末行按 width_pt 排满，段末行自然收尾。"""
    lines, cur, w = [], [], 0.0
    for t in toks:
        tw = token_width(t, size)
        add = tw if not cur else tw + space_pt
        limit = width_pt - (indent_pt if not lines else 0)
        if cur and w + add > limit + 2:
            lines.append(cur)
            cur, w = [t], tw
        else:
            cur.append(t)
            w += add
    if cur:
        lines.append(cur)
    return lines

# ---------------------------------------------------------------- 排版渲染


def register_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for name in ('R', 'I', 'B', 'BI'):
        font = FONT_FILES.get(name)
        if font is None or not font.is_file():
            raise FileNotFoundError(f'请配置有效的 render.fonts.{name} 字体文件：{font}')
        pdfmetrics.registerFont(TTFont(name, str(font)))


def draw_text(c, x: float, baseline: float, slope: float, spans: list[dict],
              word_space: float = 0, char_space: float = 0,
              page_h_px: float = 2660.0) -> None:
    c.saveState()
    c.translate(x * SCALE, (page_h_px - baseline) * SCALE)
    c.rotate(-math.degrees(math.atan(slope)))
    t = c.beginText(0, 0)
    t.setTextRenderMode(0)
    t.setWordSpace(word_space)
    for s in spans:
        t.setFont(s['font'], s['size'])
        t.setRise(s['rise'])
        t.setCharSpace(char_space + s['tracking'])
        t.textOut(s['text'])
    c.drawText(t)
    c.restoreState()

# ---------------------------------------------------------------- 页面驱动


def page_dir(uid: str, root: Path | None = None) -> Path:
    return (PAGES_ROOT if root is None else root / 'pages') / uid


def atomic_write(path: Path, text: str) -> None:
    """批1质检修正：临时文件+os.replace 原子写，防并发读到截断 JSON。"""
    import os
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


BINDING_FILES = ('源页.png', '页面配置.json', '行文本.txt', '原位排印校样.pdf',
                 '程序验收.json', '逐行坐标样式.json')


def artifact_hashes(d: Path, names: tuple[str, ...] = BINDING_FILES) -> dict[str, str]:
    """计算本页输入和交付物的版本哈希；缺失文件由调用方明确处理。"""
    return {name: hashlib.sha256((d / name).read_bytes()).hexdigest() for name in names}


def check_runtime() -> None:
    """依赖自检：给错解释器（缺 reportlab/pypdf）时立即报错并指明正确路径。"""
    import sys
    missing = []
    for mod in ('numpy', 'PIL', 'reportlab', 'pypdf'):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise RuntimeError(
            f'缺少依赖 {missing}：请在项目独立环境中安装 requirements.txt 中的依赖 '
            f'（当前 {sys.executable}）')


def draft(uid: str, root: Path | None = None) -> dict:
    """生成草稿三件套：块行带.json / 行文本.txt / 页面配置.json。"""
    units = {u['id']: u for u in load_units()}
    unit = units[uid]
    raw = load_raw(uid)
    pg = Page(unit)
    d = page_dir(uid, root)
    d.mkdir(parents=True, exist_ok=True)
    if not (d / '源页.png').exists():
        pg.rgb.save(d / '源页.png')
    blocks = parse_blocks(raw['raw'])
    size_flags: list[str] = []
    if BOOK_SPECIFIC_RULES:
        # 批5质检修正：已知页眉/节标题词典校验，OCR 误读（如 Л→А）置 flag 给候选
        import difflib
        KNOWN_HEADERS = ['HYDRACHNELLAE', 'ЛИТЕРАТУРА', 'МЕТОДЫ ИССЛЕДОВАНИЯ', 'СИСТЕМА',
                         'Фауна СССР', 'ОГЛАВЛЕНИЕ', 'ГИДРАХНЕЛЛЫ', 'HYDRACHNIDA',
                         'ОТДЕЛ', 'ОПИСАНИЕ ФОРМ', 'ЭКОЛОГИЧЕСКИЙ ОБЗОР', 'ГЕОГРАФИЧЕСКОЕ',
                         # 批12根因：本书真实 running header 精确白名单——命中即静默
                         'V. HYDRACHNIDAE', 'N. HYDRACHNA', 'I. HYDRACHNA', 'HYDRACHNA',
                         'HYDRACHNIDAE', 'ЛИТЕРАТУРА ПО ГИДРАХНЕЛАМ',
                         'X. SPERCHONIDAE', 'SPERCHONIDAE', 'SPERCHON']
        # 批13共性4：1↔I/0↔O 归一 + 长度闸门(差≥2字符不列候选) + 最优匹配≥90% 硬抑制
        def _norm_hdr(s: str) -> str:
            return s.upper().replace('Ё', 'Е').replace('1', 'I').replace('0', 'O')

        wl_hdr = [_norm_hdr(k) for k in KNOWN_HEADERS]
        try:  # 批19共性6：卷级页眉裁定缓存——同卷确认过的页眉文本自动抑制
            wl_hdr += [_norm_hdr(k) for k in json.loads((LIB_DIR / '卷级页眉裁定.json').read_text())]
        except Exception:
            pass
        for b in blocks:
            if b['label'] in ('Page-Header', 'Section-Header'):
                plain = re.sub(r'<[^>]+>', '', ' '.join(b['paragraphs'])).strip()
                up = _norm_hdr(plain)
                if up in wl_hdr:
                    continue  # 精确命中白名单（含 1↔I 归一）：不报警
                best_r, best_k = 0.0, ''
                for kn in wl_hdr:
                    if abs(len(kn) - len(up)) >= 2:
                        continue
                    r = difflib.SequenceMatcher(None, up, kn).ratio()
                    if r > best_r:
                        best_r, best_k = r, kn
                if 0.62 <= best_r < 0.90:
                    size_flags.append(f'⚠ 块头「{plain}」疑似OCR误读，候选「{best_k}」（相似{best_r:.0%}），请放大核对')
    img_boxes = [block_px(b, pg.size) for b in blocks if b['label'] == 'Image']
    # 批13共性2：页眉线必须位于首个正文块上方（防正文区行段误报）
    body_top = min((block_px(b, pg.size)[1] for b in blocks
                    if b['label'] not in ('Page-Header', 'Image')), default=None)
    rules = find_rules(pg.gray, [0, 0, pg.size[0], pg.size[1]], y_body_top=body_top)
    rules = [r for r in rules
             if not any(not (r[2] < ib[0] or r[0] > ib[2] or r[3] < ib[1] or r[1] > ib[3])
                        for ib in img_boxes)]
    # 批15共性6：页眉线兜底——专扫拟合失败时（rules 无页顶 5 元线），在页眉行带
    # 下缘 y+10~60px 低阈值扫 ≥800px 细墨 run（3/10 页整条漏检靠人工实测手补）
    has_hdr_line = any(len(r) >= 5 and r[4] <= 4 and r[1] < (body_top or pg.size[1])
                       for r in rules)  # 厚>4px 的 rules 为鬼影候选（084R 厚21px黑杠孤例），不计
    if not has_hdr_line:
        hdr_bot2 = max((block_px(b, pg.size)[3] for b in blocks
                        if b['label'] == 'Page-Header'), default=0)
        hits = []
        cand_r = []
        for y in range(hdr_bot2 + 10, min(pg.size[1], hdr_bot2 + 60)):
            idx2 = np.flatnonzero(pg.gray[y, 15:pg.size[0] - 15] < 230)
            if not len(idx2):
                continue
            splits2 = np.flatnonzero(np.diff(idx2) > 24)
            runs2 = np.split(idx2, splits2 + 1)
            best2 = max(runs2, key=len)
            if len(best2) >= 250:  # 批17: 斜线单行run 300-500px，跨行漂移拟合兜底
                cand_r.append((y, float(best2[0] + best2[-1]) / 2 + 15,
                               int(best2[0]) + 15, int(best2[-1]) + 15))
        if len(cand_r) >= 3:
            ys_c = np.array([c[0] for c in cand_r], float)
            cs_c = np.array([c[1] for c in cand_r])
            k_c, b_c = np.polyfit(ys_c, cs_c, 1)
            keep_m = np.abs(cs_c - (k_c * ys_c + b_c)) <= 30
            if keep_m.sum() >= 3:
                pts_k = [c for c, km in zip(cand_r, keep_m) if km]
                ys_k = np.array([c[0] for c in pts_k], float)
                cs_k = np.array([c[1] for c in pts_k])
                k_c, b_c = np.polyfit(ys_k, cs_k, 1)
                span_c = float(cs_k.max() - cs_k.min())
                rise_c = abs(1.0 / k_c) if abs(k_c) > 1e-6 else 99.0
                if span_c >= 500 and rise_c <= 0.05:
                    hits = [(c[0], c[2], c[3]) for c in pts_k]
        if len(hits) >= 3:
            ys_h = [h[0] for h in hits]
            rules.append([min(h[1] for h in hits), min(ys_h),
                          max(h[2] for h in hits), max(ys_h) + 1, 2])
            size_flags.append('⚠ 页眉线兜底检出（专扫拟合失败后低阈值补扫），请核对端点与斜度')
    # 批16共性3：页眉线终检断言——有页眉文本带而页眉区（底边+60px 内）无细线即报警，
    # 防整线静默漏检（批13专扫+批15兜底后仍 3/10 页靠手补）；厚>4px 的 rules 视为鬼影不计
    elif any(b['label'] == 'Page-Header' for b in blocks):
        size_flags.append('⚠ 页眉区未检出细线（专扫+兜底均失败），若原书有线请实测补 rules')
    corrs = load_corrections(uid)
    if BOOK_SPECIFIC_RULES:
        # 方案二：全书替换表（124 复核页草稿→终稿 diff 提炼，learn_corrections.py 生成）
        book_tbl = LIB_DIR / '书面替换表.json'
        if book_tbl.exists():
            for b5 in blocks:
                for i5, para in enumerate(b5['paragraphs']):
                    for old_w, new_w in json.loads(book_tbl.read_text()):
                        if old_w in para:
                            b5['paragraphs'][i5] = para = para.replace(old_w, new_w)
        # 批7质检修正：Surya LaTeX 式记法清洗（P_4→斜体+下标、\hat{3}/\delta→♂），逐处记 flag
        import re as _re
        # 批12根因：键去反斜杠存规范名，匹配时归一化前导反斜杠（单/双都命中）
        sym_map = {'hat{3}': '♂', 'delta': '♂', 'hat{♂}': '♂', 'hat{♀}': '♀',
                   'phi': '♀', 'bar{delta}': '♂', 'bar{Q}': '♀',
                   'Deltalambda': 'Дл.', 'lambda': 'Дл.', 'hat{sigma}': 'σ',
                   'varrho': '♀', 'rho': '♀'}  # 批17共性4：与 delta→♂/phi→♀ 同层
        # 批13共性3：E 记法确定性归一（变体→正体 Ep<sub>罗马数字</sub>；1→I 2→II 3→III 4→IV）
        _ROMAN = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}

        def _enote(mm):
            num = next(g for g in mm.groups() if g)
            rom = num if set(num) <= set('IVX') else _ROMAN.get(num, num)
            return f'Ep<sub>{rom}</sub>'

        _E_NOTE = re.compile(r'E_\{[PpрР]([IVX0-9]+)\}'
                             r'|E_\{\\text\{[PpрР]\}_\{\\text\{([IVX0-9]+)\}\}\}'
                             r'|E_\{\\text\{[PpрР]\}\\text\{([IVX0-9]+)\}\}'
                             r'|E_\{<i>[PpрР]</i><sub>([IVX0-9]+)</sub>\}'
                             r'|Ep_\{?([IVX0-9]+)\}?'
                             r'|EP?_\{+(?:\\text\{)?[PpрР]?([IVX0-9]+)\}*'
                             r'|EP_([IVX0-9]+)\b')
        # 批14共性5：分数归一（\frac{n}{m} 与 ¼½¾ → <sup>n</sup>/<sub>m</sub>；⅓无字形排三字符）
        _FRAC_U = {'¼': ('1', '4'), '½': ('1', '2'), '¾': ('3', '4')}
        for bi3, b in enumerate(blocks):
            for i2, para in enumerate(b['paragraphs']):
                src = para
                for k3, v3 in sym_map.items():
                    pat = re.compile(r'\\{1,2}' + re.escape(k3))
                    if pat.search(para):
                        para = pat.sub(v3, para)
                        size_flags.append(f'块{bi3}符号清洗 \\{k3}→{v3}，请放大核对')
                # 批21裁定：解剖记号 P₂/P₄ 的拉丁 P 为斜体（110L/037L/074R 三处放大
                # 独立验证+111_left 复核互证；直立系草稿端历史错误，已全书回改178处）
                para = re.sub(r'(?<![iA-Za-z>/])P<sub>', r'<i>P</i><sub>', para)
                m2 = _E_NOTE.sub(_enote, para)
                # 批19共性2：残迹变体（E_{p,i、E_{<i>P</i><sub>III、E_{\\text{p 等不完整记法）禁止原样印出
                if any(s2 in para for s2 in ('E_{', 'EP_', '\\mu', '\\varphi', 'Ер <sub', 'Р <sub')):
                    size_flags.append(f'块{bi3}E记法/LaTeX残迹（无法自动定值），请人工改写勿印出字面')
                # 批16共性2：E/Ep 记法漏网变体——西里尔 Ер 的阿拉伯/混合下标→罗马
                # （Ер1→ЕрI、Ер1V→ЕрIV；拉丁 P 的阿拉伯下标不转换）；剥残留杂符
                _ER_ROMAN = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V', '1V': 'IV'}

                def _er_sub(mm):
                    return f'Ер<sub>{_ER_ROMAN.get(mm.group(1), mm.group(1))}</sub>'

                m2 = _re.sub(r'Ер<sub>(\d+V?)</sub>', _er_sub, m2)
                m2 = _re.sub(r'(</sub>)\s*[-–]?\s*[IVX]{1,4}\}', r'\1', m2)
                m2 = m2.replace('}', '')
                if m2 != para:
                    size_flags.append(f'块{bi3}E记法归一（E_p变体→Ep+罗马下标），请放大核对')
                    para = m2
                if _re.search(r'<sub>I{4}</sub>|E_I{4}', para):
                    size_flags.append(f'块{bi3}下标连续4个I（疑III误读非IV减写），请放大核对，勿自动折叠')
                # 批18共性5：下标值系列互证（值本身仍人工终判，只自动点名孤例）
                _subs_page = _re.findall(r'(?:Ep|Ер|<i>P</i>)<sub>([IVX]+|\d+)</sub>', para)
                for v5 in set(_subs_page):
                    if len(_subs_page) >= 3 and _subs_page.count(v5) == 1 \
                            and sum(1 for w5 in set(_subs_page) if _subs_page.count(w5) >= 2) >= 2:
                        size_flags.append(f'块{bi3}下标『{v5}』为本页孤例（其余成系列），请放大核对是否OCR塌缩')
                        break
                m3 = _re.sub(r'\\frac\{(\d+)\}\{(\d+)\}',
                             lambda mm: f'<sup>{mm.group(1)}</sup>/<sub>{mm.group(2)}</sub>', para)
                m3 = _re.sub('[¼½¾]',
                             lambda mm: '<sup>%s</sup>/<sub>%s</sub>' % _FRAC_U[mm.group(0)], m3)
                if m3 != para:
                    size_flags.append(f'块{bi3}分数记法归一（\\frac/¼½¾→sup+sub），请放大核对')
                    para = m3
                # 批14共性6：文献年份冒号统一『1925: 479』体（用户 2026-10-14 裁定：前不空后空）
                m4 = _re.sub(r'(\b\d{4})[ \t]*:[ \t]*', r'\1: ', para)
                if m4 != para:
                    size_flags.append(f'块{bi3}年份冒号疏排归一，请放大核对')
                    para = m4
                m = _re.sub(r'\b([PE][rv]?)_\{?([IVX0-9]+)\}?',
                            lambda mm: f'<i>{mm.group(1)}</i><sub>{mm.group(2)}</sub>', para)
                if m != para:
                    size_flags.append(f'块{bi3}下标记法转换（P_Er 型），请核对斜体')
                    para = m
                if para != src:
                    b['paragraphs'][i2] = para
    applied, skipped = [], []
    for b in blocks:
        for i, para in enumerate(b['paragraphs']):
            plain = re.sub(r'<[^>]+>', '', para)
            for c in corrs:
                if c['原文'] == c['校正']:
                    continue
                if c['原文'] in plain and c['原文'] in para:
                    para = para.replace(c['原文'], c['校正'])
                    b['paragraphs'][i] = para
                    applied.append(c)
                    plain = re.sub(r'<[^>]+>', '', b['paragraphs'][i])
                elif c['原文'] in plain:
                    skipped.append({'原文': c['原文'], '校正': c['校正'],
                                    '原因': '替换串跨越斜体等行内标记，留人工'})
    ocr_warn = []
    if not raw.get('eos_reached', True):
        ocr_warn.append('OCR生成被截断(eos未到)，文字候选可能不完整')
    if raw.get('generation_limit_reached'):
        ocr_warn.append('OCR达到生成上限，文字候选不完整')
    spec = {'uid': uid, 'scan': unit['scan'], 'side': unit.get('side', 'whole'),
            'source_sha256': hashlib.sha256(
                raw_path(uid).read_bytes()).hexdigest(),
            'ocr_warnings': ocr_warn + (['⚠ ' + f for f in size_flags] if size_flags else []),
            'rules': rules,
            'blocks': [], '预套用校正': len(applied), '未套用校正': skipped}
    lines_txt = []
    # 批15共性1：页级行距先验——整块并带时块内最小顶差=2×行距而局部估计失效，
    # 取全页正文带顶差 P25（限35-70px正文行距窗；点线/图注块带高<1.4×行距天然不受影响）
    diffs_all: list[float] = []
    for b in blocks:
        if b['label'] in ('Image', 'Section-Header') or not b['paragraphs']:
            continue
        bx = block_px(b, pg.size)
        bx = [max(0, bx[0] - 6), max(0, bx[1] - 6),
              min(pg.size[0], bx[2] + 6), min(pg.size[1], bx[3] + 6)]
        # 批21回归：先验循环用单趟版——迭代会改变块框检测的带顶，P25 先验漂移
        # 连锁全页栏检测（268_left 45.0→42.5、073_right 45.5→46.5 实证）；先验只取顶差
        pb = detect_bands(pg.gray, bx, iterate=False)
        tops_p = sorted(bd['y0'] for bd in pb)
        diffs_all.extend(float(v) for v in np.diff(tops_p))
    cand_p = [d for d in diffs_all if 35 <= d <= 70]
    page_pitch = float(np.percentile(cand_p, 25)) if len(cand_p) >= 5 else 0.0
    for bi, b in enumerate(blocks):
        if b['label'] == 'Image' or not b['paragraphs']:
            spec['blocks'].append({'index': bi, 'label': b['label'],
                                   'bbox': block_px(b, pg.size), 'kind': 'image',
                                   'paragraphs': b['paragraphs']})
            continue
        box = block_px(b, pg.size)
        box = [max(0, box[0] - 6), max(0, box[1] - 6),
               min(pg.size[0], box[2] + 6), min(pg.size[1], box[3] + 6)]
        bands = detect_bands(pg.gray, box,
                             page_pitch=0.0 if b['label'] in ('Section-Header',) else page_pitch)
        cols = column_ranges(pg.gray, box, bands)
        blk = {'index': bi, 'label': b['label'], 'bbox': box, 'kind': 'text',
               'columns': [], 'paragraphs': b['paragraphs']}
        for cx0, cx1 in cols:
            cbd = detect_bands(pg.gray, [cx0, box[1], cx1, box[3]],
                               page_pitch=0.0 if b['label'] in ('Section-Header',) else page_pitch)
            # 批20共性4：带 y 叠插图且 x 端入图→收窄至图外窄栏墨缘
            # （103_right 人工收窄 x1=768-783 的先例脚本化）
            for bdg in cbd:
                for ib2 in img_boxes:
                    if min(bdg['y1'], ib2[3]) - max(bdg['y0'], ib2[1]) <= 2:
                        continue
                    if bdg['x0'] < ib2[0] - 4 and bdg['x1'] > ib2[0]:
                        bdg['x1'] = min(bdg['x1'], ib2[0] - 2)
                    elif bdg['x1'] > ib2[2] + 4 and bdg['x0'] < ib2[2]:
                        bdg['x0'] = max(bdg['x0'], ib2[2] + 2)
            # 批16共性1③④：并生带强拆/未拆原因出 flag，键不落库
            for bdg in cbd:
                if 'split_note' in bdg:
                    size_flags.append(f"⚠ 块{bi}{bdg['split_note']}")
                    del bdg['split_note']
                elif 'unsplit_note' in bdg:
                    size_flags.append(f"⚠ 块{bi} 有多行未拆带（{bdg['unsplit_note']}），请重拆")
                    del bdg['unsplit_note']
            hs = [bd['h_core'] for bd in cbd]
            # 批15共性7：初估抗坏带——核墨高中位数以下的碎核（坏带/墨尾）剔除后重取中位
            if hs:
                med_h = float(np.median(hs))
                hs2 = [h for h in hs if h >= 0.55 * med_h] or hs
                size_est = round(float(np.median(hs2)) * SCALE / 0.74, 1)
            else:
                size_est = 11.2
            bh = [bd['y1'] - bd['y0'] for bd in cbd]
            if len(bh) >= 3:
                disp = (float(np.percentile(bh, 75)) - float(np.percentile(bh, 25))) / max(1.0, float(np.median(bh)))
                if disp > 0.6:
                    size_flags.append(f'⚠ 块{bi}带高离散度{disp:.2f}，ink_height 不可信，以行宽拟合为准')
            # 字号标定：段落数与墨带段数一致时按 DP 拟合残差选最优字号
            segs_c = segment_bands(sorted(cbd, key=lambda bd: bd['y0']))
            paras_t = [word_tokens(p) for p in b['paragraphs']]
            if len(cols) == 1:
                size_est, cal = calibrate_size(paras_t, segs_c, size_est)
            else:
                cal = {'method': 'ink_height', 'grid': None}
            lo_b, hi_b = (4.5, 13.0) if b['label'] in ('Caption', 'Page-Footer') else (9.2, 12.5)
            # 批17共性5：图注块（Фиг. 前缀）字号下限 8.5pt——墨宽拟合成 5-6pt 散架
            # （088_right 块8 5.1→8.5 复核先例；与缓办『图注专检』独立先行）
            if any(p.strip().startswith('Фиг.') for p in b['paragraphs'][:1]):
                lo_b = max(lo_b, 8.5)
            if b['label'] == 'Page-Footer':
                lo_b = max(lo_b, 7.0)  # 批18共性6：页码/页脚 4.5pt 塌缩（27px墨高↔约9pt）
            if not lo_b <= size_est <= hi_b:  # 批10：按块类分档钳制
                cal = dict(cal, clamped_from=size_est)
                # 批19补+批20共性3：帽高先验（max核墨×0.24/0.662）全面接管钳制路径——
                # 宽度拟合被污染带拖低 6.2pt 类，帽高与宽度差>15% 即取帽先验并⚠
                cap_seed = round(float(max(bd['h_core'] for bd in cbd)) * SCALE / 0.662, 1) \
                    if cbd else size_est
                fallback7 = None
                if abs(size_est - cap_seed) > 0.15 * max(cap_seed, 0.1) and lo_b <= cap_seed <= hi_b:
                    fallback7 = cap_seed
                elif size_est < lo_b < cap_seed:
                    fallback7 = round(size_est + 0.10 * (cap_seed - size_est), 1)
                size_est = round(min(max(size_est, lo_b), hi_b), 1)
                if fallback7 is not None and lo_b <= fallback7 <= hi_b:
                    cal = dict(cal, floor_fallback=fallback7)
                    size_est = fallback7
                    size_flags.append(f'⚠ 块{bi}字号触底回退墨高种子{fallback7}pt，需复核')
                else:
                    size_flags.append(f'⚠ 块{bi}字号标定失败({size_est}pt被钳制)，行带/墨高证据异常，需复核')
            for bd in cbd:  # 批6质检修正：条目首悬挂破折号（左缘细横墨 1-5px 高、55-140px 长、距文≤80px）
                left0 = max(0, bd['x0'] - 150)
                win = pg.gray[bd['y0']:bd['y1'] + 1, left0:bd['x0']] < INK_THR
                if not win.any():
                    continue
                per_col = win.sum(0)
                cand = np.flatnonzero((per_col >= 1) & (per_col <= 5))
                if len(cand) < 55:
                    continue
                segs_d, st = [], cand[0]
                for i in range(1, len(cand)):
                    if cand[i] - cand[i - 1] > 4:
                        segs_d.append((st, cand[i - 1]))
                        st = cand[i]
                segs_d.append((st, cand[-1]))
                for a, b2 in segs_d:
                    L = b2 - a + 1
                    gap2 = bd['x0'] - (b2 + left0)
                    if 55 <= L <= 140 and 0 <= gap2 <= 80:
                        ys2 = np.flatnonzero(win[:, a:b2 + 1].any(1))
                        if len(ys2) <= 6 and ys2.min() >= (bd['y1'] - bd['y0']) * 0.3:
                            bd['dash'] = [int(a + left0), int(bd['y0'] + ys2.min()),
                                          int(b2 + 1 + left0), int(bd['y0'] + ys2.max())]
                            break
            lefts = sorted(bd['x0'] for bd in cbd)
            indent = float(np.percentile(lefts, 90) - np.percentile(lefts, 10)) if len(lefts) > 3 else 0.0
            blk['columns'].append({'x': [int(cx0), int(cx1)], 'bands': cbd,
                                   'size_pt': size_est, 'size_calibration': cal,
                                   'indent_px': round(indent, 1),
                                   'guess_slope': 0.012})
        spec['blocks'].append(blk)
        lines_txt.append(f"# 块{bi} {b['label']}")
    # 批21共性2：钳位字号同页重估——Text 块钳位值与同页未钳制 Text 块中位差
    # >1.5pt 时按中位重估（107_left 五块、110_left 整页 9.2 系钳位低估教训：
    # 污染带拖垮宽度拟合与帽高种子，但同页健康块中位仍可信）
    _uncl = [cb2['size_pt'] for b in spec['blocks']
             if b.get('kind') == 'text' and b.get('label') == 'Text'
             for cb2 in b['columns'] if 'clamped_from' not in cb2.get('size_calibration', {})]
    if len(_uncl) >= 3:
        _pmed = float(np.median(_uncl))
        for b in spec['blocks']:
            if b.get('kind') != 'text' or b.get('label') != 'Text':
                continue
            for cb2 in b['columns']:
                if 'clamped_from' in cb2.get('size_calibration', {}) \
                        and abs(cb2['size_pt'] - _pmed) > 1.5:
                    cb2['size_calibration'] = dict(cb2['size_calibration'],
                                                   page_median_refit=round(_pmed, 1))
                    size_flags.append(f"⚠ 块{b['index']}字号钳位{cb2['size_pt']}pt按同页中位"
                                      f"重估{round(_pmed, 1)}pt，需复核")
                    cb2['size_pt'] = round(_pmed, 1)
    # 批10质检修正：全页带 y1 封顶（跨块防叠印：y1=min(y1, 次带y0-2)）
    all_bds = []
    for b in spec['blocks']:
        if b['kind'] != 'text':
            continue
        for cb in b['columns']:
            all_bds.extend(cb['bands'])
    all_bds.sort(key=lambda bd: bd['y0'])
    for i in range(len(all_bds) - 1):
        nxt = all_bds[i + 1]['y0']
        if all_bds[i]['y1'] > nxt - 2:
            all_bds[i]['y1'] = max(all_bds[i]['y0'] + 4, nxt - 2)
    (d / '页面配置.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    (d / '块行带.json').write_text(json.dumps(
        {'uid': uid, 'blocks': [{'index': b['index'], 'label': b['label'],
                                 'columns': b['columns']} for b in spec['blocks'] if b['kind'] == 'text']},
        ensure_ascii=False, indent=1))
    return {'uid': uid, 'blocks': len(blocks), 'text_blocks': sum(1 for b in spec['blocks'] if b['kind'] == 'text'),
            '预套用校正': len(applied), '未套用校正': len(skipped)}


def segment_bands(bands: list[dict]) -> list[list[dict]]:
    """按段首缩进把一栏墨带切成段（段首=左界明显内缩，段末=右端未抵右边距）。

    批4质检修正：切段前剔除 h_core<10px 的窄杂墨带（墨点/降部碎片）。
    """
    if not bands:
        return []
    hs = [bd['h_core'] for bd in bands]
    if len(hs) >= 3 and float(np.median(hs)) >= 12:
        bands = [bd for bd in bands if bd['h_core'] >= 10]
    lefts = sorted(bd['x0'] for bd in bands)
    rights = sorted(bd['x1'] for bd in bands)
    margin_l = lefts[len(lefts) // 10]
    margin_r = rights[-max(1, len(rights) // 10)]
    # 批13共性1d：伪段切修正——+15px 静态阈值把微斜带(顶差致 x0 内移20px+)误判段首
    # （067_left 块8 x0=141 案例）；改为 max(30, 0.5×行距中位)，真段首缩进≈1em≫该值
    tops_d = np.diff(sorted(bd['y0'] for bd in bands))
    med_pitch = float(np.median(tops_d)) if len(tops_d) else 0
    indent_thr = max(30, 0.5 * med_pitch)
    segs, cur = [], []
    for bd in bands:
        para_start = bd['x0'] >= margin_l + indent_thr
        if para_start and cur:
            segs.append(cur)
            cur = []
        cur.append(bd)
    if cur:
        segs.append(cur)
    return segs


def is_ragged(bd: dict, margin_r: int) -> bool:
    return bd['x1'] < margin_r - 30


def fit_words_to_bands(toks: list[list[dict]], bands: list[dict], size: float
                       ) -> tuple[list[list[list[dict]]], list[str]]:
    """带宽驱动 DP 断行：每行词集的自然宽度逼近该行墨带宽度。

    正文两端对齐行的自然宽度 ≈ 墨带宽 - 词隙数×理想拉伸(2.2pt)；
    段末行（右端未抵边）只罚超宽且必须吃光剩余词。
    DP 后对仍超宽的行尝试行末断词加连字符（辅音+元音边界），复现原书断行。
    """
    from reportlab.pdfbase import pdfmetrics
    space_pt = pdfmetrics.stringWidth(' ', 'R', size)
    n, m = len(toks), len(bands)
    if n == 0 or m == 0:
        return [], ['词或墨带为空']
    w = [token_width(t, size) for t in toks]
    margin_r = max(bd['x1'] for bd in bands)
    targets = []
    # 批14共性4a：ragged 只允许段末带——中带 x1 被下标/标点降部削短时（073_left 六条带
    # 削短14~36px）误判 ragged 会强制吞尽余词致 DP 无解；中带一律按两端对齐代价处理
    for k, bd in enumerate(bands):
        targets.append(((bd['x1'] - bd['x0']) * SCALE - 1.0,
                        k == len(bands) - 1 and is_ragged(bd, margin_r)))
    INF = 1e18
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    pre = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for k in range(m):
        t, ragged = targets[k]
        for i in range(n + 1):
            if dp[i][k] >= INF:
                continue
            natural = 0.0
            for j in range(i + 1, n + 1):
                natural += w[j - 1] + (space_pt if j - 1 > i else 0)
                if ragged:
                    if j < n:
                        continue          # 段末行必须吃光剩余词
                    cost = max(0.0, natural - t) ** 2
                else:
                    if natural - t > 60:
                        break             # 剪枝：明显超宽
                    nsp = j - i - 1
                    cost = (natural - (t - 2.2 * nsp)) ** 2
                if dp[i][k] + cost < dp[j][k + 1]:
                    dp[j][k + 1] = dp[i][k] + cost
                    pre[j][k + 1] = i
    if dp[n][m] >= INF:
        return [], [f'断行失败：{n}词无法填入{m}墨带']
    lines = []
    j, k = n, m
    while k > 0:
        i = pre[j][k]
        lines.append(toks[i:j])
        j, k = i, k - 1
    lines.reverse()
    # 残差检查 + 行末断词修复（只对仍超宽的两端对齐行）
    flags = []
    for idx in range(len(lines) - 1):
        t, ragged = targets[idx]
        if ragged:
            continue
        ln = lines[idx]
        natural = sum(token_width(x, size) for x in ln) + space_pt * (len(ln) - 1)
        err = natural - t
        if err <= 6:
            continue
        last = ln[-1]
        word = ''.join(f['text'] for f in last)
        core = word.rstrip('.,;:!?»)"\u201d')
        tail_punct = word[len(core):]
        word = core
        hyph = pdfmetrics.stringWidth('-', 'R', size)
        best = None
        for cut in hyphen_cuts(word):
            head, tail = word[:cut], word[cut:]
            head_w = pdfmetrics.stringWidth(head, last[0]['font'], size) + hyph
            new_nat = natural - token_width(last, size) + head_w
            if best is None or abs(new_nat - t) < best[0]:
                best = (abs(new_nat - t), head, tail)
        if best and best[0] < err - 1:
            _, head, tail = best
            font = last[0]['font']
            fsize = last[0]['size']
            lines[idx] = ln[:-1] + [[{'font': font, 'text': head + '-',
                                      'size': fsize, 'rise': 0}]]
            lines[idx + 1] = [[{'font': font, 'text': tail + tail_punct,
                                'size': fsize, 'rise': 0}]] + lines[idx + 1]
            flags.append(f'第{idx + 1}行行末断词 {head}-{tail}{tail_punct}')
    return lines, flags


_PYRU = None  # Pyphen ru_RU 断字器缓存；导入失败置 False（回退元音启发式）


def _pyphen_ru():
    global _PYRU
    if _PYRU is None:
        try:
            import pyphen
            _PYRU = pyphen.Pyphen(lang=HYPHEN_LANGUAGE) if HYPHEN_LANGUAGE else False
        except Exception:
            _PYRU = False
    return _PYRU or None


def hyphen_cuts(word: str) -> list[int]:
    """俄文断点集：Pyphen ru_RU 断字模式优先（含 ь/ъ/й/双辅音等正字规则，
    LibreOffice 同源模式），无结果或库缺失时回退元音启发式（元音前的辅音之前断开）。

    щетинками → Pyphen [щетин|ками]；启发式（苏维埃近似）[щетин|ками, щетинка|ми]。
    两段各>=2字母；去重升序。
    """
    if not HYPHEN_LANGUAGE:
        return []
    h = _pyphen_ru()
    if h:
        cuts = sorted({p for p in h.positions(word) if 2 <= p <= len(word) - 2})
        if cuts:
            return cuts
    if HYPHEN_LANGUAGE != 'ru_RU':
        return []
    VOW = set('аеиоуыэюяёАЕИОУЫЭЮЯЁ')
    cuts = []
    for p in range(2, len(word) - 1):
        if word[p] in VOW and word[p - 1] not in VOW:
            cut = p - 1
            if cut >= 2 and len(word) - cut >= 2 and cut not in cuts:
                cuts.append(cut)
    return sorted(cuts)


def _plausible_hyphen(head: str, tail: str) -> bool:
    """俄文断词近似规则：断点前为辅音+元音起头尾部，且两段都>=2字母。"""
    if HYPHEN_LANGUAGE != 'ru_RU' or len(head) < 2 or len(tail) < 2:
        return False
    VOW = set('аеиоуыэюяёАЕИОУЫЭЮЯЁ')
    return head[-1] not in VOW and tail[0] in VOW | set('ъь')


def calibrate_size(paras: list[list[list[dict]]], segs: list[list[dict]],
                   size_est: float) -> tuple[float, dict]:
    """字号自动标定：哪个字号下段落词集最能贴合原书行结构（残差小、断词少）。

    返回 (标定字号, 诊断)。段落数与墨带段数不一致时不标定，沿用墨高估计。
    """
    # force 重建路径下本函数可能先于 fill_all 首次执行，stringWidth 依赖字体已注册（209L KeyError 'R'）
    register_fonts()
    if len(paras) != len(segs) or len(segs) < 2:
        return size_est, {'method': 'ink_height', 'grid': None}
    grid = [round(size_est + d, 1) for d in (-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4)]
    results = {}
    for s in grid:
        total, nlines, nhyp, fail = 0.0, 0, 0, False
        for ptoks, seg in zip(paras, segs):
            lines, fls = fit_words_to_bands(ptoks, seg, s)
            if not lines:
                fail = True
                break
            from reportlab.pdfbase import pdfmetrics
            space_pt = pdfmetrics.stringWidth(' ', 'R', s)
            margin_r = max(bd['x1'] for bd in seg)
            for ln, bd in zip(lines, seg):
                t = (bd['x1'] - bd['x0']) * SCALE - 1.0
                ragged = is_ragged(bd, margin_r)
                nat = sum(token_width(x, s) for x in ln) + space_pt * (len(ln) - 1)
                err = max(0.0, nat - t) if ragged else (nat - (t - 2.2 * (len(ln) - 1)))
                total += abs(err)
                nlines += 1
            nhyp += len(fls)
        if fail or not nlines:
            results[s] = 1e9
        else:
            # 批1质检修正：偏离墨高证据的字号罚 2.0/pt，防止 DP 凑行宽压字号
            results[s] = round(total / nlines + 1.5 * nhyp + 2.0 * abs(s - size_est), 2)
    best = min(grid, key=lambda s: results[s])
    return best, {'method': 'dp_fit', 'grid': results}


def fill_all(uid: str, root: Path | None = None) -> list[str]:
    """按页面配置把段落文本吸附到行带，写 行文本.txt（每行一条标记文本）。

    每栏先用段首缩进把墨带切段；段落数与段数一致时用带宽驱动 DP 断行
    （复现原书断行，含行末连字符），否则回退贪心流灌并记 flag。
    多栏块按阅读序：第 1 栏墨带填满后进入第 2 栏。
    """
    from reportlab.pdfbase import pdfmetrics
    register_fonts()
    d = page_dir(uid, root)
    spec = json.loads((d / '页面配置.json').read_text())
    out, flags = [], []

    def width_refine(lines, band_list, size, space_pt):
        """批5质检修正：行宽拟合字号——对齐行 target/natural 的中位因子缩放字号。"""
        facs = []
        for ln, bd in zip(lines, band_list):
            if not ln or not bd:
                continue
            nat = sum(token_width(x, size) for x in ln) + space_pt * (len(ln) - 1)
            t = (bd['x1'] - bd['x0']) * SCALE - 1.0
            if nat > 12 and t > 20 and nat >= 0.4 * t:  # 批7：短行剔除
                facs.append(t / nat)
        if len(facs) < 3:
            return size
        # 批15共性7：Tukey 剔离群——坏带/错行的 target/natural 因子先剔再取中位
        if len(facs) >= 5:
            q1, q3 = float(np.percentile(facs, 25)), float(np.percentile(facs, 75))
            iqr2 = q3 - q1
            facs = [f for f in facs if q1 - 1.5 * iqr2 <= f <= q3 + 1.5 * iqr2] or facs
        new_size = size * float(np.median(facs))
        # 批10质检修正：因子一致（IQR<0.08）时绕过 ±pt 限幅（初估偏低30-50%时仍可回收）
        iqr = float(np.percentile(facs, 75) - np.percentile(facs, 25))
        consistent = iqr < 0.08
        if consistent and 4.5 <= new_size <= 16.0:
            return round(new_size, 1)
        # 批7质检修正：限幅（±2.5pt/±1.5pt，绝对界 8-14pt），坏值回退原字号
        if not 8.0 <= new_size <= 14.0 or not (size - 2.5) <= new_size <= (size + 1.5):
            return size  # 被否决时由调用方（fill_all）记 flag
        return round(new_size, 1)

    for b in spec['blocks']:
        if b['kind'] != 'text':
            continue
        out.append(f"# 块{b['index']} {b['label']}")
        paras = [word_tokens(p) for p in b['paragraphs']]
        queue = []  # 跨栏未耗尽段落
        for ci, cb in enumerate(b['columns']):
            size = cb.get('size_pt', 11.2)
            space_pt = pdfmetrics.stringWidth(' ', 'R', size)
            bands = sorted(cb['bands'], key=lambda bd: bd['y0'])
            segs = segment_bands(bands)
            if ci == 0 and len(bands) >= 3:
                # 批9质检修正：首轮先行宽度拟合（贪心量尺），DP 用精化字号
                probe = fill_lines([t for pr in paras for t in pr],
                                   (cb['x'][1] - cb['x'][0]) * SCALE - 1.5, size, 0, space_pt)
                size_p = width_refine(probe, bands[:len(probe)], size, space_pt)
                if size_p != size and 4.5 <= size_p <= 16.0:
                    size = size_p
                    cb['size_pt'] = size_p
                elif abs(size_p - size) > 0.5 and size_p == size:
                    pass
                # 批10：初估与拟合差大且被限幅否决 → 记 flag 供复核改 size_pt
                probe_facs = size_p / size if size else 1.0
                if abs(probe_facs - 1.0) > 0.2 and size_p == size:
                    flags.append(f"⚠ 块{b['index']}栏{ci}: 初估字号{size}pt与行宽证据偏离>{20:.0f}%，请复核 size_pt")
            if ci == 0 and len(paras) == len(segs) and segs:
                dash_rules = spec.setdefault('rules', [])
                for pi in range(len(paras)):
                    seg = segs[pi]
                    dsh = seg[0].get('dash') if seg else None
                    if dsh and paras[pi] and paras[pi][0] and paras[pi][0][0]['text'] in ('—', '–', '——', '-'):
                        paras[pi] = paras[pi][1:]
                        dash_rules.append([dsh[0], dsh[1], dsh[2], dsh[3], 3])
                for pi, (ptoks, seg) in enumerate(zip(paras, segs)):
                    fl, fls = fit_words_to_bands(ptoks, seg, size)
                    flags.extend(f'块{b["index"]}段{pi + 1}: ' + f for f in fls)
                    if not fl:
                        # 批19共性1b：失败路径先谷点预拆重试 DP（096R/098L/100R 并生带类）
                        try:
                            from PIL import Image as _Img
                            _gray_f = np.asarray(_Img.open(d / '源页.png').convert('L'))
                        except Exception:
                            _gray_f = None
                        _pitch_f = _pitch_of([bd['y0'] for bd in seg])
                        if _gray_f is not None and _pitch_f >= 14 and seg:
                            seg_o = list(seg)
                            seg_p = _presplit_failed_seg(_gray_f, seg, _pitch_f)
                            if len(seg_p) > len(seg):
                                fl_p, fls_p = fit_words_to_bands(ptoks, seg_p, size)
                                if fl_p and not fls_p:
                                    # 预拆带回写栏带表（渲染侧行/带一一对应）
                                    old_ids = {id(x) for x in seg_o}
                                    cb['bands'] = sorted(
                                        [x for x in cb['bands'] if id(x) not in old_ids] + seg_p,
                                        key=lambda bd3: bd3['y0'])
                                    seg = seg_p
                                    fl = fl_p
                                    flags.append(f'块{b["index"]}段{pi + 1}: DP失败谷点预拆{len(seg_p)}带后重试成功')
                        if not fl:
                            # 批14共性4c：DP失败回退——逐带按各自墨宽贪心装词（075_right 人工
                            # 验证过的方案），与等宽 fill_lines 截断比取丢词少者；丢词>0 仍硬 ⚠
                            gl, leftover_g = greedy_fill_bands(ptoks, seg, size, space_pt)
                            all_lines = fill_lines(ptoks, (seg[0]['x1'] - seg[0]['x0']) * SCALE,
                                                   size, 0, space_pt)
                            lost_f = sum(len(l2) for l2 in all_lines[len(seg):])
                            if len(leftover_g) <= lost_f:
                                fl = gl
                                lost_src, lost_w = leftover_g, len(leftover_g)
                            else:
                                fl = all_lines[:len(seg)]
                                lost_src = [t for l2 in all_lines[len(seg):len(seg) + 2] for t in l2]
                                lost_w = lost_f
                            if lost_w:
                                lost_t = ' '.join(re.sub(r'<[^>]+>', '', line_markup([t], size))[:30]
                                                  for t in lost_src[:4])
                                flags.append(f'⚠ 块{b["index"]}段{pi + 1}: DP失败贪心回退，截断{lost_w}词：{lost_t}')
                    size2 = width_refine(fl, seg, size, space_pt)
                    if abs(size2 - size) > 0.25:
                        fl2, _ = fit_words_to_bands(ptoks, seg, size2)
                        if fl2:
                            fl = fl2
                        size = size2
                        cb['size_pt'] = size2
                        flags.append(f'块{b["index"]}段{pi + 1}: 字号宽度拟合精化→{size2}pt')
                    for ln in fl:
                        out.append(line_markup(ln, size))
                    for _ in range(len(fl), len(seg)):
                        out.append('⟦空带⟧')
            else:
                if ci == 0:
                    queue = list(paras)
                    flags.append(f"块{b['index']}栏{ci}: 段落数{len(paras)}与墨带段数"
                                 f"{len(segs)}不匹配，贪心流灌")
                used = 0
                width_pt = (cb['x'][1] - cb['x'][0]) * SCALE - 1.5
                taken_lines, taken_bands = [], []
                dash_rules = spec.setdefault('rules', [])
                while queue and used < len(bands):
                    toks = queue.pop(0)
                    dsh = bands[used].get('dash') if used < len(bands) else None
                    if dsh and toks and toks[0] and toks[0][0]['text'] in ('—', '–', '——', '-'):
                        toks = toks[1:]
                        dash_rules.append([dsh[0], dsh[1], dsh[2], dsh[3], 3])
                    # 批7质检修正：按各带自身墨宽断行（检索表条目带宽各异），逐带填词
                    take = []
                    wi = 0
                    bi = used
                    while bi < len(bands) and wi < len(toks):
                        bw = (bands[bi]['x1'] - bands[bi]['x0']) * SCALE - 1.0
                        cur, w = [], 0.0
                        while wi < len(toks):
                            tw = token_width(toks[wi], size)
                            add = tw if not cur else tw + space_pt
                            if cur and w + add > bw + 2:
                                break
                            cur.append(toks[wi])
                            w += add
                            wi += 1
                        if not cur:
                            break
                        take.append(cur)
                        bi += 1
                    taken_lines.extend(take)
                    taken_bands.extend(bands[used:used + len(take)])
                    for ln in take:
                        out.append(line_markup(ln, size))
                    if wi < len(toks):
                        queue.insert(0, toks[wi:])
                    used += len(take)
                    # 批9质检修正：词数守恒——段末仍有带而词尽/词余，显式记 flag
                    if not queue and wi >= len(toks) and used < len(bands) and ci == len(b['columns']) - 1 and len(take) < 999:
                        pass
                if len(taken_lines) >= 3:
                    size2 = width_refine(taken_lines, taken_bands, size, space_pt)
                    if abs(size2 - size) > 0.25:
                        cb['size_pt'] = size2
                        flags.append(f"块{b['index']}栏{ci}: 字号宽度拟合精化→{size2}pt（贪心分支，重渲染生效）")
                for _ in range(used, len(bands)):
                    out.append('⟦空带⟧')
                if queue and ci == len(b['columns']) - 1:
                    flags.append(f"块{b['index']}: {len(queue)}段未耗尽，剩余墨带不足")
    if BOOK_SPECIFIC_RULES:
        # 批4质检修正：删除 </i> , 型标记收尾标点前的伪空格
        # 批6质检修正：词内斜体/正体切换处伪空格（-Formen / ( <i> 型）
        out = [re.sub(r'(</[a-z]+>)\s+([,.;:!?»)-])', r'\1\2', ln) for ln in out]
        out = [re.sub(r'\(\s+(<[a-z/]+>)', r'(\1', ln) for ln in out]
        # 批14共性1a：下标/上标恒紧排（</i> <sub>n</sub> 型伪空格，071_right P1/P2 等4页）
        out = [re.sub(r'(</[a-z]+>)\s+(<su[bp]>)', r'\1\2', ln) for ln in out]

        def _tight_paren(ln: str) -> str:
            # 批14共性1b：『闭标签 ( 』删空格仅当标签内末字符为非字母——
            # 斜体词名后跟 ( 是原书实空格（<b>Hydrachna</b> (s. str.) 等3页），不能删
            def _repl(mm):
                before = ln[max(0, mm.start() - 1):mm.start()]
                return mm.group(0) if before and before.isalpha() else mm.group(1) + '('
            return re.sub(r'(</[a-z]+>)(\s+\()', _repl, ln)

        out = [_tight_paren(ln) for ln in out]
    (d / '行文本.txt').write_text('\n'.join(out) + '\n')
    (d / '页面配置.json').write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    return flags


def line_markup(tokens: list[list[dict]], size: float) -> str:
    """词 token → 行内标记文本（保留 i/b/bi 与上下标）。"""
    parts = []
    for tk in tokens:
        buf, cur_font, cur_rise, cur_tr = '', None, None, None
        for f in tk:
            if (cur_font and f['font'] != cur_font) or (cur_rise is not None and f['rise'] != cur_rise) \
                    or (cur_tr is None) != (f.get('tracking') is None) or (cur_tr is not None and f.get('tracking') != cur_tr):
                parts.append(wrap_rise(wrap_font(buf, cur_font, cur_tr), cur_rise))
                buf = ''
            cur_font = f['font']
            cur_rise = f['rise']
            cur_tr = f.get('tracking')
            buf += f['text']
        if buf:
            parts.append(wrap_rise(wrap_font(buf, cur_font, cur_tr), cur_rise))
    s = ' '.join(parts)
    s = re.sub(r'</i> <i>', ' ', s)
    s = re.sub(r'</b> <b>', ' ', s)
    s = re.sub(r'</sub> <sub>', ' ', s)
    s = re.sub(r'</sup> <sup>', ' ', s)
    return s


def wrap_rise(text: str, rise) -> str:
    if rise is None or text == '':
        return text
    if rise < 0:
        return f'<sub>{text}</sub>'
    if rise > 0:
        return f'<sup>{text}</sup>'
    return text


def wrap_font(text: str, font: str, tracking=None) -> str:
    # 批9质检修正：<sp> 疏排纳入序列化白名单
    inner = text
    # 批8质检修正：序列化保留全部字体层级（此前只留 <i>，粗体/粗斜体静默降级）
    if font == 'BI':
        inner = f'<b><i>{text}</i></b>'
    elif font == 'B':
        inner = f'<b>{text}</b>'
    elif font == 'I':
        inner = f'<i>{text}</i>'
    if tracking == 'auto':
        return f'<sp>{inner}</sp>'
    return inner


def _migrate_spec_keys(spec: dict, flags: list[str]) -> None:
    """批14共性3：配置键层级自愈——块级 size_pt 下移栏级、栏级 ws_checked 上移块级。
    静默吞键曾浪费整轮渲染（074_left size_pt 写块级、075_left ws_checked 写栏级）。"""
    for b in spec.get('blocks', []):
        if 'size_pt' in b:
            for cb in b.get('columns', []):
                cb.setdefault('size_pt', b['size_pt'])
            flags.append(f"⚠ 块{b['index']}: size_pt 误写在块级，已自动下移各栏（应写 columns[].size_pt）")
            del b['size_pt']
        ups = [ci for ci, cb in enumerate(b.get('columns', [])) if 'ws_checked' in cb]
        if ups:
            merged = set(b.get('ws_checked', []))
            for ci in ups:
                merged |= set(b['columns'][ci].pop('ws_checked'))
            b['ws_checked'] = sorted(merged)
            flags.append(f"⚠ 块{b['index']}: ws_checked 误写在栏级，已自动上移合并（应写在块级）")


def build(uid: str, root: Path | None = None) -> dict:
    """从 行文本.txt + 页面配置.json 渲染单页 PDF、对照图并跑程序验收。"""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader
    d = page_dir(uid, root)
    input_hashes = artifact_hashes(d, BINDING_FILES[:3])
    spec = json.loads((d / '页面配置.json').read_text())
    flags: list[str] = []
    _migrate_spec_keys(spec, flags)
    # 批17共性2①：页眉线断言进渲染路径——复核员重写配置/重渲染也会复查
    # （086_right 曾被整页重写静默丢线且无 flag）；有页眉文本带而页眉区
    # （页眉块底边+60px）无 5 元细线规则（厚≤4px）即 ⚠
    hdr_blks17 = [b2 for b2 in spec.get('blocks', []) if b2.get('label') == 'Page-Header']
    if hdr_blks17:
        hdr_zone17 = max(b2['bbox'][3] for b2 in hdr_blks17) + 60
        if not any(len(r) >= 5 and r[4] <= 4 and r[1] <= hdr_zone17
                   for r in spec.get('rules', [])):
            flags.append('⚠ 页眉区无细线规则（重渲染复查），若原书有线请实测补 rules')
    src = Image.open(d / '源页.png').convert('RGB')
    gray = np.asarray(src.convert('L'))
    W, H = src.size
    lines = [l for l in (d / '行文本.txt').read_text().splitlines() if l.strip()]
    groups: dict[int, list[str]] = {}
    head = None
    for l in lines:
        if l.startswith('#'):
            head = int(re.match(r'# 块(\d+)', l).group(1)) if re.match(r'# 块(\d+)', l) else None
            groups.setdefault(head, [])
        else:
            groups[head if head is not None else -1].append(l)
    register_fonts()
    pdf = d / '原位排印校样.pdf'
    c = canvas.Canvas(str(pdf), pagesize=(W * SCALE, H * SCALE))
    c.setTitle(f'{uid} 原位排印校样：脚本草稿，待模型逐行复核')
    records = []
    images_embedded = []
    rules = spec.get('rules')
    if rules is None:  # 旧配置无 rules：现场检测（避开插图块）
        img_boxes = [b['bbox'] for b in spec['blocks'] if b.get('kind') == 'image']
        rules = find_rules(gray, [0, 0, W, H])
        rules = [r for r in rules
                 if not any(not (r[2] < ib[0] or r[0] > ib[2] or r[3] < ib[1] or r[1] > ib[3])
                            for ib in img_boxes)]
    for r in rules:  # 页眉下划线/装饰横线（批1质检补）；批4：5元=斜线段原语
        if len(r) == 5:
            rx0, ry0, rx1, ry1, thick = r
            c.setLineWidth(max(0.3, thick * SCALE))
            c.setStrokeColorRGB(0, 0, 0)
            c.line(rx0 * SCALE, (H - ry0) * SCALE, rx1 * SCALE, (H - ry1) * SCALE)
            continue
        rx0, ry0, rx1, ry1 = r
        c.setFillColorRGB(0, 0, 0)
        c.rect(rx0 * SCALE, (H - ry1) * SCALE, (rx1 - rx0) * SCALE,
               max(0.3, (ry1 - ry0) * SCALE), stroke=0, fill=1)
    for b in spec['blocks']:
        if b['kind'] == 'image':
            box = b['bbox']
            fig = src.crop(tuple(box))
            fp = d / f'插图_块{b["index"]}.png'
            fig.save(fp)
            c.drawImage(ImageReader(fig), box[0] * SCALE, (H - box[3]) * SCALE,
                        width=fig.width * SCALE, height=fig.height * SCALE)
            images_embedded.append({'block': b['index'], 'bbox': box, 'path': fp.name})
            continue
        # 墨带按栏序（栏内按 y），与 fill_all 的行文本顺序一一对应；带栏号防跨栏测窗
        bands_all: list[tuple[dict, int]] = []
        for ci2, cb in enumerate(b['columns']):
            bands_all.extend((bd, ci2) for bd in sorted(cb['bands'], key=lambda bd: bd['y0']))
        block_lines = groups.get(b['index'], [])
        n = min(len(block_lines), len(bands_all))
        real_lines = [l for l in block_lines if not l.startswith('⟦空带')]
        if len(real_lines) != len(bands_all):  # 批8：占位行不计（渲染端已跳过，口径对齐）
            flags.append(f"块{b['index']}: 行文本{len(real_lines)}行对墨带{len(bands_all)}行")
        # 每带所属栏的字号（多栏时各栏可不同）
        sizes = []
        for cb in b['columns']:
            sizes.extend([cb.get('size_pt', 11.2)] * len(cb['bands']))
        guesses = []
        for cb in b['columns']:
            guesses.extend([cb.get('guess_slope', 0.012)] * len(cb['bands']))
        for k in range(n):
            markup = block_lines[k]
            if markup == '⟦空带⟧':
                continue
            bd, col_i = bands_all[k]
            size = sizes[k]
            guess = guesses[k]
            cap_px = int(0.74 * size / SCALE)
            # 批2质检修正：below 只取同栏下一带，防跨栏测窗退化
            nxt_same = next((bd2['y0'] for bd2, c2 in bands_all[k + 1:] if c2 == col_i), None)
            below = min(nxt_same - 4 if nxt_same is not None else bd['y1'] + 40,
                        bd['y1'] + 40, b['bbox'][3] + 2)  # 批9：块界封顶，越块不再圈墨
            xs = [max(0, bd['x0'] - 12), min(W, bd['x1'] + 12)]
            x0, x1, base, slope = measure_line(gray, bd['y0'], below, xs, guess, cap_px)
            if 'x0_px' in bd:  # 人工确认的行首左界（如排除竖栏线旁孤立墨点），与墨带框分别保存
                x0 = float(bd['x0_px'])
            if 'baseline_px' in bd:  # 人工确认的基线与真实墨带边界分别保存，避免为定位伪造带框。
                base = float(bd['baseline_px'])
            spans = markup_spans(markup, size)
            plain = ''.join(s['text'] for s in spans)
            auto_chars = sum(len(s['text']) for s in spans if s.get('tracking') == 'auto')
            has_sp = auto_chars > 0
            if auto_chars:
                # 批2质检修正：扣除全部 span 自然宽（含疏排段自身），字距按 auto 字符数-1 分配
                all_nat = sum(pdfm_width(s2) for s2 in spans)
                sp_track = ((x1 - x0) * SCALE + 1.5 - all_nat) / max(auto_chars, 1)  # 批9：分母与施加对象一致
                # 批6质检修正：疏排幅度钳位 0~0.5em（原书 разрядка≈0.25em），超额由词距吸收
                sp_track = min(max(sp_track, 0.0), 0.5 * size)
                for s2 in spans:
                    if s2.get('tracking') == 'auto':
                        s2['tracking'] = sp_track
            # 页眉（运行标题）按原墨宽反算疏排，字隙保持为正
            if b['label'] == 'Page-Header' and len(plain) >= 4 and not plain.isdigit():
                hslack = ((x1 - x0) * SCALE + 2.6 - sum(pdfm_width(s) for s in spans)) / len(plain)
                for s_h in spans:  # 批11：按 span 分摊（混字体页眉分段疏排）
                    if s_h.get('tracking') != 'auto':
                        s_h['tracking'] = hslack
            natural = sum(pdfm_width(s) + s.get('tracking', 0) * len(s['text']) for s in spans)
            nsp = sum(s['text'].count(' ') for s in spans)
            nch = sum(len(s['text']) for s in spans)
            target = (x1 - x0) * SCALE
            ws = (target - natural) / nsp if nsp else 0.0
            cs = 0.0
            minimum = 1.25 - size * 0.25
            if ws < minimum:
                # 批3质检修正：元组右侧曾引用钳制前旧 ws，导致钳制行渲染宽出带宽
                ws = minimum
                cs = (target - natural - nsp * ws) / max(nch, 1)
            eff = size * 0.25 + ws + cs
            if (eff < 0.8 or abs(cs) > 0.25) and (k + 1) not in b.get('ws_checked', []):
                flags.append(f"块{b['index']}第{k + 1}行: 有效空白{eff:.2f}或字距{cs:.2f}需复核")
            from reportlab.pdfbase import pdfmetrics as _pm
            eff_space = _pm.stringWidth(' ', 'R', size) + ws
            if eff_space < 0.4:
                flags.append(f"块{b['index']}第{k + 1}行: 实际空格宽{eff_space:.2f}pt过窄（字面偏宽或断行不符）")
            elif ws > 0.30 * size:
                if '....' in plain or re.search(r'(\.\s*){8,}', plain):
                    pass  # 点线引导行词距天然大，不按断行误报
                elif (has_sp or any(isinstance(sp2.get('tracking'), float) and sp2['tracking'] > 0.5 for sp2 in spans)) and (k + 1) not in b.get('ws_checked', []):
                    flags.append(f"块{b['index']}第{k + 1}行: 疏排行词距{ws:.1f}pt，请对照原书疏排节奏")
                elif (k + 1) not in b.get('ws_checked', []):
                    # 批3质检修正：中性措辞（断行指控从未成立）；已核行（ws_checked）不再复活
                    flags.append(f"块{b['index']}第{k + 1}行: 词距{ws:.1f}pt偏大，请对照原书疏行节奏: {markup[:40]}")
            draw_text(c, x0, base, slope, spans, ws, cs, H)
            records.append({'block': b['index'], 'line': k + 1, 'markup': markup,
                            'text': ''.join(s['text'] for s in spans),
                            'bbox': [x0, bd['y0'], x1, bd['y1']], 'col': col_i,
                            'baseline_left': base, 'slope': slope,
                            'size': size, 'word_space': ws, 'char_space': cs,
                            'rendered_right_px': x0 + (natural + nsp * ws + nch * cs) / SCALE})
    c.showPage()
    c.save()
    # ---- 渲染与验收
    subprocess.run(['pdftoppm', '-png', '-singlefile', '-scale-to-x', str(W),
                    '-scale-to-y', str(H), str(pdf), str(d / '新排')],
                   check=True, capture_output=True)
    rendered = Image.open(d / '新排.png').convert('RGB')
    comb = Image.new('RGB', (W * 2 + 20, H), '#eeeeee')
    comb.paste(src, (0, 0))
    comb.paste(rendered, (W + 20, 0))
    comb.save(d / '原图新排对照.png')
    # 方案四：预裁剪切片（宽≤1400），复核员直接读图，无需 PIL
    cw = 1400
    ch = int(H * cw / comb.width)
    small = comb.resize((cw, ch), Image.LANCZOS)
    n_slice = max(2, min(4, (ch + 899) // 900))
    for i in range(n_slice):
        y0s = int(ch * i / n_slice)
        y1s = int(ch * (i + 1) / n_slice)
        small.crop((0, y0s, cw, y1s)).save(d / f'对照_切片{i + 1}.png')
    extracted = PdfReader(pdf).pages[0].extract_text() or ''
    norm = lambda t: ''.join(t.split())
    missing = [r['line'] for r in records if norm(r['text']) not in norm(extracted)]
    ren_gray = np.asarray(rendered.convert('L'))
    offsets, collisions, window_trunc = [], [], []
    all_ren_boxes: list[dict] = []  # 批20共性1：全页渲染盒，跨块碰撞检测用
    prevs: dict[tuple, dict] = {}  # (block, col) -> 上一行渲染底缘；批2质检：碰撞限同栏
    # 批3质检修正：右窗需按右邻元素封顶（同 y 的邻栏带/插图 x0-2），防圈入邻栏墨迹
    all_boxes = [(int(r2['bbox'][0]), int(r2['bbox'][1]), int(r2['bbox'][3])) for r2 in records] +                 [(int(ib2[0]), int(ib2[1]), int(ib2[3])) for ib2 in
                 (im['bbox'] for im in images_embedded)]
    for r in records:
        x0, y0, x1, y1 = [int(v) for v in r['bbox']]
        wx0 = max(0, x0 - 30)
        near2em = x1 + max(30, int(2 * r['size'] / SCALE))
        obstacles = [bx for bx, by0, by1 in all_boxes
                     if bx > x1 + 4 and not (by1 <= y0 or by0 >= y1 + 2)]
        wx1 = min(W, near2em, min(obstacles) - 2) if obstacles else min(W, near2em)

        def ibox(arr, keep_from=None, keep_to=None):
            # 行带内紧致窗口（带间本有>=4px 空隙，越窗会把邻行墨迹算进来）
            # 批15共性10：低矮带(h<12px，图注/分数行)按字形全高扩窗±4-6px，
            # 否则升降部墨被裁出窗 → dx/dy 伪差 ±17-109px
            ry0 = max(0, y0 - 4) if (y1 - y0) < 12 else y0
            ry1 = min(arr.shape[0], y1 + 6) if (y1 - y0) < 12 else y1 + 2
            ink = arr[ry0:ry1, wx0:wx1] < 150
            rows = np.flatnonzero(ink.sum(1) >= 1)
            if len(rows):
                rows = rows + (ry0 - y0)  # 保持相对 y0（下游 + y0 还原）
            cols = np.flatnonzero(ink.sum(0) >= 1)
            if not len(rows):
                return None
            # 批5质检修正：两侧统一锚定行带界（疏排多簇不再塌缩到单一词块），只剔带外尘簇
            if keep_from is not None and len(cols) >= 2:
                keep = cols[(cols + wx0 >= keep_from - 10) & (cols + wx0 <= keep_to + 10)]
                if len(keep):
                    cols = keep
            return (int(cols.min()) + wx0, int(rows.min()) + y0,
                    int(cols.max()) + wx0, int(rows.max()) + y0)
        br = ibox(ren_gray, keep_from=x0, keep_to=x1)
        bo = ibox(gray, keep_from=x0, keep_to=x1)
        key = (r['block'], r.get('col', 0))
        pv = prevs.get(key)
        if bo and br:
            o = {'block': r['block'], 'line': r['line'],
                 'dx_left': br[0] - bo[0], 'dx_right': br[2] - bo[2],
                 'dy_top': br[1] - bo[1]}
            if abs(o['dx_left']) > 25 or abs(o['dy_top']) > 8:
                # 批13共性5：离群墨几何自动分类——上行降部入窗归非差错；
                # 行首左侧孤立薄墨(≤6px)归尘点/纸痕；其余保持人工判读
                o['note'] = '疑似杂墨/邻行降部/出窗，先排除非行墨再判错'
                if o['dy_top'] > 8 and pv and bo[1] <= int(pv['bottom']) + 8:
                    o['note'] = f'自动归类：上行降部入窗（上行渲染底≈y{int(pv["bottom"])}），非本行差错'
                    o['auto_class'] = 'descender'
                elif o['dx_left'] > 25:
                    ink_l = gray[y0:min(gray.shape[0], y1 + 2), wx0:wx1] < 150
                    zone = ink_l.any(0)[:max(1, br[0] - 5 - wx0)]
                    on_l = np.flatnonzero(zone)
                    if len(on_l):
                        splits_l = np.flatnonzero(np.diff(on_l) > 10)
                        first_cl = np.split(on_l, splits_l + 1)[0]
                        if first_cl[-1] - first_cl[0] <= 6:
                            o['note'] = '自动归类：行首左侧孤立薄墨(宽≤6px)，尘点/纸痕，待查'
                            o['auto_class'] = 'dust'
            offsets.append(o)
            if br[2] >= wx1 - 1:
                window_trunc.append({'block': r['block'], 'line': r['line'],
                                     'note': '渲染墨迹触右测窗界，疑似溢出被截断'})
            if pv and br[1] < pv['bottom'] - 2:
                collisions.append({'block': r['block'], 'line': r['line'],
                                   'overlap_px': int(pv['bottom'] - br[1])})
            prevs[key] = {'bottom': br[3]}
            all_ren_boxes.append({'block': r['block'], 'line': r['line'],
                                  'x0': br[0], 'y0': br[1], 'x1': br[2], 'y1': br[3]})
    # 批20共性1：跨块叠印——原碰撞门只比同块同栏前带，102_left 块8↔块9 漏检
    all_ren_boxes.sort(key=lambda b3: (b3['y0'], b3['x0']))
    for _i3 in range(len(all_ren_boxes)):
        a3 = all_ren_boxes[_i3]
        for _j3 in range(_i3 + 1, len(all_ren_boxes)):
            b4 = all_ren_boxes[_j3]
            if b4['y0'] > a3['y1'] - 2:  # y0 升序，其后不可能 y 叠>2px
                break
            if a3['block'] == b4['block']:
                continue
            x_ov = min(a3['x1'], b4['x1']) - max(a3['x0'], b4['x0'])
            y_ov = min(a3['y1'], b4['y1']) - max(a3['y0'], b4['y0'])
            if y_ov > 2 and x_ov > 0.3 * min(a3['x1'] - a3['x0'], b4['x1'] - b4['x0']):
                collisions.append({'block': b4['block'], 'line': b4['line'],
                                   'overlap_px': int(y_ov),
                                   'note': f"跨块叠印：与块{a3['block']}第{a3['line']}行 x 重叠且 y 叠 {int(y_ov)}px"})
    fig_checks = []
    for im in images_embedded:
        box = im['bbox']
        ok_pre = np.array_equal(np.asarray(src.crop(tuple(box))),
                                np.asarray(Image.open(d / im['path'])))
        fig_checks.append({'block': im['block'], 'preembed_equal': bool(ok_pre)})
    checks = {'pages': 1, 'text_lines': len(records), 'images': len(images_embedded),
              'pending_fill_flags': len(flags),  # 批18共性3：真源=未决清单.json，防报告.json恒空间接误读
              'missing_text_lines': missing,
              'latex_residual': sorted({r['line'] for r in records
                                        if any(s in r['text'] for s in
                                               ('E_{', 'EP_', '\\text{', 'Ep_', '\\mu', '\\varphi', 'E_{p', 'Ер <sub', 'Р <sub'))}),
              'overflow_lines': [r['line'] for r in records if r['rendered_right_px'] > W],
              'overflow_band_lines': [{'block': r['block'], 'line': r['line'],
                                       'beyond_px': round(r['rendered_right_px'] - r['bbox'][2], 1)}
                                      for r in records if r['rendered_right_px'] > r['bbox'][2] + 12],
              'measure_window_truncated': window_trunc,
              'min_effective_space': round(min((r['word_space'] + r['size'] * 0.25 + r['char_space']
                                                for r in records), default=0), 2),
              'ink_offsets_max': ({'dx_left': max((abs(o['dx_left']) for o in offsets), default=0),
                                   'dx_right': max((abs(o['dx_right']) for o in offsets), default=0),
                                   'dy_top': max((abs(o['dy_top']) for o in offsets), default=0)} if offsets else {}),
              'ink_offsets': offsets,
              'line_collisions': collisions,
              'figures_preembed_equal': all(f['preembed_equal'] for f in fig_checks) if fig_checks else None,
              'font_missing_glyphs': _glyph_check(extracted)}

    (d / '程序验收.json').write_text(json.dumps(checks, ensure_ascii=False, indent=1))
    (d / '逐行坐标样式.json').write_text(json.dumps(
        {'uid': uid, 'records': records}, ensure_ascii=False, indent=1))
    review = '未开始（脚本草稿）'
    if (d / '复核记录.json').exists():  # 批1质检修正：不再抹掉复核结论
        try:
            rr = json.loads((d / '复核记录.json').read_text())
            review = {'结论': rr.get('结论', '已复核'), '修订': rr.get('修订'),
                      '遗留': rr.get('遗留', rr if isinstance(rr, list) else []),
                      '来源': '复核记录.json'}
        except Exception:
            review = '已复核（复核记录.json 存在但解析失败，以该文件为准）'
    (d / '未决清单.json').write_text(json.dumps(
        {'fill_flags': flags, 'layout_flags': checks, '复核': review},
        ensure_ascii=False, indent=1))
    hashes = artifact_hashes(d)
    if any(hashes[name] != digest for name, digest in input_hashes.items()):
        raise RuntimeError('渲染期间页面输入发生变更，拒绝绑定此版本；请重新渲染')
    atomic_write(d / '技术预筛绑定.json', json.dumps(
        {'uid': uid, 'scope': '仅技术预筛，不代表人工终验', 'hashes': hashes,
         'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        ensure_ascii=False, indent=1))
    return checks


def _glyph_check(extracted: str):
    """批2质检修正：用已注册 Times 字体的 cmap 实查缺字形，无接口则如实回退。"""
    try:
        from reportlab.pdfbase import pdfmetrics
        missing = set()
        for ch in set(extracted):
            if ch.isspace():
                continue
            for fname in ('R', 'I'):
                face = pdfmetrics.getFont(fname).face
                cmap = getattr(face, 'charToGlyph', None) or getattr(face, 'cmap', None)
                if cmap is not None and ord(ch) not in cmap:
                    missing.add(ch)
        return sorted(missing) if missing else '通过（提取文本码点均在 Times R/I cmap 内）'
    except Exception as e:
        return f'未检查（cmap 接口不可用：{e}）'


def pdfm_width(s: dict) -> float:
    from reportlab.pdfbase import pdfmetrics
    return pdfmetrics.stringWidth(s['text'], s['font'], s['size'])

# ---------------------------------------------------------------- 失败登记


def log_failure(uid: str, stage: str, message: str) -> None:
    rec = {'ts': time.strftime('%F %T'), 'uid': uid, 'stage': stage,
           'message': message[:2000], 'status': 'open', 'attempts': 1}
    rows = []
    if REGISTRY.exists():
        rows = [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()]
    for r in rows:
        if r['uid'] == uid and r['status'] == 'open':
            r['attempts'] += 1
            r['message'] = rec['message']
            rec = None
            break
    if rec:
        rows.append(rec)
    REGISTRY.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')


def mark_status(uid: str, status: str) -> None:
    if not REGISTRY.exists():
        return
    rows = [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()]
    for r in rows:
        if r['uid'] == uid:
            r['status'] = status
    REGISTRY.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')


def make_page(uid: str, root: Path | None = None, force: bool = False) -> dict:
    """完整驱动：草稿（可跳过）+ 渲染 + 验收；异常记入失败登记。

    force=False 且配置/行文本已存在时直接用现有文件重建（供模型改完
    行文本.txt 或 页面配置.json 后重渲染，不覆盖人工修订）。
    """
    d = page_dir(uid, root)
    t0 = time.time()
    try:
        check_runtime()
        # 批2质检修正：已复核且 行文本.txt 比配置新 → 拒绝 --force（防陈旧 paragraphs 覆盖人工修订）
        if force and (d / '复核记录.json').exists() and (d / '行文本.txt').exists() \
                and (d / '页面配置.json').exists() \
                and (d / '行文本.txt').stat().st_mtime > (d / '页面配置.json').stat().st_mtime:
            raise RuntimeError(f'{uid} 已复核且行文本含人工修订，拒绝 --force；'
                               f'确要重建请先归档 行文本.txt/复核记录.json')
        info = None
        flags = []
        if force or not (d / '页面配置.json').exists():
            info = draft(uid, root)
        if info or not (d / '行文本.txt').exists():
            flags = fill_all(uid, root)
        checks = build(uid, root)
        report = {'uid': uid, 'status': 'ok', 'seconds': round(time.time() - t0, 1),
                  'draft': info, 'checks': checks}
        if flags:
            report['fill_flags'] = flags
        atomic_write(d / '报告.json', json.dumps(report, ensure_ascii=False, indent=1))
        mark_status(uid, 'fixed')
        return report
    except Exception as e:
        msg = f'{e}\n{traceback.format_exc()[-1200:]}'
        log_failure(uid, 'make_page', msg)
        report = {'uid': uid, 'status': 'fail', 'stage': 'make_page',
                  'error': msg[:800], 'seconds': round(time.time() - t0, 1)}
        try:
            atomic_write(d / '报告.json', json.dumps(report, ensure_ascii=False, indent=1))
        except Exception:
            pass
        return report
