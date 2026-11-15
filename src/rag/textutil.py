"""文本工具：token 估算、句子切分。

关于 count_tokens：不用真实 tokenizer，因为切分只需要**相对可比**的预算，而且
切片时会反复对子串计数（二分找边界），逐次调 tokenizer 是 O(n²) 的。

但「估算」不等于「拍脑袋」。下面的系数是用语料实测拟合出来的：随机抽 300 个
窗口，用 bge-small-zh 的真实 tokenizer 数出 token 数，对
（中文字符数，ASCII 字符数，其他非空白字符数）做最小二乘：

    R² = 0.98，单窗口相对误差中位数 3.5%，全语料总 token 偏差 -0.3%

早期版本用的是手写规则「中文 1 token + ASCII 单词 1.3 token + 其余 0」，
问题有两个：ASCII 部分误差 -28%～-35%（`PNGStreamingResponse` 只算 1.3 token，
真实是 6）；标点被计 0，使「------」这类表格行在窗口累加时免费，块可以无限膨胀，
实测出现过 17153 字符的评测块。块过大 → 一块覆盖多个子主题 → 各种切分的 chunk
都能映射到它 → Recall 虚高且失去区分度。这是评测口径的 bug，不是切分策略的差异。
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_OTHER = re.compile(r"[^\s\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7afA-Za-z0-9_]")
_BREAKS = set("。！？!?；;\n")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    ascii_chars = sum(len(w) for w in _WORD.findall(text))
    others = len(_OTHER.findall(text))
    return int(round(0.968 * cjk + 0.220 * ascii_chars + 1.397 * others))


def has_indexable_text(text: str) -> bool:
    """块里是否含可被 BM25 索引的词项（中文或 ASCII 词）。

    纯标点/表格分隔线（如 `|---|---|`）在段落切分下会独立成块，这类块没有词项，
    稀疏向量全零 —— Milvus 会直接以 "sparse vector must contain at least one
    nonzero float32 weight" 拒绝整批插入。它们本来也不承载检索信息，入库前丢弃。
    """
    return bool(_CJK.search(text) or _WORD.search(text))


def iter_sentence_spans(text: str, max_len: int = 400):
    """产出 (start, end) 句子区间。max_len 防止无标点的代码块把整篇吞成一个句子。"""
    n = len(text)
    start = 0
    i = 0
    while i < n:
        if text[i] in _BREAKS:
            j = i
            while j < n and text[j] in _BREAKS:
                j += 1
            if j > start:
                yield (start, j)
            start = j
            i = j
        elif i - start >= max_len:
            yield (start, i + 1)
            start = i + 1
            i += 1
        else:
            i += 1
    if start < n:
        yield (start, n)


def iter_paragraph_spans(text: str):
    """按空行切段落，保留偏移。"""
    pos = 0
    for block in re.split(r"(\n\s*\n)", text):
        if block and not re.fullmatch(r"\n\s*\n", block):
            s = text.find(block, pos)
            if s >= 0:
                yield (s, s + len(block))
                pos = s + len(block)


def slice_by_token_budget(text: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """把 [start,end) 硬切成每段 ≤ size token 的片段（二分找最长前缀）。"""
    out: list[tuple[int, int]] = []
    s = start
    while s < end:
        if count_tokens(text[s:end]) <= size:
            out.append((s, end))
            break
        lo, hi = s + 1, end
        best = s + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(text[s:mid]) <= size:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        out.append((s, best))
        s = best
    return out


def windows(text: str, size: int, overlap: int = 0, unit: str = "sentence") -> list[tuple[int, int]]:
    """通用累加窗口：按 unit 累积到 size token，回退 overlap token 制造重叠。

    切分策略与评测块网格共用这一个累加器 —— 保证「切分单元」是唯一变量。
    """
    if not text.strip():
        return []

    if unit == "sentence":
        base = list(iter_sentence_spans(text))
    elif unit == "paragraph":
        base = []
        for s, e in iter_paragraph_spans(text):
            if count_tokens(text[s:e]) > size:
                base.extend(slice_by_token_budget(text, s, e, size))
            else:
                base.append((s, e))
    else:
        raise ValueError(f"unknown unit: {unit}")

    # 二次保险：仍超预算的单元硬切（例如无标点长代码块）
    units: list[tuple[int, int]] = []
    for s, e in base:
        if count_tokens(text[s:e]) > size:
            units.extend(slice_by_token_budget(text, s, e, size))
        else:
            units.append((s, e))
    if not units:
        return []

    toks = [count_tokens(text[s:e]) for s, e in units]
    out: list[tuple[int, int]] = []
    i, n = 0, len(units)
    while i < n:
        total, j = 0, i
        while j < n and (j == i or total + toks[j] <= size):
            total += toks[j]
            j += 1
        out.append((units[i][0], units[j - 1][1]))
        if j >= n:
            break
        # 按整单元回退 overlap
        back, k = 0, j
        while k > i + 1 and back < overlap:
            k -= 1
            back += toks[k]
        i = k if k > i else j
    return out
