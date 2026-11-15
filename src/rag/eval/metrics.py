"""检索指标。全部在**评测块（eval_block）粒度**上计算（PRD §3 / §12.3）。

两个容易算错、且会直接让数字虚高的地方，这里都做了处理：

1. **先按块去重，再取 top-k。**
   父子块策略下一个父块会对应 4 个子块，如果直接对 chunk 列表取 top-5 再映射成块，
   同一个块可能占掉 5 个名额中的 4 个——看起来"召回 4 条"，实际只召回 1 个块。
   反过来，如果不去重就统计"命中数/5"，父子块会天然占优。正确做法是把命中序列
   映射成**块序列**（保留首次出现顺序、去重），再取前 k 个块。

2. **在块粒度上算 Recall，而不是 chunk 粒度。**
   chunk 粒度下，切得越碎的策略 block 覆盖越集中，Recall 越容易虚高（PRD §3 已论证）。

未命中时各指标返回 0.0，不是 None——聚合时按 0 参与平均，否则会高估。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def ranked_blocks(block_ids: Iterable[str]) -> list[str]:
    """把命中序列映射成去重后的块序列，保留首次出现顺序。"""
    seen: set[str] = set()
    out: list[str] = []
    for b in block_ids:
        if not b or b in seen:
            continue
        seen.add(b)
        out.append(b)
    return out


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """命中 gold 块的比例。gold 为空（无答案题）时返回 0，调用方应将其排除在聚合外。"""
    if not gold:
        return 0.0
    got = set(ranked[:k]) & gold
    return len(got) / len(gold)


def hit_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """top-k 里是否至少有一个 gold 块。"""
    if not gold:
        return 0.0
    return 1.0 if set(ranked[:k]) & gold else 0.0


def mrr_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    """第一个 gold 块的倒数排名（1-based）。"""
    if not gold:
        return 0.0
    for i, b in enumerate(ranked[:k], start=1):
        if b in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: dict[str, float], k: int) -> float:
    """分级相关的 nDCG。gains 里没出现的块按 0 计。

    用 2^rel - 1 作为增益，与 PRD §12.1 的定义一致（rel ∈ {0,1,2}）。
    """
    ideal = sorted(gains.values(), reverse=True)[:k]
    if not ideal or ideal[0] <= 0:
        return 0.0
    dcg = 0.0
    for i, b in enumerate(ranked[:k], start=1):
        rel = gains.get(b, 0.0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数。样本少时 p95 会被拉到接近最大值，这是诚实的行为——
    15 个样本算不出可信的 P99，宁可让它偏保守。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
