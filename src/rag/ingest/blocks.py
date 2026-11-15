"""与切分策略无关的「评测块网格」。

这是整个评测公平性的地基（PRD §3 / §12.3）：
- 若按各策略自己的 chunk 粒度算 Recall，父子块策略会因"相关内容被切成 4 块、
  命中 1 块即算命中"而系统性占优，数字虚高。
- 因此对每篇文档额外铺一层固定的 1024-token 网格（block），所有 Recall 都在
  这个粒度上算，四种切分策略才可比。
- 任何 chunk 通过「字符区间最大重叠」映射到一个 block；block id 形如 `docid:b3`。
"""
from __future__ import annotations

from rag.textutil import windows


def build_block_grid(text: str, block_tokens: int) -> list[tuple[int, int]]:
    return windows(text, block_tokens, overlap=0, unit="sentence")


def span_index(grid: list[tuple[int, int]], start: int, end: int) -> int:
    """返回与 [start,end) 字符重叠最多的网格下标。"""
    best_i, best_ov = 0, -1
    for i, (s, e) in enumerate(grid):
        ov = min(e, end) - max(s, start)
        if ov > best_ov:
            best_ov, best_i = ov, i
    return best_i
