"""切分策略：S1 固定窗口 / S2 递归 / S4 父子块（S3 语义切分 M2 实现）。

设计要点
- 四种策略共用同一个 token 计数器与同一个累加器 `windows()`（在 textutil 里），
  差异只在「切分单元」：S1 = 句子单元，S2 = 段落单元（超长段落降级为句子），
  S4 = 子块句子 + 父块网格。这样对比实验里唯一的变量就是切分策略本身。
- 每个 chunk 保留原文字符区间 (start,end)，用于映射评测块（blocks.py）与父子回溯。
"""
from __future__ import annotations

import re

from rag.ingest.blocks import build_block_grid, span_index
from rag.textutil import count_tokens, has_indexable_text, windows
from rag.types import Chunk, Document

SUPPORTED = ("fixed512", "recursive", "parentchild", "semantic")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)


def _headings(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(text)]


def _section_at(headings: list[tuple[int, str]], offset: int) -> str:
    cur = ""
    for pos, title in headings:
        if pos > offset:
            break
        cur = title
    return cur


def chunk_document(doc: Document, strategy: str, params: dict, eval_block_tokens: int = 1024) -> list[Chunk]:
    if strategy not in SUPPORTED:
        raise ValueError(f"unknown strategy: {strategy} (supported: {SUPPORTED})")
    if strategy == "semantic":
        raise NotImplementedError("S3 语义切分在 M2 实现（PRD §8）")

    grid = build_block_grid(doc.text, eval_block_tokens)
    heads = _headings(doc.text)

    parent_grid: list[tuple[int, int]] | None = None
    if strategy == "parentchild":
        parent_grid = windows(doc.text, int(params.get("parent", 1024)), 0, "sentence")
        spans = windows(
            doc.text,
            int(params.get("child", 256)),
            int(params.get("overlap", 32)),
            "sentence",
        )
    else:
        spans = windows(
            doc.text,
            int(params["size"]),
            int(params.get("overlap", 0)),
            "sentence" if strategy == "fixed512" else "paragraph",
        )

    chunks: list[Chunk] = []
    for s, e in spans:
        raw = doc.text[s:e]
        if not raw.strip():
            continue
        # 纯标点块（表格分隔线等）没有 BM25 词项，稀疏向量会全零，Milvus 直接拒收；
        # 它们也不承载检索信息，直接丢弃，序号仍保持连续。
        if not has_indexable_text(raw):
            continue
        cid = f"{strategy}_{doc.doc_id}_{len(chunks):04d}"
        parent_id = f"{doc.doc_id}:p{span_index(parent_grid, s, e)}" if parent_grid is not None else cid
        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc.doc_id,
                strategy=strategy,
                text=raw.strip(),
                start=s,
                end=e,
                token_len=count_tokens(raw),
                eval_block_id=f"{doc.doc_id}:b{span_index(grid, s, e)}",
                parent_id=parent_id,
                meta={
                    "source": doc.source,
                    "title": doc.title,
                    "section": _section_at(heads, s),
                },
            )
        )
    return chunks
