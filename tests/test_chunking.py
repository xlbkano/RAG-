"""切分器与评测块网格的单测。

重点验证三件事：
1. 窗口不超 token 预算（除单个超长单元外）；
2. 所有 span 都落在原文范围内且能还原文本；
3. 评测块映射与策略无关、稳定可复现。
"""
from __future__ import annotations

from rag.ingest.chunker import chunk_document
from rag.ingest.blocks import build_block_grid, span_index
from rag.textutil import count_tokens, iter_sentence_spans, windows
from rag.types import Document

DOC = Document(
    doc_id="d1",
    source="test.md",
    title="测试文档",
    text=(
        "# 一级标题\n\n"
        "第一段内容。它有两句话。\n\n"
        "第二段讲另一件事，句子稍微长一些，包含中文与 English words 混排的情况。\n\n"
        + "填充句子。" * 200
    ),
)


def test_sentence_spans_cover_text():
    spans = list(iter_sentence_spans(DOC.text))
    assert spans[0][0] == 0
    assert spans[-1][1] == len(DOC.text)
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert s2 >= e1, "句子区间必须单调不重叠"


def test_windows_respect_budget():
    for u in ("sentence", "paragraph"):
        for s, e in windows(DOC.text, size=64, overlap=0, unit=u):
            # 允许单个无法再切的单元超出，但不应超过 2 倍预算
            assert count_tokens(DOC.text[s:e]) <= 128


def test_windows_overlap_monotonic():
    spans = windows(DOC.text, size=64, overlap=16, unit="sentence")
    assert spans[0][0] == 0
    for (s1, _), (s2, _) in zip(spans, spans[1:]):
        assert s2 >= s1, "窗口起点必须单调前进，否则会死循环"


def test_eval_block_grid_is_strategy_independent():
    grid = build_block_grid(DOC.text, 128)
    assert span_index(grid, 0, 10) == 0
    last = span_index(grid, len(DOC.text) - 5, len(DOC.text))
    assert last == len(grid) - 1


def test_fixed_and_recursive_produce_chunks():
    params = {"size": 64, "overlap": 0}
    a = chunk_document(DOC, "fixed512", params, eval_block_tokens=128)
    b = chunk_document(DOC, "recursive", {**params, "overlap": 16}, eval_block_tokens=128)
    assert a and b
    for c in a + b:
        assert c.chunk_id.startswith(("fixed512_", "recursive_"))
        assert c.text.strip()
        assert c.eval_block_id.startswith("d1:b")
        assert DOC.text[c.start : c.end].strip() == c.text


def test_parentchild_links_to_parent_block():
    chunks = chunk_document(DOC, "parentchild", {"child": 48, "parent": 160, "overlap": 8}, 128)
    assert chunks
    parents = {c.parent_id for c in chunks}
    assert all(p.startswith("d1:p") for p in parents)
    # 子块数应多于父块数
    assert len(chunks) > len(parents)
