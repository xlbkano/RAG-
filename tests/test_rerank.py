"""重排接入的单测：用桩替掉 Milvus 与模型，只验证「候选池 → 重排 → top-k」这段编排。

这段最容易出的错不是模型效果，而是**编排**：
- 候选池等于 top_k（重排没得选，收益被系统性低估）；
- 重排后 rank / origin_rank 记错（报告里"正确块从第 7 位提到第 2 位"就说不清了）；
- 关闭重排时走了不同的分支（对比实验因此不干净）。
"""
from __future__ import annotations

import numpy as np

from rag.config import Config
from rag.retrieve.search import HybridSearcher

CFG = Config(
    {
        "chunking": {"active": "fixed512", "strategies": {"fixed512": {"size": 512, "overlap": 0}}},
        "search": {
            "default_mode": "hybrid_rrf",
            "candidates": 50,
            "top_k": 3,
            "rrf_k": 60,
            "weights": {"dense": 1.0, "sparse": 1.0},
        },
        "rerank": {"enabled": False, "top_n": 6},
    }
)


class _Emb:
    name = "stub"
    dim = 4
    is_fake = True

    def encode(self, texts, is_query=False):
        return np.zeros((len(texts), 4), dtype=np.float32)


class _Store:
    """按给定顺序返回候选，模拟已排序的一路召回。"""

    def __init__(self, ids):
        self.ids = ids

    def _rows(self, ids, source):
        return [
            {"chunk_id": cid, "score": 1.0 - i * 0.01, "rank": i, "source": source,
             "text": f"text-{cid}", "doc_id": "d", "parent_id": cid, "eval_block_id": f"blk-{cid}"}
            for i, cid in enumerate(ids)
        ]

    def search_dense(self, strategy, vec, top_k, expr=None, ef=128):
        return self._rows(self.ids, "dense")

    def search_sparse(self, strategy, qsparse, top_k, expr=None):
        return self._rows([], "sparse")


class _ReverseReranker:
    """把顺序完全颠倒，用来验证编排确实按重排结果输出。"""

    name = "reverse"
    is_fake = False

    def rerank(self, query, docs, top_n=None):
        idx = list(range(len(docs)))[::-1]
        return [(i, float(i)) for i in (idx[:top_n] if top_n else idx)]


def _searcher(store, reranker=None):
    return HybridSearcher(CFG, store, _Emb(), bm25={}, reranker=reranker)


def test_rerank_off_preserves_fused_order():
    s = _searcher(_Store(["a", "b", "c", "d"]))
    r = s.search("q", mode="dense", rerank=False)
    assert [h.chunk_id for h in r.hits] == ["a", "b", "c"]
    assert r.rerank is False
    assert [h.origin_rank for h in r.hits] == [0, 1, 2]
    assert [h.score_pre for h in r.hits] == [h.score for h in r.hits]


def test_rerank_uses_pool_larger_than_top_k():
    """top_k=3 但重排池为 6：先按池子取候选，重排后再截断。"""
    ids = list("abcdefgh")
    s = _searcher(_Store(ids), reranker=_ReverseReranker())
    r = s.search("q", mode="dense", top_k=3, rerank=True, rerank_top_n=6)
    assert r.n_candidates["pool"] == 6, "池子应为 rerank_top_n，而不是 top_k"
    assert r.n_candidates["reranked"] == 6
    assert [h.chunk_id for h in r.hits] == ["f", "e", "d"]
    # origin_rank 记录的是重排前的位置，报告里据此说"从第 6 位提到第 1 位"
    assert [h.origin_rank for h in r.hits] == [5, 4, 3]
    assert [h.rank for h in r.hits] == [0, 1, 2]


def test_rerank_switching_on_does_not_lose_hits():
    s = _searcher(_Store(list("abcdef")), reranker=_ReverseReranker())
    off = s.search("q", mode="dense", top_k=3, rerank=False)
    on = s.search("q", mode="dense", top_k=3, rerank=True, rerank_top_n=6)
    assert len(off.hits) == len(on.hits) == 3
    assert on.timings.get("rerank") is not None
    assert "rerank" not in off.timings


def test_rerank_pool_never_smaller_than_top_k():
    s = _searcher(_Store(list("abcdef")), reranker=_ReverseReranker())
    r = s.search("q", mode="dense", top_k=5, rerank=True, rerank_top_n=2)
    assert r.n_candidates["pool"] >= 5, "配置写小了也不能少于 top_k，否则候选凑不满"
    assert len(r.hits) == 5


def test_empty_candidates_skip_rerank():
    s = _searcher(_Store([]), reranker=_ReverseReranker())
    r = s.search("q", mode="dense", rerank=True)
    assert r.hits == []
