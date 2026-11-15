"""融合与 BM25 的单测。BM25 部分专门验证 doc/query 两侧 term_id 一致 ——
这是稀疏检索最容易错、又最难看出来的地方（错了两侧维度对不上，分数全为 0）。"""
from __future__ import annotations

import pytest

from rag.index.bm25 import BM25Encoder, term_id, tokenize
from rag.retrieve.search import dedup_by_parent, rrf_fuse, weighted_fuse


def _items(ids, source="dense"):
    return [{"chunk_id": i, "score": 1.0 - 0.1 * n, "rank": n, "source": source} for n, i in enumerate(ids)]


def test_rrf_prefers_items_ranked_high_by_both():
    dense = _items(["a", "b", "c"], "dense")
    sparse = _items(["b", "a", "d"], "sparse")
    fused = rrf_fuse([dense, sparse], k=60)
    # a: 1/61 + 1/62 ; b: 1/62 + 1/61  —— 两者并列最高，c/d 更低
    top2 = {fused[0]["chunk_id"], fused[1]["chunk_id"]}
    assert top2 == {"a", "b"}
    assert fused[-1]["chunk_id"] in {"c", "d"}


def test_rrf_weights_shift_ranking():
    dense = _items(["a", "b"], "dense")
    sparse = _items(["b", "a"], "sparse")
    strong_dense = rrf_fuse([dense, sparse], k=60, weights=[1.0, 0.0])
    assert strong_dense[0]["chunk_id"] == "a"
    strong_sparse = rrf_fuse([dense, sparse], k=60, weights=[0.0, 1.0])
    assert strong_sparse[0]["chunk_id"] == "b"


def test_rrf_ignores_raw_score_scale():
    """RRF 只看排名，分数尺度不同的两路可以直接融合 —— 这正是选它而非加权求和的原因。"""
    dense = [{"chunk_id": "a", "score": 0.99, "rank": 0, "source": "dense"},
             {"chunk_id": "b", "score": 0.98, "rank": 1, "source": "dense"}]
    sparse = [{"chunk_id": "b", "score": 42.0, "rank": 0, "source": "sparse"},
              {"chunk_id": "a", "score": 7.0, "rank": 1, "source": "sparse"}]
    fused = rrf_fuse([dense, sparse], k=60)
    assert fused[0]["score"] == pytest.approx(fused[1]["score"])


def test_weighted_fuse_normalizes():
    dense = _items(["a", "b"], "dense")
    sparse = _items([], "sparse")
    fused = weighted_fuse([dense, sparse], [1.0, 1.0])
    assert fused[0]["chunk_id"] == "a"
    assert fused[0]["score"] <= 1.0


def test_dedup_by_parent_keeps_best_and_counts_folded():
    items = [
        {"chunk_id": "c1", "parent_id": "p1", "score": 0.9},
        {"chunk_id": "c2", "parent_id": "p1", "score": 0.8},
        {"chunk_id": "c3", "parent_id": "p2", "score": 0.7},
    ]
    out, folded = dedup_by_parent(items, top_n=5)
    assert [x["chunk_id"] for x in out] == ["c1", "c3"]
    assert folded == 1


def test_bm25_term_ids_consistent_between_doc_and_query():
    enc = BM25Encoder().fit([tokenize("向量检索服务"), tokenize("文本检索服务")])
    doc = enc.doc_vector(tokenize("向量检索服务"))
    q = enc.query_vector("向量检索")
    assert doc and q
    assert set(q).issubset(set(doc)), "query 侧维度必须能在 doc 侧找到，否则分数恒为 0"


def test_bm25_idf_always_positive():
    enc = BM25Encoder().fit([["a"], ["a"], ["a"]])
    assert enc.idf("a") > 0
    assert enc.idf("never-seen") == 0.0


def test_bm25_rare_term_outweighs_common_term():
    docs = [["common", "rare"]] + [["common"]] * 20
    enc = BM25Encoder().fit(docs)
    assert enc.idf("rare") > enc.idf("common")


def test_term_id_stable():
    assert term_id("向量") == term_id("向量")
    assert 0 <= term_id("任意词") < 2**31
