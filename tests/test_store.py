"""存储层集成测试：用 Milvus Lite（本地文件）验证稠密 + 稀疏双路写入与检索。

这条测试存在的意义：稀疏向量的「写入-加载-检索」链路在不同 Milvus 版本上
行为不一致（Lite 与 standalone 的索引类型不同）。用单测钉住它，避免换环境后
召回率莫名掉到 0 才发现。
"""
from __future__ import annotations

import time

import numpy as np

from rag.index.store import MilvusStore
from rag.types import Chunk


def _chunk(cid: str, text: str, parent: str = "", block: str = "") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d1",
        strategy="fixed512",
        text=text,
        start=0,
        end=len(text),
        token_len=10,
        eval_block_id=block or f"d1:b0",
        parent_id=parent or cid,
        meta={"source": "test.md", "title": "T", "section": "S"},
    )


def _wait_visible(store: MilvusStore, strategy: str, expect: int, timeout: float = 5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if store.count(strategy) >= expect:
            return
        time.sleep(0.1)


def test_dense_and_sparse_roundtrip(tmp_path):
    store = MilvusStore(str(tmp_path / "t.db"), dim=4, prefix="t")
    store.ensure("fixed512", drop=True)

    chunks = [_chunk("c1", "向量检索服务"), _chunk("c2", "文本分类模型")]
    dense = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32)
    sparse = [{11: 2.0, 12: 1.0}, {21: 3.0}]

    assert store.insert("fixed512", chunks, dense, sparse) == 2
    _wait_visible(store, "fixed512", 2)
    assert store.count("fixed512") == 2

    d = store.search_dense("fixed512", [1.0, 0, 0, 0], top_k=2)
    assert d and d[0]["chunk_id"] == "c1"
    assert d[0]["text"] == "向量检索服务"

    s = store.search_sparse("fixed512", {11: 1.0}, top_k=2)
    assert s and s[0]["chunk_id"] == "c1"
    # 稀疏内积 = 2.0 * 1.0
    assert abs(s[0]["score"] - 2.0) < 1e-6

    # 空稀疏查询应安全返回空，而不是抛异常
    assert store.search_sparse("fixed512", {}, top_k=2) == []


def test_scalar_filter(tmp_path):
    store = MilvusStore(str(tmp_path / "f.db"), dim=4, prefix="f")
    store.ensure("fixed512", drop=True)
    chunks = [_chunk("c1", "甲"), _chunk("c2", "乙")]
    chunks[1].meta["source"] = "other.md"
    dense = np.eye(2, 4, dtype=np.float32)
    store.insert("fixed512", chunks, dense, [{1: 1.0}, {2: 1.0}])
    _wait_visible(store, "fixed512", 2)

    hits = store.search_dense("fixed512", [1.0, 0, 0, 0], top_k=5, expr='source == "other.md"')
    assert all(h["source"] == "other.md" for h in hits)
