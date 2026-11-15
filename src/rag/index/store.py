"""Milvus 存储层。

一个策略一个 collection（PRD §4 取舍说明）：避免不同切分策略共享 HNSW 图/
稀疏倒排表导致索引参数互相污染，评测才干净。

与 PRD §5.1 的一处偏离：把原先的 `meta` JSON 字段拆成 source/title/section
三个独立标量字段。原因：面板要按 source/section 做过滤与分组统计，JSON 字段
建不了标量索引、也没法做 group by。这是实现阶段对设计的修正。
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from pymilvus import DataType, MilvusClient

from rag.types import Chunk

OUTPUT_FIELDS = ["doc_id", "parent_id", "eval_block_id", "text", "source", "title", "section"]


class MilvusStore:
    def __init__(
        self,
        uri: str,
        dim: int = 1024,
        prefix: str = "rag_chunks",
        hnsw_m: int = 16,
        ef_construction: int = 200,
    ):
        self.uri = uri
        self.dim = dim
        self.prefix = prefix
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.is_lite = not str(uri).startswith(("http://", "https://"))
        self.client = MilvusClient(uri)
        self._loaded: set[str] = set()

    # ---------- schema ----------
    def name(self, strategy: str) -> str:
        return f"{self.prefix}_{strategy}"

    def _schema(self):
        s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        s.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        s.add_field("doc_id", DataType.VARCHAR, max_length=64)
        s.add_field("parent_id", DataType.VARCHAR, max_length=128)
        s.add_field("eval_block_id", DataType.VARCHAR, max_length=128)
        s.add_field("strategy", DataType.VARCHAR, max_length=32)
        s.add_field("text", DataType.VARCHAR, max_length=8192)
        s.add_field("token_len", DataType.INT32)
        s.add_field("source", DataType.VARCHAR, max_length=512)
        s.add_field("title", DataType.VARCHAR, max_length=512)
        s.add_field("section", DataType.VARCHAR, max_length=512)
        s.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        return s

    def _index_params(self):
        ip = self.client.prepare_index_params()
        if self.is_lite:
            # Milvus Lite 只支持桶式索引：稠密用 AUTOINDEX，稀疏用倒排（已验证可检索）
            ip.add_index("dense", index_type="AUTOINDEX", metric_type="IP")
        else:
            ip.add_index(
                "dense",
                index_type="HNSW",
                metric_type="IP",
                params={"M": self.hnsw_m, "efConstruction": self.ef_construction},
            )
        ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        return ip

    # ---------- 生命周期 ----------
    def ensure(self, strategy: str, drop: bool = False) -> str:
        name = self.name(strategy)
        if drop and self.client.has_collection(name):
            self.client.drop_collection(name)
        if not self.client.has_collection(name):
            self.client.create_collection(name, schema=self._schema(), index_params=self._index_params())
        self.load(strategy)
        return name

    def load(self, strategy: str) -> None:
        try:
            self.client.load_collection(self.name(strategy))
        except Exception:  # 已加载 / lite 实现差异，忽略
            pass
        self._loaded.add(strategy)

    def ensure_loaded(self, strategy: str) -> None:
        """新进程里 collection 状态是 released，检索前必须显式 load。
        幂等：同一 store 实例只加载一次，避免每次查询多一次 RPC。"""
        if strategy not in self._loaded:
            self.load(strategy)

    def drop(self, strategy: str) -> None:
        if self.client.has_collection(self.name(strategy)):
            self.client.drop_collection(self.name(strategy))

    def count(self, strategy: str) -> int:
        name = self.name(strategy)
        if not self.client.has_collection(name):
            return 0
        return int(self.client.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"])

    # ---------- 写入 ----------
    def insert(
        self,
        strategy: str,
        chunks: list[Chunk],
        dense: np.ndarray,
        sparse: list[dict[int, float]],
        batch: int = 500,
    ) -> int:
        name = self.name(strategy)
        rows: list[dict[str, Any]] = []
        for c, dv, sv in zip(chunks, dense, sparse):
            rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "parent_id": c.parent_id,
                    "eval_block_id": c.eval_block_id,
                    "strategy": c.strategy,
                    "text": c.text[:8000],
                    "token_len": c.token_len,
                    "source": str(c.meta.get("source", ""))[:500],
                    "title": str(c.meta.get("title", ""))[:500],
                    "section": str(c.meta.get("section", ""))[:500],
                    "dense": dv.tolist(),
                    "sparse": sv,
                }
            )
        for i in range(0, len(rows), batch):
            self.client.insert(name, rows[i : i + batch])
        self.load(strategy)
        return len(rows)

    # ---------- 检索 ----------
    def _search(self, strategy: str, field: str, data: Any, top_k: int, expr: str | None, ef: int):
        self.ensure_loaded(strategy)
        params: dict[str, Any] = {"metric_type": "IP"}
        if field == "dense" and not self.is_lite:
            params["params"] = {"ef": ef}
        return self.client.search(
            self.name(strategy),
            data=[data],
            anns_field=field,
            limit=top_k,
            filter=expr or "",
            output_fields=OUTPUT_FIELDS,
            search_params=params,
        )

    def search_dense(self, strategy, vector, top_k, expr=None, ef=128) -> list[dict]:
        res = self._search(strategy, "dense", list(map(float, vector)), top_k, expr, ef)
        return self._to_dicts(res, "dense")

    def search_sparse(self, strategy, sparse: dict[int, float], top_k, expr=None) -> list[dict]:
        if not sparse:
            return []
        res = self._search(strategy, "sparse", sparse, top_k, expr, 0)
        return self._to_dicts(res, "sparse")

    @staticmethod
    def _to_dicts(res: Iterable, source: str) -> list[dict]:
        """注意：pymilvus 的 Hit 是 dict 子类，但主键不一定叫 "id"。
        本项目主键字段名为 chunk_id，因此必须走 `hit.id` 属性取主键，
        直接 `hit["id"]` 会在 entity 里找不到而抛 KeyError。
        """
        out: list[dict] = []
        for hits in res:
            for rank, h in enumerate(hits):
                cid = getattr(h, "id", None)
                score = getattr(h, "distance", None)
                row: dict[str, Any] = {
                    "chunk_id": str(cid if cid is not None else ""),
                    "score": float(score if score is not None else 0.0),
                    "rank": rank,
                    "source": source,
                }
                for k in OUTPUT_FIELDS:
                    val = h.get(k, "") if hasattr(h, "get") else ""
                    row[k] = "" if val is None else val
                out.append(row)
        return out
