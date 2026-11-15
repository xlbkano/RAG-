"""重排（cross-encoder）后端。

与双塔召回的本质区别：cross-encoder 把 query 和 doc 拼成**一条**输入过一遍
Transformer，能建模逐词交互，精度更高；代价是无法预计算文档向量，只能对
少量候选做在线打分。所以它一定是「召回 N 个 → 精排 → 取 top-k」的结构，
N 的取值直接决定延迟与收益的平衡点（见 configs/default.yaml 的 rerank.top_n）。

CPU 现实约束（PRD §7.1: 端到端 P95 ≤ 250ms，召回已占约 40ms）：
- bge-reranker-base（278M）而非 v2-m3（568M）；
- 文档侧按字符截断（doc_max_chars），cross-encoder 对超长文本收益递减、
  注意力开销却随长度平方增长；
- 批内并行（batch_size），避免逐条过模型。

一个容易写错的地方：`CrossEncoder.predict` 对单条输入返回 0 维标量数组，
对多条返回 1 维；某些模型（num_labels=1 的回归头）会返回 (n,1) 二维。
三种形态都要能吞下去，否则会在「只召回 1 条候选」的边界上炸。
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Reranker(Protocol):
    name: str
    is_fake: bool

    def rerank(self, query: str, docs: Sequence[str], top_n: int | None = None) -> list[tuple[int, float]]:
        """返回 [(原始下标, 分数)]，按分数降序。"""
        ...


class IdentityReranker:
    """不重排，仅把融合分数原样透传。

    存在的意义不是「占位」，而是让评测矩阵里 rerank=off 这一列走**同一条代码路径**，
    避免两套分支带来的实现差异污染对比（PRD §13 消融实验的公平性要求）。
    """

    name = "identity"
    is_fake = True

    def rerank(self, query: str, docs: Sequence[str], top_n: int | None = None) -> list[tuple[int, float]]:
        return [(i, 0.0) for i in range(len(docs))]


class CrossEncoderReranker:
    def __init__(
        self,
        model: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        doc_max_chars: int = 800,
        hf_endpoint: str | None = None,
    ):
        from rag.hfutil import apply_endpoint, local_snapshot_or_name

        apply_endpoint(hf_endpoint)

        from sentence_transformers import CrossEncoder  # 重依赖，延迟导入

        self.name = model
        self.is_fake = False
        self.batch_size = batch_size
        self.doc_max_chars = doc_max_chars
        self._model = CrossEncoder(
            local_snapshot_or_name(model), device=device, max_length=max_length
        )

    def rerank(self, query: str, docs: Sequence[str], top_n: int | None = None) -> list[tuple[int, float]]:
        if not docs:
            return []
        pairs = [(query, d[: self.doc_max_chars]) for d in docs]
        raw = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw, dtype=np.float64).reshape(-1)
        order = np.argsort(-scores, kind="stable")
        if top_n is not None:
            order = order[:top_n]
        return [(int(i), float(scores[i])) for i in order]


def build_reranker(cfg) -> Reranker:
    kind = str(cfg.get("rerank.kind", "cross_encoder"))
    if kind in ("identity", "none", "off"):
        return IdentityReranker()
    if kind == "cross_encoder":
        return CrossEncoderReranker(
            model=cfg.get("rerank.model", "BAAI/bge-reranker-base"),
            device=cfg.get("rerank.device", "cpu"),
            batch_size=int(cfg.get("rerank.batch_size", 16)),
            max_length=int(cfg.get("rerank.max_length", 512)),
            doc_max_chars=int(cfg.get("rerank.doc_max_chars", 800)),
            hf_endpoint=cfg.get("rerank.hf_endpoint", None),
        )
    raise ValueError(f"unknown rerank.kind: {kind}")
