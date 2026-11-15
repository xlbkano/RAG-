"""混合检索：稠密 + 稀疏双路召回 → 融合 → 去重。

两路召回并行没有意义（pymilvus 客户端是同步的），但**分阶段计时**必须做，
面板 P3 的瀑布图就靠这里的 timings（PRD §7.2）。
"""
from __future__ import annotations

import time
from typing import Any

from rag.config import Config
from rag.embed import Embedder
from rag.index.bm25 import BM25Encoder
from rag.index.store import MilvusStore
from rag.types import Hit, SearchResult

MODES = ("dense", "sparse", "hybrid_rrf", "hybrid_weighted")


def rrf_fuse(
    lists: list[list[dict]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    """Reciprocal Rank Fusion：score = Σ w_i / (k + rank_i)。

    k=60 是原论文的经验值：k 越大，排名差异被抹平得越多。本项目另测了
    k ∈ {10, 60, 100}，见 PRD §9 敏感性实验。
    """
    weights = weights or [1.0] * len(lists)
    agg: dict[str, dict] = {}
    for w, ranked in zip(weights, lists):
        for item in ranked:
            cid = item["chunk_id"]
            slot = agg.setdefault(cid, {**item, "score": 0.0, "sources": []})
            slot["score"] += w / (k + item["rank"] + 1)
            slot["sources"].append(item["source"])
    return sorted(agg.values(), key=lambda d: d["score"], reverse=True)


def weighted_fuse(lists: list[list[dict]], weights: list[float]) -> list[dict]:
    """min-max 归一后加权求和。缺点：对单路分数分布敏感，作为 RRF 的对照组存在。"""
    agg: dict[str, dict] = {}
    for w, ranked in zip(weights, lists):
        if not ranked:
            continue
        scores = [r["score"] for r in ranked]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for item in ranked:
            cid = item["chunk_id"]
            slot = agg.setdefault(cid, {**item, "score": 0.0, "sources": []})
            slot["score"] += w * (item["score"] - lo) / span
            slot["sources"].append(item["source"])
    return sorted(agg.values(), key=lambda d: d["score"], reverse=True)


def dedup_by_parent(items: list[dict], top_n: int) -> tuple[list[dict], int]:
    """同一父块的多个子块只保留最高分（PRD §9）。返回 (结果, 被折叠数)。"""
    seen: set[str] = set()
    out: list[dict] = []
    folded = 0
    for it in items:
        pid = it.get("parent_id") or it["chunk_id"]
        if pid in seen:
            folded += 1
            continue
        seen.add(pid)
        out.append(it)
        if len(out) >= top_n:
            break
    return out, folded


class HybridSearcher:
    def __init__(
        self,
        cfg: Config,
        store: MilvusStore,
        embedder: Embedder,
        bm25: dict[str, BM25Encoder] | None = None,
        reranker: Any = None,
    ):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25 or {}
        self.reranker = reranker

    def get_reranker(self) -> Any:
        """惰性加载：rerank 关闭时不该为一个 278M 的模型付 1.5s 启动开销。"""
        if self.reranker is None:
            from rag.rerank import build_reranker

            self.reranker = build_reranker(self.cfg)
        return self.reranker

    def _bm25(self, strategy: str) -> BM25Encoder:
        """BM25 统计量按策略隔离（df/avgdl 依赖该策略的 chunk 集合）。"""
        if strategy not in self.bm25:
            base = self.cfg.path_of("bm25.stats_path")
            path = base.parent / f"bm25_{strategy}.json"
            if not path.exists():
                raise FileNotFoundError(f"缺少 BM25 统计文件: {path}（先跑 ingest）")
            self.bm25[strategy] = BM25Encoder.load(path)
        return self.bm25[strategy]

    def search(
        self,
        query: str,
        strategy: str | None = None,
        mode: str | None = None,
        top_k: int | None = None,
        candidates: int | None = None,
        expr: str | None = None,
        dedup_parent: bool | None = None,
        rerank: bool | None = None,
        rerank_top_n: int | None = None,
    ) -> SearchResult:
        strategy = strategy or self.cfg.get("chunking.active")
        mode = mode or self.cfg.get("search.default_mode")
        top_k = int(top_k or self.cfg.get("search.top_k"))
        candidates = int(candidates or self.cfg.get("search.candidates"))
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode} (choose from {MODES})")
        if rerank is None:
            rerank = bool(self.cfg.get("rerank.enabled", False))
        # 重排池必须 ≥ top_k：池子只有 5 条时 cross-encoder 只能重排 5 条，
        # 等于让它在自己也没得选的范围里挑，收益会被系统性低估。
        rerank_top_n = max(int(rerank_top_n or self.cfg.get("rerank.top_n", 30)), top_k)

        timings: dict[str, float] = {}
        n_cand: dict[str, int] = {}
        t0 = time.perf_counter()

        dense_items: list[dict] = []
        sparse_items: list[dict] = []

        if mode in ("dense", "hybrid_rrf", "hybrid_weighted"):
            t = time.perf_counter()
            qvec = self.embedder.encode([query], is_query=True)[0]
            timings["embed"] = round((time.perf_counter() - t) * 1000, 2)
            t = time.perf_counter()
            dense_items = self.store.search_dense(
                strategy, qvec, candidates, expr, ef=int(self.cfg.get("search.index.ef_search", 128))
            )
            timings["dense"] = round((time.perf_counter() - t) * 1000, 2)
            n_cand["dense"] = len(dense_items)

        if mode in ("sparse", "hybrid_rrf", "hybrid_weighted"):
            t = time.perf_counter()
            qsparse = self._bm25(strategy).query_vector(query)
            timings["sparse_encode"] = round((time.perf_counter() - t) * 1000, 2)
            t = time.perf_counter()
            sparse_items = self.store.search_sparse(strategy, qsparse, candidates, expr)
            timings["sparse"] = round((time.perf_counter() - t) * 1000, 2)
            n_cand["sparse"] = len(sparse_items)

        t = time.perf_counter()
        if mode == "dense":
            fused = sorted(dense_items, key=lambda d: d["score"], reverse=True)
        elif mode == "sparse":
            fused = sorted(sparse_items, key=lambda d: d["score"], reverse=True)
        elif mode == "hybrid_rrf":
            w = self.cfg.get("search.weights")
            fused = rrf_fuse(
                [dense_items, sparse_items],
                k=int(self.cfg.get("search.rrf_k", 60)),
                weights=[w["dense"], w["sparse"]],
            )
        else:  # hybrid_weighted
            w = self.cfg.get("search.weights")
            fused = weighted_fuse([dense_items, sparse_items], [w["dense"], w["sparse"]])

        # 去重与截断：重排开启时保留更大的候选池，否则重排无从发挥
        folded = 0
        if dedup_parent is None:
            dedup_parent = strategy == "parentchild"
        pool = rerank_top_n if rerank else top_k
        if dedup_parent:
            fused, folded = dedup_by_parent(fused, pool)
        else:
            fused = fused[:pool]

        timings["fuse"] = round((time.perf_counter() - t) * 1000, 2)
        n_cand["pool"] = len(fused)
        n_cand["folded_by_parent"] = folded

        # 重排：cross-encoder 对池内每个 (query, doc) 打一次分，按新分数重排。
        # 注意这里喂给模型的是**子块**文本；父子块策略下回溯父块属于生成阶段的事
        # （PRD §20.11），混进重排会让 256-token 子块与 1024-token 父块比分数，不可比。
        if rerank and fused:
            t = time.perf_counter()
            order = self.get_reranker().rerank(
                query, [it.get("text", "") for it in fused], top_n=top_k
            )
            timings["rerank"] = round((time.perf_counter() - t) * 1000, 2)
            n_cand["reranked"] = len(fused)
            staged = [
                {**fused[i], "_score_pre": fused[i]["score"], "_rank_pre": i, "score": s}
                for i, s in order
            ]
        else:
            staged = [
                {**it, "_score_pre": it["score"], "_rank_pre": i}
                for i, it in enumerate(fused[:top_k])
            ]

        timings["total"] = round((time.perf_counter() - t0) * 1000, 2)

        hits = [
            Hit(
                chunk_id=it["chunk_id"],
                score=float(it["score"]),
                text=it.get("text", ""),
                doc_id=it.get("doc_id", ""),
                parent_id=it.get("parent_id", ""),
                eval_block_id=it.get("eval_block_id", ""),
                source="+".join(sorted(set(it.get("sources", [it.get("source", "")])))),
                rank=i,
                score_pre=float(it.get("_score_pre", it["score"])),
                origin_rank=int(it.get("_rank_pre", i)),
            )
            for i, it in enumerate(staged[:top_k])
        ]
        return SearchResult(
            hits=hits,
            mode=mode,
            strategy=strategy,
            timings=timings,
            n_candidates=n_cand,
            rerank=bool(rerank),
        )
