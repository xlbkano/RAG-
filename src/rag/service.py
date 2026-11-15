"""服务装配：把 config / 索引 / 编码器 / 检索器组装成一个对象，供 CLI 与 API 共用。"""
from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import Config
from rag.embed import Embedder
from rag.index.bm25 import BM25Encoder
from rag.index.store import MilvusStore
from rag.ingest.pipeline import build_embedder, make_store
from rag.retrieve.search import HybridSearcher
from rag.types import SearchResult


@dataclass
class Service:
    cfg: Config
    store: MilvusStore
    embedder: Embedder
    searcher: HybridSearcher
    _bm25: dict[str, BM25Encoder] = field(default_factory=dict)

    @classmethod
    def create(cls, cfg: Config, load_reranker: bool | None = None) -> "Service":
        store = make_store(cfg)
        embedder = build_embedder(cfg)
        searcher = HybridSearcher(cfg, store, embedder, bm25={})
        svc = cls(cfg=cfg, store=store, embedder=embedder, searcher=searcher)
        # 默认按配置决定是否预加载重排模型；评测脚本会显式传 True，
        # 避免"第一次 rerank 请求把 1.5s 模型加载算进延迟分位"。
        if load_reranker is None:
            load_reranker = bool(cfg.get("rerank.enabled", False))
        if load_reranker:
            searcher.get_reranker()
        svc.warmup()
        return svc

    def warmup(self) -> None:
        """预热三件事，缺一个都会让首个真实请求变慢（PRD §7.3 冷启动要求）：
        1. jieba 词典首次加载约 400ms；
        2. collection 首次 load 约 200ms；
        3. Milvus 首次 search 的段懒加载约 300ms（实测，不预热会完整落在首请求上）。
        """
        from rag.index.bm25 import tokenize

        tokenize("预热")
        strategy = self.cfg.get("chunking.active")
        try:
            self.store.ensure_loaded(strategy)
            if self.store.count(strategy) > 0:
                dim = int(self.cfg.get("embedding.dim", 512))
                self.store.search_dense(strategy, [0.0] * dim, top_k=1)
        except Exception:
            pass  # 索引尚未建立时允许启动，由 /readyz 报告

    def search(self, query: str, **kw) -> SearchResult:
        return self.searcher.search(query, **kw)

    def stats(self) -> dict:
        strategies = list(self.cfg.get("chunking.strategies").keys())
        reranker = self.searcher.reranker
        return {
            "config_hash": self.cfg.config_hash,
            "embedder": self.embedder.name,
            "embedder_is_fake": self.embedder.is_fake,
            "reranker": None if reranker is None else reranker.name,
            "reranker_loaded": reranker is not None and not reranker.is_fake,
            "milvus_uri": self.store.uri,
            "collections": {s: self.store.count(s) for s in strategies},
        }
