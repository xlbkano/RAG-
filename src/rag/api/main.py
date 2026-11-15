"""FastAPI 接入层。

/healthz 只表示进程活着；/readyz 才表示索引与模型就绪 —— 容器编排里这两个
必须分开，否则冷启动期间的流量会把 500 打满。
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.config import Config
from rag.service import Service

_lock = threading.Lock()
_service: Service | None = None
_cfg: Config | None = None


def get_service() -> Service:
    global _service, _cfg
    with _lock:
        if _service is None:
            if _cfg is None:
                raise RuntimeError("config 未初始化")
            _service = Service.create(_cfg)
        return _service


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    strategy: str | None = Field(None, description="切分策略，默认取配置 chunking.active")
    mode: str | None = Field(None, description="dense|sparse|hybrid_rrf|hybrid_weighted")
    top_k: int = Field(5, ge=1, le=50)
    candidates: int | None = Field(None, ge=1, le=500, description="每路召回候选数")
    filters: str | None = Field(None, description='Milvus 标量过滤表达式，如 source like "milvus%"')
    rerank: bool | None = Field(None, description="是否做 cross-encoder 重排；None 取配置 rerank.enabled")
    rerank_top_n: int | None = Field(None, ge=1, le=200, description="参与精排的候选数")


class HitOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    doc_id: str
    parent_id: str
    eval_block_id: str
    source: str
    rank: int
    score_pre: float
    origin_rank: int


class SearchResponse(BaseModel):
    query: str
    strategy: str
    mode: str
    rerank: bool
    hits: list[HitOut]
    timings: dict[str, float]
    n_candidates: dict[str, int]


def create_app(cfg: Config | None = None) -> FastAPI:
    global _cfg, _service
    if cfg is not None:
        _cfg = cfg
        _service = None
    if _cfg is None:
        _cfg = Config.load()

    app = FastAPI(title="RAG 检索服务 + 评估体系", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "ts": time.time()}

    @app.get("/readyz")
    def readyz():
        try:
            st = get_service().stats()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"not ready: {e}")
        total = sum(st["collections"].values())
        if total == 0:
            raise HTTPException(status_code=503, detail="索引为空，先执行 ingest")
        return {"status": "ready", **st}

    @app.get("/v1/stats")
    def stats():
        return get_service().stats()

    @app.post("/v1/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        svc = get_service()
        t0 = time.perf_counter()
        try:
            r = svc.search(
                req.query,
                strategy=req.strategy,
                mode=req.mode,
                top_k=req.top_k,
                candidates=req.candidates,
                expr=req.filters,
                rerank=req.rerank,
                rerank_top_n=req.rerank_top_n,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
        resp = SearchResponse(
            query=req.query,
            strategy=r.strategy,
            mode=r.mode,
            rerank=r.rerank,
            hits=[HitOut(**asdict(h)) for h in r.hits],
            timings={**r.timings, "api_total": round((time.perf_counter() - t0) * 1000, 2)},
            n_candidates=r.n_candidates,
        )
        return resp

    return app
