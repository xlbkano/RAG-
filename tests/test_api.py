"""API 集成测试：临时目录建库 → 进程内起服务 → 打接口。

不依赖外部已建好的索引，所以可以随时跑（CI 友好）。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.api.main import create_app
from rag.config import Config, REPO_ROOT
from rag.ingest.pipeline import build_index


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    data = Config.load().as_dict()
    data["milvus"]["uri"] = str(tmp / "api.db")
    data["bm25"]["stats_path"] = str(tmp / "bm25.json")
    data["corpus"]["dir"] = str(REPO_ROOT / "tests" / "fixtures" / "corpus")
    data["chunking"]["active"] = "fixed512"
    data["embedding"]["kind"] = "hash"
    cfg = Config(data)
    build_index(cfg, strategies=["fixed512"], rebuild=True, verbose=False)
    return TestClient(create_app(cfg))


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readyz_reports_index(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["collections"]["fixed512"] > 0
    assert body["embedder_is_fake"] is True  # 提醒：这不是正式指标环境


def test_search_returns_hits_with_timing(client):
    r = client.post(
        "/v1/search",
        json={"query": "HNSW 的 ef 参数控制什么", "mode": "hybrid_rrf", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "hybrid_rrf"
    assert len(body["hits"]) >= 1
    top = body["hits"][0]
    assert top["eval_block_id"].startswith(top["doc_id"])
    assert "total" in body["timings"]
    assert body["timings"]["api_total"] >= 0


def test_sparse_mode_hits_expected_doc(client):
    r = client.post(
        "/v1/search",
        json={"query": "BM25 的 k1 控制什么", "mode": "sparse", "top_k": 3},
    )
    sources = [h["source"] for h in r.json()["hits"]]
    assert any("bm25" in s for s in sources), f"BM25 检索没命中 BM25 文档: {sources}"


def test_invalid_mode_returns_500_with_detail(client):
    r = client.post("/v1/search", json={"query": "任意", "mode": "not_a_mode"})
    assert r.status_code == 500
    assert "unknown mode" in r.json()["detail"]


def test_empty_query_rejected(client):
    r = client.post("/v1/search", json={"query": "", "mode": "dense"})
    assert r.status_code == 422
