"""建库流水线：加载 → 切分 → BM25 统计 → 双向量编码 → 写入 Milvus。

每个策略的 BM25 统计量是**独立**的（df/avgdl 依赖该策略的 chunk 集合），
所以统计文件也按策略分开存 —— 混用会让 IDF 错位，分数不可比。
"""
from __future__ import annotations

from pathlib import Path

from rag.ingest.chunker import SUPPORTED, chunk_document
from rag.config import Config
from rag.index.bm25 import BM25Encoder, tokenize
from rag.index.store import MilvusStore
from rag.ingest.loader import load_documents
from rag.types import Chunk


def bm25_path(cfg: Config, strategy: str) -> Path:
    base = cfg.path_of("bm25.stats_path")
    return base.parent / f"bm25_{strategy}.json"


def build_embedder(cfg: Config):
    from rag.embed import Embedder, HashingEmbedder, STEmbedder

    kind = cfg.get("embedding.kind")
    dim = int(cfg.get("embedding.dim", 512))
    if kind == "hash":
        return HashingEmbedder(dim)

    emb: Embedder = STEmbedder(
        model=cfg.get("embedding.model"),
        batch_size=int(cfg.get("embedding.batch_size", 32)),
        device=cfg.get("embedding.device", "cpu"),
        hf_endpoint=cfg.get("embedding.hf_endpoint", None),
        query_instruction=cfg.get("embedding.query_instruction", None),
    )
    if emb.dim != dim:
        raise ValueError(
            f"embedding.dim 配置为 {dim}，但模型 {emb.name} 实际输出 {emb.dim} 维。"
            f"改配置后必须 --rebuild，否则 Milvus 写入维度不匹配。"
        )
    return emb


def make_store(cfg: Config) -> MilvusStore:
    uri = str(cfg.path_of("milvus.uri")) if not str(cfg.get("milvus.uri")).startswith("http") else cfg.get("milvus.uri")
    if not str(uri).startswith("http"):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    return MilvusStore(
        str(uri),
        dim=int(cfg.get("embedding.dim", 512)),
        prefix=cfg.get("milvus.collection_prefix"),
        hnsw_m=int(cfg.get("search.index.hnsw_m", 16)),
        ef_construction=int(cfg.get("search.index.ef_construction", 200)),
    )


def build_index(
    cfg: Config,
    strategies: list[str] | None = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> dict[str, int]:
    docs = load_documents(cfg.path_of("corpus.dir"), cfg.get("corpus.globs"))
    if not docs:
        raise RuntimeError("语料为空，检查 corpus.dir")

    strats = strategies or [cfg.get("chunking.active")]
    for s in strats:
        if s not in SUPPORTED:
            raise ValueError(f"未支持的切分策略: {s}")

    store = make_store(cfg)
    embedder = build_embedder(cfg)
    eval_block_tokens = int(cfg.get("chunking.eval_block_tokens", 1024))
    batch = int(cfg.get("milvus.insert_batch", 500))

    if verbose:
        print(f"[ingest] docs={len(docs)} strategies={strats} embedder={embedder.name} fake={embedder.is_fake}")

    result: dict[str, int] = {}
    for strategy in strats:
        params = cfg.get(f"chunking.strategies.{strategy}")
        raw_chunks: list[Chunk] = []
        for d in docs:
            raw_chunks.extend(chunk_document(d, strategy, params, eval_block_tokens))

        # 兜底：chunker 已丢弃无词项块，这里再按实际分词结果过滤一次。
        # 稀疏向量全零会让 Milvus 拒绝整批插入（code=6），一次失败要重跑十几分钟，
        # 所以在写库前逐个校验，宁可多一次 O(n) 的 tokenize。
        toks = [tokenize(c.text) for c in raw_chunks]
        keep = [i for i, t in enumerate(toks) if t]
        chunks = [raw_chunks[i] for i in keep]
        toks = [toks[i] for i in keep]
        dropped = len(raw_chunks) - len(chunks)

        # 1) BM25 统计（全库一次 fit）
        bm25 = BM25Encoder(k1=float(cfg.get("bm25.k1", 1.2)), b=float(cfg.get("bm25.b", 0.75)))
        bm25.fit(toks)

        # 2) 双向量编码
        dense = embedder.encode([c.text for c in chunks])
        sparse = [bm25.doc_vector(t) for t in toks]

        # 3) 写入
        store.ensure(strategy, drop=rebuild)
        n = store.insert(strategy, chunks, dense, sparse, batch=batch)
        bm25.save(bm25_path(cfg, strategy))
        result[strategy] = n
        if verbose:
            lens = sorted(c.token_len for c in chunks)
            med = lens[len(lens) // 2] if lens else 0
            print(
                f"[ingest] {strategy:<12} chunks={n:5d} dropped_empty={dropped:3d} "
                f"token_len(med/avg)={med}/{sum(lens) / max(len(lens), 1):.0f} "
                f"collections={store.name(strategy)}"
            )
    return result
