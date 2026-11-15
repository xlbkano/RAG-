"""评测块 → 原文映射。

评测集的 gold 标签存的是 block_id（`docid:b12`），人工复核和泄漏检测都需要把
block_id 还原成原文。这里从语料实时算，不落盘：块网格的构造是纯函数，
每次算一遍（约 1 秒）比维护一份可能过期的缓存可靠。
"""
from __future__ import annotations

from rag.config import Config
from rag.ingest.blocks import build_block_grid
from rag.ingest.loader import load_documents


def block_texts(cfg: Config) -> dict[str, str]:
    docs = load_documents(cfg.path_of("corpus.dir"), cfg.get("corpus.globs"))
    bt = int(cfg.get("chunking.eval_block_tokens", 1024))
    out: dict[str, str] = {}
    for d in docs:
        for i, (s, e) in enumerate(build_block_grid(d.text, bt)):
            out[f"{d.doc_id}:b{i}"] = d.text[s:e]
    return out


def doc_sources(cfg: Config) -> dict[str, str]:
    docs = load_documents(cfg.path_of("corpus.dir"), cfg.get("corpus.globs"))
    return {d.doc_id: d.source for d in docs}


def resolve_source(cfg: Config, doc_id: str) -> str:
    return doc_sources(cfg).get(doc_id, f"<未知 doc_id {doc_id}>")
