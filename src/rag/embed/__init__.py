"""Embedding 后端。可插拔是为了让「链路验证」和「真实效果」解耦：

- HashingEmbedder：离线、零下载、确定性。**仅用于跑通链路与单测，指标无意义**。
- STEmbedder：sentence-transformers 加载的真实语义模型（bge-small-zh / bge-m3 等）。

评测执行器会检查 embedder.is_fake，非真实模型时拒绝产出正式指标（PRD §19 纪律）。

关于查询侧指令前缀（易踩坑）：
BGE 系列在 s2p（短查询检索长段落）场景下要求**只在 query 侧**拼接指令，文档侧不拼。
bge-m3 不需要指令（query_instruction 留空即可）。
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

QUERY_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："


class Embedder(Protocol):
    name: str
    dim: int
    is_fake: bool

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class HashingEmbedder:
    """字符 n-gram 哈希向量。确定性（用 crc32 而非内置 hash，避免 PYTHONHASHSEED 影响）。"""

    is_fake = True

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hash-{dim}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        import re

        toks: list[str] = [w for w in re.findall(r"[A-Za-z0-9_]+", text.lower())]
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        toks.extend(cjk)
        toks.extend("".join(p) for p in zip(cjk, cjk[1:]))  # 中文 bigram
        return toks

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                h = zlib.crc32(tok.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class STEmbedder:
    """sentence-transformers 通用后端。

    dim 从模型自动读取；与配置不一致时由 pipeline 显式报错，
    避免"配置写 1024、模型出 512"这种只在写 Milvus 时才炸的隐性问题。
    """

    is_fake = False

    def __init__(
        self,
        model: str,
        batch_size: int = 32,
        device: str | None = "cpu",
        hf_endpoint: str | None = None,
        query_instruction: str | None = None,
    ):
        from rag.hfutil import apply_endpoint, local_snapshot_or_name

        apply_endpoint(hf_endpoint)

        from sentence_transformers import SentenceTransformer  # 重依赖，延迟导入

        self.name = model
        self.batch_size = batch_size
        self.query_instruction = query_instruction or ""
        self._model = SentenceTransformer(local_snapshot_or_name(model), device=device)
        # ST 6.x 把 get_sentence_embedding_dimension 改名了，两个都兼容
        get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self.dim = int(get_dim())

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
