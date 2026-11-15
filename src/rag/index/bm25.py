"""BM25 → Milvus 稀疏向量。

为什么要自己算，而不是用 Milvus 内置 BM25 Function：
1. IDF 依赖全库统计量，自己算才能在面板上解释「为什么这个词被降权」；
2. 内置 Function 需要 analyzer 配置，中文切词行为不可控，跨版本可能变；
3. 自己算 = 可以把 doc 侧权重与 query 侧权重分开存，与 Elasticsearch 的 BM25 语义一致，便于横向对齐。

核心技巧（面试可讲）：
    doc 侧权重 = idf(t) * tf_norm(t)      ← 存在稀疏向量里
    query 侧权重 = 1.0                     ← 每个 query 词一项
    两向量内积(IP) = Σ idf * tf_norm      ← 正好是 BM25 打分（省略 query 侧 tf，短查询标准做法）
如果把 idf 也乘到 query 侧，就会变成 idf²，打分被过度放大 —— 这是常见实现错误。
"""
from __future__ import annotations

import json
import math
import re
import zlib
from pathlib import Path

import jieba

_ASCII = re.compile(r"[A-Za-z0-9_]+")
_DROP = set(" \t\r\n，。、；：？！“”‘’（）《》【】…—～·,.;:?!\"'()[]{}<>|/\\`~@#$%^&*+=_-")


def tokenize(text: str) -> list[str]:
    """中英混排切词：ASCII 词整体保留，其余交给 jieba。"""
    text = text.lower()
    toks = [w for w in _ASCII.findall(text)]
    # 去掉 ASCII 词后剩下的中文片段交给 jieba
    rest = _ASCII.sub(" ", text)
    toks.extend(w for w in jieba.lcut(rest) if w.strip() and w not in _DROP)
    return toks


def term_id(term: str) -> int:
    """稳定映射到 uint32 正区间（crc32 跨进程稳定，内置 hash 不稳定）。"""
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = 0
        self.avgdl = 0.0
        self.df: dict[str, int] = {}

    # ---------- 统计量 ----------
    def fit(self, docs_tokens: list[list[str]]) -> "BM25Encoder":
        self.n_docs = len(docs_tokens)
        total = 0
        df: dict[str, int] = {}
        for toks in docs_tokens:
            total += len(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.avgdl = (total / self.n_docs) if self.n_docs else 0.0
        self.df = df
        return self

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        if d == 0 or self.n_docs == 0:
            return 0.0
        return math.log(1.0 + (self.n_docs - d + 0.5) / (d + 0.5))

    # ---------- 向量化 ----------
    def doc_vector(self, tokens: list[str]) -> dict[int, float]:
        if not tokens:
            return {}
        dl = len(tokens)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        out: dict[int, float] = {}
        for t, c in tf.items():
            idf = self.idf(t)
            if idf <= 0:
                continue
            norm = c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out[term_id(t)] = round(idf * norm, 6)
        return out

    def query_vector(self, text: str) -> dict[int, float]:
        toks = tokenize(text)
        out: dict[int, float] = {}
        for t in toks:
            tid = term_id(t)
            if self.idf(t) > 0:
                out[tid] = 1.0
        return out

    # ---------- 持久化 ----------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"k1": self.k1, "b": self.b, "n_docs": self.n_docs, "avgdl": self.avgdl, "df": self.df},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Encoder":
        d = json.loads(path.read_text(encoding="utf-8"))
        enc = cls(k1=d["k1"], b=d["b"])
        enc.n_docs, enc.avgdl, enc.df = d["n_docs"], d["avgdl"], d["df"]
        return enc
