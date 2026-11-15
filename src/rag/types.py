"""核心数据结构。刻意保持薄：只做数据承载，不放业务逻辑。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    source: str          # 相对路径，面板上展示用
    title: str
    text: str


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: str
    text: str
    start: int          # 在原文中的字符偏移
    end: int
    token_len: int
    eval_block_id: str  # 与策略无关的评测块（PRD §3：所有 Recall 在此粒度算）
    parent_id: str      # 父块 id；扁平策略下等于自身
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Hit:
    chunk_id: str
    score: float
    text: str
    doc_id: str
    parent_id: str
    eval_block_id: str
    source: str = ""     # dense | sparse | fused
    rank: int = 0
    # 重排前（融合）分数与在候选池中的位次。保留这两个字段是为了能量化
    # 「rerank 把正确块从第几位提到第几位」——只报 MRR 变化看不出这件事的幅度。
    score_pre: float = 0.0
    origin_rank: int = -1

    def key(self) -> str:
        return self.chunk_id


@dataclass(slots=True)
class SearchResult:
    hits: list[Hit]
    mode: str
    strategy: str
    timings: dict[str, float]   # 阶段耗时 ms，写入 traces 表（PRD §7.2）
    n_candidates: dict[str, int] = field(default_factory=dict)
    rerank: bool = False
