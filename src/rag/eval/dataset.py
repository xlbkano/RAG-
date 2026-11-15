"""评测集加载与校验。

JSONL 每行一条：
    qid          唯一 id
    query        查询原文
    type         single_hop | paraphrase | multihop | no_answer
    difficulty   easy | medium | hard
    gold_blocks  回答该问题所必需的评测块 id（可多个）
    partial_blocks 部分相关块（nDCG 里记 rel=1；Recall 不计）
    gold_docs    可选的文档来源，便于人工复核
    evidence     标注时在原文里看到的证据片段（不进检索，只用于复核）

有一类标注错误比"标错块"更隐蔽：**query 直接抄了原文**。这种题 token 层面完全重合，
BM25 一定排第一，跑出来的 Recall 会明显虚高，而且看不出是标注问题。所以加载后要跑
4-gram 泄漏检测：query 的字符 4-gram 有多少比例原样出现在 gold 块里，超过 0.4 就报警。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_TYPES = ("single_hop", "paraphrase", "multihop", "no_answer")
LEAK_THRESHOLD = 0.4


@dataclass(slots=True)
class EvalItem:
    qid: str
    query: str
    type: str = "single_hop"
    difficulty: str = "medium"
    gold_blocks: list[str] = field(default_factory=list)
    partial_blocks: list[str] = field(default_factory=list)
    gold_docs: list[str] = field(default_factory=list)
    evidence: str = ""
    note: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold_blocks)

    def gains(self) -> dict[str, float]:
        """nDCG 用的分级相关度：gold=2，partial=1。"""
        g = {b: 2.0 for b in self.gold_blocks}
        for b in self.partial_blocks:
            g.setdefault(b, 1.0)
        return g


def load_suite(path: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno} JSON 解析失败: {e}") from e
        it = EvalItem(
            qid=str(raw["qid"]),
            query=str(raw["query"]).strip(),
            type=str(raw.get("type", "single_hop")),
            difficulty=str(raw.get("difficulty", "medium")),
            gold_blocks=list(raw.get("gold_blocks") or []),
            partial_blocks=list(raw.get("partial_blocks") or []),
            gold_docs=list(raw.get("gold_docs") or []),
            evidence=str(raw.get("evidence", "")),
            note=str(raw.get("note", "")),
        )
        if it.qid in seen:
            raise ValueError(f"{path}:{lineno} qid 重复: {it.qid}")
        seen.add(it.qid)
        if not it.query:
            raise ValueError(f"{path}:{lineno} query 为空")
        if it.type not in VALID_TYPES:
            raise ValueError(f"{path}:{lineno} 未知 type: {it.type}（可选 {VALID_TYPES}）")
        if it.type == "no_answer" and it.gold_blocks:
            raise ValueError(f"{path}:{lineno} no_answer 题不应有 gold_blocks")
        if it.type != "no_answer" and not it.gold_blocks:
            raise ValueError(f"{path}:{lineno} 有答案题缺少 gold_blocks")
        items.append(it)
    return items


def suite_summary(items: list[EvalItem]) -> dict:
    by_type: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    for it in items:
        by_type[it.type] = by_type.get(it.type, 0) + 1
        by_diff[it.difficulty] = by_diff.get(it.difficulty, 0) + 1
    return {
        "n": len(items),
        "n_answerable": sum(1 for it in items if it.answerable),
        "by_type": by_type,
        "by_difficulty": by_diff,
        "multi_gold": sum(1 for it in items if len(it.gold_blocks) > 1),
    }


def _grams(text: str, n: int = 4) -> set[str]:
    t = "".join(text.split())
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def leakage(query: str, block_text: str, n: int = 4) -> float:
    """query 的 n-gram 有多大比例原样出现在块里。0 表示完全改写，1 表示照抄。"""
    q = _grams(query, n)
    if not q:
        return 0.0
    return len(q & _grams(block_text, n)) / len(q)
