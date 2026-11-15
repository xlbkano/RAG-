"""评测执行器：按配置矩阵跑一遍评测集，产出可复现的 run 目录。

产物（PRD §19 的「数字 ↔ run_id ↔ config_hash」链条就靠这些文件）：
    results/{run_id}/config.json     完整配置 + 评测集 sha1
    results/{run_id}/metrics.json    聚合指标与延迟分位
    results/{run_id}/per_query.jsonl 每条 query 的命中序列，便于人工复核

一条纪律：embedder 是占位实现（hash）时拒绝产出正式指标。占位嵌入是哈希噪声，
排序没有语义，跑出来的 Recall 只是随机数——这种数字一旦进了报告就很难收回。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag.config import REPO_ROOT, Config
from rag.eval import metrics as M
from rag.eval.dataset import EvalItem

KS = (1, 3, 5, 10)
METRIC_NAMES = ("recall", "hit", "mrr", "ndcg")


@dataclass(frozen=True)
class RunSpec:
    strategy: str
    mode: str
    rerank: bool = False
    rerank_top_n: int = 30
    tag: str = ""

    def name(self) -> str:
        base = f"{self.strategy}__{self.mode}" + ("__rr" if self.rerank else "")
        return f"{base}__{self.tag}" if self.tag else base


@dataclass
class RunResult:
    spec: RunSpec
    metrics: dict
    rows: list[dict] = field(default_factory=list)


def _sha1_file(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def _metric_block(blocks: list[str], item: EvalItem) -> dict[str, float]:
    gold = set(item.gold_blocks)
    gains = item.gains()
    out: dict[str, float] = {}
    for k in KS:
        out[f"recall@{k}"] = M.recall_at_k(blocks, gold, k)
        out[f"hit@{k}"] = M.hit_at_k(blocks, gold, k)
        out[f"mrr@{k}"] = M.mrr_at_k(blocks, gold, k)
        out[f"ndcg@{k}"] = M.ndcg_at_k(blocks, gains, k)
    return out


def run_spec(
    svc,
    items: list[EvalItem],
    spec: RunSpec,
    top_k: int = 10,
    candidates: int = 50,
    warmup: int = 3,
    rerank_top_n: int | None = None,
) -> RunResult:
    """跑一个配置。top_k 取 10 以便同时算 @1/@3/@5/@10，报告里只取需要的列。"""
    kw = dict(strategy=spec.strategy, mode=spec.mode, top_k=top_k, candidates=candidates,
              rerank=spec.rerank, rerank_top_n=rerank_top_n or spec.rerank_top_n)
    for it in items[:warmup]:  # 预热，避免冷启动混进延迟分位
        svc.search(it.query, **kw)

    rows: list[dict] = []
    lat: list[float] = []
    stage_sum: dict[str, float] = {}

    for it in items:
        t0 = time.perf_counter()
        r = svc.search(it.query, **kw)
        elapsed = (time.perf_counter() - t0) * 1000
        lat.append(elapsed)
        for k, v in r.timings.items():
            if k != "total":
                stage_sum[k] = stage_sum.get(k, 0.0) + v

        blocks = M.ranked_blocks(h.eval_block_id for h in r.hits)
        row: dict = {
            "qid": it.qid,
            "type": it.type,
            "difficulty": it.difficulty,
            "n_blocks": len(blocks),
            "top1_score": round(r.hits[0].score, 4) if r.hits else None,
            "top1_origin_rank": r.hits[0].origin_rank if r.hits else None,
            "latency_ms": round(elapsed, 2),
        }
        if it.answerable:
            gold = set(it.gold_blocks)
            row.update(_metric_block(blocks, it))
            row["first_gold_rank"] = next((i for i, b in enumerate(blocks, 1) if b in gold), 0)
        rows.append(row)

    ans = [r for r in rows if "recall@5" in r]
    agg: dict = {"n_queries": len(items), "n_answerable": len(ans), "n_no_answer": len(items) - len(ans)}
    for name in METRIC_NAMES:
        for k in KS:
            key = f"{name}@{k}"
            agg[key] = M.mean([r[key] for r in ans])
    agg["avg_first_gold_rank"] = M.mean([r["first_gold_rank"] or (top_k + 1) for r in ans])
    agg["missed_all"] = sum(1 for r in ans if r["first_gold_rank"] == 0) / len(ans) if ans else 0.0

    by_type: dict[str, dict] = {}
    for t in sorted({r["type"] for r in ans}):
        sub = [r for r in ans if r["type"] == t]
        by_type[t] = {
            "n": len(sub),
            "recall@5": M.mean([r["recall@5"] for r in sub]),
            "mrr@10": M.mean([r["mrr@10"] for r in sub]),
        }

    # 无答案题：比较 top1 分数与有答案题的差距。这是将来做拒答阈值的依据，
    # 现在只记录不下结论——15 条无答案 query 定不出阈值。
    ans_scores = [r["top1_score"] for r in ans if r["top1_score"] is not None]
    noans_scores = [r["top1_score"] for r in rows if "recall@5" not in r and r["top1_score"] is not None]
    agg["score_top1_answerable"] = M.mean(ans_scores) if ans_scores else None
    agg["score_top1_no_answer"] = M.mean(noans_scores) if noans_scores else None

    timing = {
        "n": len(lat),
        "p50_ms": round(M.percentile(lat, 0.50), 2),
        "p95_ms": round(M.percentile(lat, 0.95), 2),
        "mean_ms": round(M.mean(lat), 2),
        "stage_mean_ms": {k: round(v / max(1, len(lat)), 2) for k, v in stage_sum.items()},
    }
    return RunResult(spec=spec, metrics={"aggregate": agg, "by_type": by_type, "latency": timing}, rows=rows)


def save_run(
    result: RunResult,
    cfg: Config,
    dataset_path: Path,
    out_root: Path | None = None,
    extra: dict | None = None,
) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}__{result.spec.name()}"
    out = (out_root or REPO_ROOT / "results") / run_id
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "run_id": run_id,
        "spec": asdict(result.spec),
        "config_hash": cfg.config_hash,
        "dataset": {"path": str(dataset_path), "sha1": _sha1_file(dataset_path)},
        "config": cfg.as_dict(),
        "created_at": ts,
        **(extra or {}),
    }
    (out / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in result.rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def build_specs(kind: str) -> list[RunSpec]:
    """预设矩阵。编号与 PRD §13 的表对应。"""
    strategies = ["fixed512", "recursive", "parentchild"]
    modes = ["dense", "sparse", "hybrid_rrf"]
    if kind == "smoke":
        return [RunSpec("fixed512", "hybrid_rrf", False)]
    if kind == "chunk":
        # E1 切分策略对比：检索配置固定为最强，只改切分
        return [RunSpec(s, "hybrid_rrf", True) for s in strategies]
    if kind == "rerank":
        # E3 单变量：固定 parentchild + hybrid_rrf，只看 rerank 开关
        return [RunSpec("parentchild", "hybrid_rrf", rr) for rr in (False, True)]
    if kind == "quick":
        # 精简矩阵：保留三组关键对比，砍掉 sparse+rerank（CPU 上重排是瓶颈，
        # 而稀疏单路的结论在 sparse 免重排那一行已经能读到）。
        #   E1 切分：hybrid + rerank 关/开
        #   E2 检索方式：dense / sparse / hybrid（全部免重排）
        #   E3 重排增益：dense 与 hybrid 各自的开关对比
        out: list[RunSpec] = []
        for s in strategies:
            out += [
                RunSpec(s, "dense", False),
                RunSpec(s, "dense", True),
                RunSpec(s, "sparse", False),
                RunSpec(s, "hybrid_rrf", False),
                RunSpec(s, "hybrid_rrf", True),
            ]
        return out
    if kind == "main":
        # E2 检索方式 × E3 rerank：每策略 6 组，共 18 组
        return [RunSpec(s, m, rr) for s in strategies for m in modes for rr in (False, True)]
    raise ValueError(f"unknown spec preset: {kind}")
