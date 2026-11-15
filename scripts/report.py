"""把 results/*/metrics.json 汇总成对比表。

原则：表里的每个数字都来自某个 run 目录的真实文件，不做任何手工填写。
如果某一组配置没跑，表格里就是空的——空白比编造的数字有价值。

用法：
    python scripts/report.py                 # 打印全部表格
    python scripts/report.py --out docs/results-quick.md
    python scripts/report.py --suite quick_v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_runs(suite: str | None = None) -> list[dict]:
    runs = []
    for d in sorted(RESULTS.glob("*")):
        mf, cf = d / "metrics.json", d / "config.json"
        if not (mf.exists() and cf.exists()):
            continue
        meta = json.loads(cf.read_text(encoding="utf-8"))
        ds = Path(meta.get("dataset", {}).get("path", "")).name
        if suite and suite not in ds:
            continue
        pq = d / "per_query.jsonl"
        rows = []
        if pq.exists():
            rows = [json.loads(x) for x in pq.read_text(encoding="utf-8").splitlines() if x.strip()]
        runs.append(
            {
                "run_id": meta["run_id"],
                "spec": meta["spec"],
                "config_hash": meta.get("config_hash", ""),
                "dataset": ds,
                "metrics": json.loads(mf.read_text(encoding="utf-8")),
                "rows": rows,
            }
        )
    return runs


def paired_diff(
    rows_a: list[dict], rows_b: list[dict], metric: str = "recall@5", iters: int = 2000, seed: int = 13
) -> dict | None:
    """成对 bootstrap：同一批 query 上两个配置的指标差值的置信区间。

    为什么必须成对：44 条样本上单组 Recall@5 的 95% 置信区间约 ±13pt，
    两个独立区间一比，任何 10pt 以内的差异都说不清。但同一批 query 上的**差值**
    方差小得多——难题两个配置都错、简单题两个配置都对，这些对差异没有贡献的样本
    在成对检验里被消掉了。

    返回的 lo/hi 是差值（b - a）的 95% 区间。区间跨 0 就意味着"这点差距还解释不了"。
    """
    a = {r["qid"]: r[metric] for r in rows_a if r.get(metric) is not None}
    b = {r["qid"]: r[metric] for r in rows_b if r.get(metric) is not None}
    qids = sorted(set(a) & set(b))
    if not qids:
        return None
    diffs = [b[q] - a[q] for q in qids]
    n = len(diffs)
    mean_d = sum(diffs) / n

    rng = random.Random(seed)
    boots = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * iters)]
    hi = boots[int(0.975 * iters) - 1]
    return {
        "n": n,
        "mean": mean_d,
        "lo": lo,
        "hi": hi,
        "improved": sum(1 for d in diffs if d > 0),
        "worse": sum(1 for d in diffs if d < 0),
        "tie": sum(1 for d in diffs if d == 0),
        "significant": lo > 0 or hi < 0,
    }


def _fmt(v, nd=3, pct=False):
    if v is None:
        return "—"
    return f"{v * 100:.1f}" if pct else f"{v:.{nd}f}"


def _order(values, preferred):
    """按预设顺序排，未列出的按字母排在后面。"""
    vals = list(values)
    known = [v for v in preferred if v in vals]
    return known + sorted(v for v in vals if v not in preferred)


STRAT_ORDER = ["fixed512", "recursive", "parentchild"]
MODE_ORDER = ["dense", "sparse", "hybrid_rrf", "hybrid_weighted"]


def table_matrix(runs: list[dict]) -> str:
    """主线：切分策略 × 检索方式 × 是否重排。"""
    rows = {}
    for r in runs:
        s = r["spec"]
        rows[(s["strategy"], s["mode"], bool(s["rerank"]))] = r

    strategies = _order({k[0] for k in rows}, STRAT_ORDER)
    modes = _order({k[1] for k in rows}, MODE_ORDER)
    out = [
        "| 切分策略 | 检索方式 | rerank | Recall@5 | Hit@5 | MRR@10 | nDCG@10 | P50 (ms) | P95 (ms) | run_id |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for st in strategies:
        for md in modes:
            for rr in (False, True):
                r = rows.get((st, md, rr))
                if not r:
                    continue
                a, lt = r["metrics"]["aggregate"], r["metrics"]["latency"]
                out.append(
                    f"| {st} | {md} | {'开' if rr else '关'} | "
                    f"{_fmt(a['recall@5'], pct=True)}% | {_fmt(a['hit@5'], pct=True)}% | "
                    f"{_fmt(a['mrr@10'])} | {_fmt(a['ndcg@10'])} | "
                    f"{lt['p50_ms']:.1f} | {lt['p95_ms']:.1f} | `{r['run_id']}` |"
                )
    return "\n".join(out)


def table_rerank_gain(runs: list[dict]) -> str:
    """同一 (策略, 检索方式) 下，rerank 开关的差值。"""
    pairs: dict[tuple[str, str], dict[bool, dict]] = {}
    for r in runs:
        s = r["spec"]
        pairs.setdefault((s["strategy"], s["mode"]), {})[bool(s["rerank"])] = r

    out = [
        "| 切分策略 | 检索方式 | Recall@5 关→开 | Δ Recall@5 | MRR@10 关→开 | Δ MRR@10 | Δ P95 (ms) |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for (st, md), d in sorted(
        pairs.items(), key=lambda kv: (_order([kv[0][0]], STRAT_ORDER).index(kv[0][0]) if kv[0][0] in STRAT_ORDER else 99,
                                       _order([kv[0][1]], MODE_ORDER).index(kv[0][1]) if kv[0][1] in MODE_ORDER else 99)
    ):
        if True not in d or False not in d:
            continue
        off, on = d[False], d[True]
        a0, a1 = off["metrics"]["aggregate"], on["metrics"]["aggregate"]
        d5 = (a1["recall@5"] - a0["recall@5"]) * 100
        dm = a1["mrr@10"] - a0["mrr@10"]
        dp = on["metrics"]["latency"]["p95_ms"] - off["metrics"]["latency"]["p95_ms"]
        out.append(
            f"| {st} | {md} | {_fmt(a0['recall@5'], pct=True)}% → {_fmt(a1['recall@5'], pct=True)}% | "
            f"{d5:+.1f} pt | {_fmt(a0['mrr@10'])} → {_fmt(a1['mrr@10'])} | {dm:+.3f} | {dp:+.1f} |"
        )
    return "\n".join(out)


def table_by_type(runs: list[dict]) -> str:
    out = [
        "| 配置 | 题型 | n | Recall@5 | MRR@10 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in runs:
        for t, v in sorted(r["metrics"].get("by_type", {}).items()):
            out.append(
                f"| {r['spec']['strategy']}/{r['spec']['mode']}"
                f"{'/rr' if r['spec']['rerank'] else ''} | {t} | {v['n']} | "
                f"{_fmt(v['recall@5'], pct=True)}% | {_fmt(v['mrr@10'])} |"
            )
    return "\n".join(out)


def table_latency_breakdown(runs: list[dict]) -> str:
    stages = ["embed", "dense", "sparse", "fuse", "rerank"]
    out = [
        "| 配置 | P50 | P95 | " + " | ".join(stages) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in stages) + " |",
    ]
    for r in runs:
        lt = r["metrics"]["latency"]
        cells = " | ".join(_fmt(lt["stage_mean_ms"].get(s), nd=1) for s in stages)
        out.append(
            f"| {r['spec']['strategy']}/{r['spec']['mode']}"
            f"{'/rr' if r['spec']['rerank'] else ''} | {lt['p50_ms']:.1f} | {lt['p95_ms']:.1f} | {cells} |"
        )
    return "\n".join(out)


def _sig(d: dict | None) -> str:
    if not d:
        return "—"
    mark = "显著" if d["significant"] else "不显著"
    return f"[{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}] {mark}"


def table_mode_vs_dense(runs: list[dict]) -> str:
    """检索方式相对同策略 dense 基线的差值（重排关闭，隔离出"召回方式"这一个变量）。"""
    idx = {(r["spec"]["strategy"], r["spec"]["mode"], bool(r["spec"]["rerank"])): r for r in runs}
    out = [
        "| 切分策略 | 对比 | Δ Recall@5 | 95% CI (pt) | 变好/变差/不变 | n |",
        "| --- | --- | ---: | --- | --- | ---: |",
    ]
    for (st, md, rr), r in sorted(
        idx.items(), key=lambda kv: (_order([kv[0][0]], STRAT_ORDER).index(kv[0][0]) if kv[0][0] in STRAT_ORDER else 99,
                                     _order([kv[0][1]], MODE_ORDER).index(kv[0][1]) if kv[0][1] in MODE_ORDER else 99,
                                     kv[0][2])
    ):
        if rr or md == "dense":
            continue
        base = idx.get((st, "dense", False))
        if not base:
            continue
        d = paired_diff(base["rows"], r["rows"])
        if not d:
            continue
        out.append(
            f"| {st} | dense → {md} | {d['mean'] * 100:+.1f} pt | {_sig(d)} | "
            f"{d['improved']}/{d['worse']}/{d['tie']} | {d['n']} |"
        )
    return "\n".join(out)


def table_rerank_gain_ci(runs: list[dict]) -> str:
    idx = {(r["spec"]["strategy"], r["spec"]["mode"], bool(r["spec"]["rerank"])): r for r in runs}
    out = [
        "| 切分策略 | 检索方式 | Δ Recall@5 | 95% CI (pt) | 变好/变差/不变 | Δ MRR@10 | n |",
        "| --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for (st, md, rr), r in sorted(
        idx.items(), key=lambda kv: (_order([kv[0][0]], STRAT_ORDER).index(kv[0][0]) if kv[0][0] in STRAT_ORDER else 99,
                                     _order([kv[0][1]], MODE_ORDER).index(kv[0][1]) if kv[0][1] in MODE_ORDER else 99,
                                     kv[0][2])
    ):
        if not rr:
            continue
        base = idx.get((st, md, False))
        if not base:
            continue
        d = paired_diff(base["rows"], r["rows"])
        dm = paired_diff(base["rows"], r["rows"], metric="mrr@10")
        if not d:
            continue
        out.append(
            f"| {st} | {md} | {d['mean'] * 100:+.1f} pt | {_sig(d)} | "
            f"{d['improved']}/{d['worse']}/{d['tie']} | "
            f"{(dm['mean'] if dm else 0):+.3f} | {d['n']} |"
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None, help="只统计文件名含该串的评测集")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = load_runs(a.suite)
    if not runs:
        print("没有找到 results/*/metrics.json，先跑 `python tasks.py eval`")
        return 1

    datasets = sorted({r["dataset"] for r in runs})
    hashes = sorted({r["config_hash"] for r in runs})
    n_q = max((r["metrics"]["aggregate"]["n_answerable"] for r in runs), default=0)
    parts = [
        f"### 数据集：{', '.join(datasets)}　配置哈希：{', '.join(hashes)}",
        f"\n共 {len(runs)} 个 run，有答案题 {n_q} 条。\n",
        "## 1. 主矩阵：切分策略 × 检索方式 × 重排\n",
        table_matrix(runs),
        "\n## 2. 检索方式相对 dense 基线的差值（成对 bootstrap，重排关闭）\n",
        table_mode_vs_dense(runs),
        "\n## 3. 重排增益（同配置开关对比，成对 bootstrap）\n",
        table_rerank_gain_ci(runs),
        "\n## 4. 重排增益点估计\n",
        table_rerank_gain(runs),
        "\n## 5. 分题型\n",
        table_by_type(runs),
        "\n## 6. 延迟拆解（阶段均值为该配置全部 query 的平均）\n",
        table_latency_breakdown(runs),
    ]
    text = "\n".join(parts)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"written to {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
