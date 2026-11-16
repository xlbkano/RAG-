#!/usr/bin/env python
"""把简历里「X/30」「少 Y 条」这类数字算出来，并且算得能站得住脚。

为什么需要单独一个脚本：简历上的计数类指标（命中条数、路由正确条数）看起来最简单，
实际上最容易算错、也最容易被面试官一句话问穿。这里固定三件事：

1. **配对比较**。两个方法必须在**同一批题**上跑，用 2x2 表呈现，而不是各报一个数。
   常见错误是写「混合 X 条，纯向量少 Y 条」，读者会默认 X - Y 就是纯向量的命中数。
   实际可能出现「纯向量命中、混合没命中」的反向案例，此时 Y 是净差、不是全部差异。
2. **显著性**。n=30 时，两个方法差 3 条以内基本是噪声。McNemar 精确检验只看
   不一致的那几对，比直接比准确率敏感得多，也能防止把噪声写成「提升」。
3. **可复现**。输出里带 suite 路径与 config_hash，和 results/ 下的 run 对得上。

两个子命令：

  retrieval  检索类：跑 dense 与 hybrid，输出 top-k 命中条数 + 2x2 + McNemar
  routing    路由类：读一份人工标注 CSV，输出正确条数 + 混淆矩阵 + McNemar

用法：
  python scripts/measure_resume_numbers.py retrieval --suite evalsets/quick_v1.jsonl --k 3
  python scripts/measure_resume_numbers.py routing --csv routing_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------- 统计工具
def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双侧）。

    b / c 是两张「不一致」的格子：A 对 B 错、A 错 B 对。一致的那两格不提供信息，
    所以只看 b + c。n 很小时用二项精确分布，不套卡方近似——30 条样本上卡方近似
    误差很大，容易把 p=0.07 算成 p=0.03。
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间。比正态近似稳，n=30 且比例接近 1 时不会算出 >1 的上界。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({k / n:.1%}, 95%CI {lo:.0%}–{hi:.0%})" if n else "0/0"


# ---------------------------------------------------------------- retrieval
def cmd_retrieval(a) -> int:
    from rag.config import Config
    from rag.eval import metrics as M
    from rag.eval.dataset import load_suite
    from rag.service import Service

    cfg = Config.load(a.config)
    items = [i for i in load_suite(a.suite) if i.answerable]
    if a.limit:
        items = items[: a.limit]
    if not items:
        print("评测集里没有有答案的题，检查 gold_blocks 是否为空")
        return 1

    svc = Service.create(cfg)
    modes = ("dense", "hybrid_rrf")
    hit: dict[str, dict[str, bool]] = {m: {} for m in modes}
    rows = []
    for it in items:
        gold = set(it.gold_blocks)
        rec = {"qid": it.qid, "query": it.query}
        for m in modes:
            r = svc.search(it.query, strategy=a.strategy, mode=m, top_k=a.k)
            blocks = M.ranked_blocks(h.eval_block_id for h in r.hits)
            ok = bool(set(blocks[: a.k]) & gold)
            hit[m][it.qid] = ok
            rec[m] = ok
            rec[f"{m}_top1"] = blocks[0] if blocks else ""
        rows.append(rec)

    n = len(items)
    h, d = hit["hybrid_rrf"], hit["dense"]
    both = sum(1 for q in h if h[q] and d[q])
    only_h = sum(1 for q in h if h[q] and not d[q])
    only_d = sum(1 for q in h if not h[q] and d[q])
    neither = sum(1 for q in h if not h[q] and not d[q])
    p = mcnemar_exact(only_h, only_d)

    print(f"### 检索计数（strategy={a.strategy}, top-{a.k}, config_hash={cfg.config_hash}）\n")
    print(f"样本：{n} 条有答案题，来自 {Path(a.suite).name}\n")
    print("| 方法 | 命中 |")
    print("| --- | --- |")
    print(f"| 混合（dense+BM25, RRF） | {fmt_pct(both + only_h, n)} |")
    print(f"| 纯向量（dense） | {fmt_pct(both + only_d, n)} |")
    print(f"\n2x2 配对表（同一批题）：\n")
    print("| | 纯向量命中 | 纯向量未命中 |")
    print("| --- | --- | --- |")
    print(f"| **混合命中** | {both} | {only_h} |")
    print(f"| **混合未命中** | {only_d} | {neither} |")
    print(f"\n- 混合比纯向量多命中：**{only_h} 条**（这就是简历里的「少 Y 条」= {only_h}）")
    print(f"- 纯向量反而命中、混合没命中：{only_d} 条")
    print(f"- 两者都没命中：{neither} 条")
    print(f"- 净差：{only_h - only_d:+d} 条；McNemar 精确检验 p = {p:.3f}", end="")
    print("（< 0.05 可称显著）" if p < 0.05 else "（≥ 0.05，n 太小，不宜写成「提升」）")

    if only_h or only_d:
        print("\n配对差异明细（人工复核这些题，确认不是标注错误）：")
        for r in rows:
            if r["hybrid_rrf"] != r["dense"]:
                flag = "混合独有" if r["hybrid_rrf"] else "纯向量独有"
                print(f"  [{flag}] {r['qid']}  {r['query'][:46]}")

    if a.out:
        out_p = Path(a.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)  # 允许 .tmp/retrieval.json 这类未建目录
        out_p.write_text(
            json.dumps(
                {
                    "suite": str(a.suite),
                    "strategy": a.strategy,
                    "k": a.k,
                    "config_hash": cfg.config_hash,
                    "n": n,
                    "cells": {"both": both, "only_hybrid": only_h, "only_dense": only_d, "neither": neither},
                    "mcnemar_p": round(p, 4),
                    "per_query": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n明细已写入 {a.out}")
    return 0


# ---------------------------------------------------------------- routing
def cmd_routing(a) -> int:
    """读标注 CSV。列：qid, gold, pred, [baseline]

    gold / pred / baseline 都是意图标签字符串。baseline 列是可选的单轮 prompt 分类结果，
    有它才能做「路由 vs 直接分类」的配对比较——否则只能报单边准确率，对比不成立。
    """
    path = Path(a.csv)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        print("CSV 为空")
        return 1
    need = {"qid", "gold", "pred"}
    if not need.issubset(rows[0]):
        print(f"CSV 缺少列：{need - set(rows[0])}；实际列={list(rows[0])}")
        return 1
    has_base = "baseline" in rows[0]

    n = len(rows)
    correct = sum(1 for r in rows if r["pred"].strip() == r["gold"].strip())
    print(f"### 路由计数（{path.name}）\n")
    print(f"路由正确：**{fmt_pct(correct, n)}**")

    # 混淆矩阵：行=真实意图，列=预测意图
    labels = sorted({r["gold"].strip() for r in rows} | {r["pred"].strip() for r in rows})
    print("\n混淆矩阵（行=真实，列=预测）：\n")
    print("| 真实\\预测 | " + " | ".join(labels) + " | 召回 |")
    print("| --- " * (len(labels) + 2) + "|")
    for g in labels:
        sub = [r for r in rows if r["gold"].strip() == g]
        cells = [sum(1 for r in sub if r["pred"].strip() == p) for p in labels]
        rec = cells[labels.index(g)] / len(sub) if sub else 0
        print(f"| **{g}** | " + " | ".join(str(c) for c in cells) + f" | {rec:.0%} |")

    # 转人工单独看：这是兜底节点，错在这里代价最高（该转没转 = 硬答）
    to_human = [r for r in rows if r["gold"].strip() in (a.human_label,)]
    if to_human:
        ok = sum(1 for r in to_human if r["pred"].strip() == a.human_label)
        missed = [r["qid"] for r in to_human if r["pred"].strip() != a.human_label]
        print(f"\n应转人工：{fmt_pct(ok, len(to_human))}")
        if missed:
            print(f"  漏转（会硬答）：{', '.join(missed)}")

    if has_base:
        b = [r for r in rows if r["baseline"].strip()]
        only_route = sum(1 for r in b if r["pred"].strip() == r["gold"].strip() and r["baseline"].strip() != r["gold"].strip())
        only_base = sum(1 for r in b if r["pred"].strip() != r["gold"].strip() and r["baseline"].strip() == r["gold"].strip())
        base_ok = sum(1 for r in b if r["baseline"].strip() == r["gold"].strip())
        p = mcnemar_exact(only_route, only_base)
        print("\n### 与单轮 prompt 基线对比（配对）\n")
        print("| 方法 | 正确 |")
        print("| --- | --- |")
        print(f"| LangGraph 路由 | {fmt_pct(correct, n)} |")
        print(f"| 单轮 prompt 直接分类 | {fmt_pct(base_ok, len(b))} |")
        print(f"\n- 只有路由对：{only_route} 条")
        print(f"- 只有基线对：{only_base} 条")
        print(f"- McNemar 精确检验 p = {p:.3f}", end="")
        print("（显著）" if p < 0.05 else "（不显著：n 小，别写成「提升 X 条」当结论）")
    else:
        print("\n（CSV 没有 baseline 列，无法做配对对比。补一列单轮 prompt 的分类结果才能支撑「对比」这个词）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("retrieval", help="检索类计数（命中条数 + 2x2 + McNemar）")
    p1.add_argument("--suite", required=True)
    p1.add_argument("--strategy", default="fixed512")
    p1.add_argument("--k", type=int, default=3)
    p1.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    p1.add_argument("--config", default="configs/default.yaml")
    p1.add_argument("--out", default=None, help="明细 JSON 输出路径")
    p1.set_defaults(func=cmd_retrieval)

    p2 = sub.add_parser("routing", help="路由类计数（正确条数 + 混淆矩阵 + McNemar）")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--human-label", default="转人工", help="兜底节点的标签名")
    p2.set_defaults(func=cmd_routing)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
