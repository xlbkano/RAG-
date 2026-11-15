"""命令行入口：python -m rag.cli <ingest|search|stats|serve>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.config import Config


def cmd_ingest(cfg: Config, args) -> int:
    from rag.ingest.pipeline import build_index

    res = build_index(cfg, strategies=args.strategies, rebuild=args.rebuild)
    print(json.dumps({"config_hash": cfg.config_hash, "indexed": res}, ensure_ascii=False))
    return 0


def cmd_search(cfg: Config, args) -> int:
    from rag.service import Service

    svc = Service.create(cfg)
    r = svc.search(args.query, strategy=args.strategy, mode=args.mode, top_k=args.top_k)
    print(f"\nquery: {args.query}")
    print(f"strategy={r.strategy} mode={r.mode} timings={r.timings} candidates={r.n_candidates}\n")
    for h in r.hits:
        head = h.text[:90].replace("\n", " ")
        print(f"  #{h.rank + 1} score={h.score:.4f} [{h.source}] {h.doc_id} block={h.eval_block_id}")
        print(f"      {head}")
    if not r.hits:
        print("  (无结果)")
    print()
    return 0


def cmd_stats(cfg: Config, args) -> int:
    from rag.service import Service

    svc = Service.create(cfg)
    print(json.dumps(svc.stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_check_suite(cfg: Config, args) -> int:
    """校验评测集：配额、gold 块是否真实存在、query 是否照抄原文。"""
    from rag.eval.corpus_index import block_texts, resolve_source
    from rag.eval.dataset import LEAK_THRESHOLD, leakage, load_suite, suite_summary

    items = load_suite(args.suite)
    summary = suite_summary(items)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    texts = block_texts(cfg)
    missing, leaks = [], []
    for it in items:
        for b in it.gold_blocks:
            if b not in texts:
                missing.append((it.qid, b))
                continue
            r = leakage(it.query, texts[b])
            if r >= LEAK_THRESHOLD:
                leaks.append((it.qid, round(r, 3), b))
    if missing:
        print(f"\n[!] gold 块不存在于当前语料/网格：{len(missing)} 处")
        for qid, b in missing[:10]:
            print(f"    {qid} -> {b}  (doc={resolve_source(cfg, b.split(':')[0])})")
    if leaks:
        print(f"\n[!] 泄漏超阈值（4-gram 重叠 ≥ {LEAK_THRESHOLD}）：{len(leaks)} 条")
        for qid, r, b in leaks[:10]:
            print(f"    {qid} overlap={r} block={b}")
    if not missing and not leaks:
        print("\n[ok] 所有 gold 块存在，且无 query 照抄原文")
    return 0 if not missing and not leaks else 2


def cmd_eval(cfg: Config, args) -> int:
    import json as _json

    from pathlib import Path as _P

    from rag.eval.dataset import load_suite
    from rag.eval.runner import build_specs, run_spec, save_run
    from rag.service import Service

    suite_path = _P(args.suite)
    items = load_suite(suite_path)
    specs = build_specs(args.specs)
    svc = Service.create(cfg, load_reranker=any(s.rerank for s in specs))

    if svc.embedder.is_fake and not args.allow_fake:
        raise SystemExit(
            "拒绝执行：embedding.kind 是占位实现，排序无语义，产出的指标没有意义。\n"
            "如只是验证管线，加 --allow-fake。"
        )

    print(f"suite={suite_path.name} n={len(items)} specs={len(specs)} "
          f"embedder={svc.embedder.name} reranker={svc.searcher.reranker}")
    out_dirs = []
    for i, spec in enumerate(specs, 1):
        try:
            res = run_spec(
                svc, items, spec, top_k=args.top_k, candidates=args.candidates,
                rerank_top_n=args.rerank_top_n,
            )
        except Exception as e:
            # 典型情形：某个策略的 collection 还没建。整批矩阵会因此不可比，
            # 直接停下比跑出一半结果再解释要诚实。
            msg = str(e)
            if "does not exist" in msg or "collection not found" in msg:
                raise SystemExit(
                    f"中止：策略 {spec.strategy} 的 collection 不存在。"
                    f"先建索引：python tasks.py index-all\n原始错误：{msg[:160]}"
                )
            raise
        d = save_run(res, cfg, suite_path, out_root=_P(args.out) if args.out else None,
                     extra={
                         "suite_n": len(items),
                         "spec_index": i,
                         # 检索参数必须落盘：config.json 里是默认值，命令行可能覆盖它。
                         # 不记这一层，事后就无法确认某个数字是在什么召回深度下跑出来的。
                         "run_params": {
                             "top_k": args.top_k,
                             "candidates": args.candidates,
                             "rerank_top_n": args.rerank_top_n or spec.rerank_top_n,
                         },
                     })
        agg, lt = res.metrics["aggregate"], res.metrics["latency"]
        print(
            f"[{i:2d}/{len(specs)}] {spec.name():<40} "
            f"R@5={agg['recall@5']:.3f} Hit@5={agg['hit@5']:.3f} "
            f"MRR@10={agg['mrr@10']:.3f} nDCG@10={agg['ndcg@10']:.3f} "
            f"P50={lt['p50_ms']:.1f}ms P95={lt['p95_ms']:.1f}ms -> {d.name}"
        )
        out_dirs.append(str(d))
    print(_json.dumps({"runs": out_dirs}, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(cfg: Config, args) -> int:
    import uvicorn

    from rag.api.main import create_app

    uvicorn.run(create_app(cfg), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag", description="RAG 检索服务 + 评估体系")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="建库")
    p_ingest.add_argument("--strategies", nargs="*", default=None, help="要建索引的切分策略（默认取配置）")
    p_ingest.add_argument("--rebuild", action="store_true", help="先删除已有 collection")

    p_search = sub.add_parser("search", help="命令行检索")
    p_search.add_argument("query")
    p_search.add_argument("--strategy", default=None)
    p_search.add_argument("--mode", default=None)
    p_search.add_argument("--top-k", type=int, default=None)

    sub.add_parser("stats", help="查看索引状态")

    p_check = sub.add_parser("check-suite", help="校验评测集（gold 块存在性 + 泄漏检测）")
    p_check.add_argument("--suite", required=True, help="评测集 jsonl 路径")

    p_eval = sub.add_parser("eval", help="跑评测矩阵")
    p_eval.add_argument("--suite", required=True, help="评测集 jsonl 路径")
    p_eval.add_argument("--specs", default="rerank", choices=["smoke", "rerank", "chunk", "quick", "main"])
    p_eval.add_argument("--top-k", type=int, default=10)
    p_eval.add_argument("--candidates", type=int, default=50)
    p_eval.add_argument("--rerank-top-n", type=int, default=None)
    p_eval.add_argument("--allow-fake", action="store_true", help="允许在占位嵌入下跑（仅验证管线）")
    p_eval.add_argument("--out", default=None, help="结果输出根目录，默认 results/")

    p_serve = sub.add_parser("serve", help="启动 API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    cfg = Config.load(Path(args.config))
    handlers = {
        "ingest": cmd_ingest,
        "search": cmd_search,
        "stats": cmd_stats,
        "check-suite": cmd_check_suite,
        "eval": cmd_eval,
        "serve": cmd_serve,
    }
    return handlers[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
