"""跨平台任务入口（Windows 上没有 make，所以用 Python 写）。

用法：
    python tasks.py index                # 按配置建当前策略的索引
    python tasks.py index-all            # 三个已实现策略全建（重建）
    python tasks.py search "HNSW 的 M 参数" --mode hybrid_rrf
    python tasks.py serve                # 起 API
    python tasks.py stats                # 看索引状态
    python tasks.py test                 # 跑单测
    python tasks.py check-suite --suite evalsets/quick_v1.jsonl
    python tasks.py eval --suite evalsets/quick_v1.jsonl --specs rerank
    python tasks.py report               # 汇总 results/ 出对比表
    python tasks.py clean                # 删索引与 BM25 统计（可重建）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def run(mod_args: list[str]) -> int:
    return subprocess.call([sys.executable, *mod_args], cwd=ROOT, env=ENV)


def main() -> int:
    p = argparse.ArgumentParser(prog="tasks")
    p.add_argument(
        "cmd",
        choices=[
            "index", "index-all", "search", "serve", "stats", "test", "clean",
            "check-suite", "eval", "report",
        ],
    )
    p.add_argument("rest", nargs=argparse.REMAINDER)
    a = p.parse_args()
    rest = [x for x in a.rest if x != "--"]

    if a.cmd == "index":
        return run(["-m", "rag.cli", "ingest", *rest])
    if a.cmd == "index-all":
        return run(["-m", "rag.cli", "ingest", "--strategies", "fixed512", "recursive", "parentchild", "--rebuild", *rest])
    if a.cmd == "search":
        return run(["-m", "rag.cli", "search", *rest])
    if a.cmd == "serve":
        return run(["-m", "rag.cli", "serve", *rest])
    if a.cmd == "stats":
        return run(["-m", "rag.cli", "stats"])
    if a.cmd == "check-suite":
        return run(["-m", "rag.cli", "check-suite", *rest])
    if a.cmd == "eval":
        return run(["-m", "rag.cli", "eval", *rest])
    if a.cmd == "report":
        return run([str(ROOT / "scripts" / "report.py"), *rest])
    if a.cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", *rest], cwd=ROOT, env=ENV)
    if a.cmd == "clean":
        for d in ("data/milvus", "data/bm25"):
            path = ROOT / d
            if path.exists():
                shutil.rmtree(path)
                print(f"removed {path}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
