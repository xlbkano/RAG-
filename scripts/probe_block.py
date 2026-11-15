"""按内容定位评测块（block），用于给评测集打 gold 标签。

为什么不用检索结果定位：
那份 PRD 明确写了「标注时不看自己系统的排名」。如果拿 HybridSearcher 的 top-1 当
gold，标注就被检索器污染了——系统召不回来的内容永远不会成为 gold，Recall 天然虚高。
所以这里只做**纯文本包含匹配**：给定一个答案里的特征词（probe），在语料原文里找到
它出现的位置，映射到该文档的 1024-token 网格，输出 block_id。

用法：
    python scripts/probe_block.py probes.txt          # 每行一个 probe
    python scripts/probe_block.py probes.txt --window 260
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import Config  # noqa: E402
from rag.ingest.blocks import build_block_grid, span_index  # noqa: E402
from rag.ingest.loader import load_documents  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", help="每行 `qid ||| doc_pattern ||| probe`；doc_pattern 留空表示全语料搜索")
    ap.add_argument("--window", type=int, default=260, help="匹配点前后展示的字符数")
    ap.add_argument("--max", type=int, default=1, help="每个 probe 最多展示几个命中文档")
    a = ap.parse_args()

    cfg = Config.load()
    docs = load_documents(cfg.path_of("corpus.dir"), cfg.get("corpus.globs"))
    bt = int(cfg.get("chunking.eval_block_tokens", 1024))

    for raw_line in Path(a.probes).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|||")]
        if len(parts) == 3:
            qid, doc_pat, probe = parts
        elif len(parts) == 2:
            qid, doc_pat, probe = parts[0], "", parts[1]
        else:
            qid, doc_pat, probe = "", "", parts[0]
        print(f"\n### {qid or '-'}  probe={probe!r}  doc~{doc_pat or '*'}")

        hits = 0
        pool = [d for d in docs if (not doc_pat or doc_pat in d.source)]
        if not pool:
            print(f"  !!! 未找到匹配 {doc_pat!r} 的文档")
            continue
        for d in pool:
            pos = d.text.find(probe)
            if pos < 0:
                continue
            grid = build_block_grid(d.text, bt)
            bi = span_index(grid, pos, pos + len(probe))
            if bi >= len(grid):  # span_index 的兜底：越界时不打标
                continue
            s, e = grid[bi]
            lo = max(s, pos - a.window // 2)
            hi = min(e, pos + len(probe) + a.window // 2)
            seg = d.text[lo:hi].replace("\n", " ")
            print(f"  doc={d.source}")
            print(f"  block={d.doc_id}:b{bi}  (block 字符区间 {s}-{e}, 共 {e - s} 字符)")
            print(f"  ...{seg}...")
            hits += 1
            if hits >= a.max:
                break
        if hits == 0:
            print("  !!! 未命中：probe 在语料中不存在，需改写")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
