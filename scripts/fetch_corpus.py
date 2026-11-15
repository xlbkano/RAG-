"""抓取官方中文文档，生成评测语料。

为什么自己写而不是用现成爬虫框架：只需要「取 sitemap → 抽正文 → 存 md」三段，
引入 scrapy 反而要处理更多配置。这里 200 行以内就够了。

三条纪律：
1. **可复现**：sitemap 驱动 + manifest 记录，同样的 sitemap 跑出同样的文件集；
2. **可断点**：已存在且非空的文件跳过，中断后重跑不重复下载；
3. **不冒犯**：并发 4、每请求间隔 0.2s、失败退避重试，不压对方服务器。

用法：
    python scripts/fetch_corpus.py --limit 10      # 先试 10 页
    python scripts/fetch_corpus.py                 # 全量
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "corpus"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rag-eval-lab/0.1; personal research)"}
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "p", "li", "pre", "blockquote", "tr", "dd", "dt"}
PREFIX = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "li": "- ", "dd": "- ", "dt": "- "}

SOURCES = [
    {
        "name": "fastapi",
        "sitemaps": ["https://fastapi.tiangolo.com/zh/sitemap.xml"],
        "include": r"/zh/",
        "exclude": r"/(release-notes|contributing|help-fastapi|management|benchmarks)/",
        "strip": r"^zh/",
        # mkdocs-material：正文在 article.md-content__inner，侧栏是 md-sidebar
        "selectors": ["article.md-content__inner", "article"],
    },
    {
        "name": "kubernetes",
        "sitemaps": ["https://kubernetes.io/zh-cn/sitemap.xml"],
        "include": r"/zh-cn/docs/",
        "exclude": r"/zh-cn/docs/(reference/command-line-tools-reference/kubelet-tls|setup/production-environment/tools/kubeadm/create-cluster-kubeadm)/",
        "strip": r"^zh-cn/docs/",
        # Docsy 主题：正文在 div.td-content，外层 td-main 会把整条侧边栏一起带进来
        "selectors": ["div.td-content", "main"],
    },
]

# 说明：Milvus 的 /docs/zh/ 路径实测返回 100% 英文正文（无中文翻译），
# 与「中文语料 + 中文查询 + bge-small-zh」的设定不匹配，故剔除。详见 PRD §20.9。

# 站点固定的噪音行（导航、翻译提示、页脚 CTA）。只删整行匹配，不删包含正文的句子。
NOISE = [
    r"^本翻译由人类引导的 AI 生成",
    r"^可能存在误解原意或不够自然等问题",
    r"^你可以通过 帮助我们更好地引导 AI LLM 来改进此翻译",
    r"^Use the official FastAPI skill",
    r"^See Develop with AI",
    r"^Install it from your project directory",
    r"^英文版本$",
    r"^(Was this page helpful\?|Edit this page|Create an issue|Feedback|On this page|Table of contents)$",
    r"^Try Managed Milvus for Free$",
    r"^Zilliz Cloud is hassle-free",
    r"^(略|Skip to content)$",
]
NOISE_RE = [re.compile(p) for p in NOISE]


def src_strip(source: str) -> str:
    for s in SOURCES:
        if s["name"] == source:
            return s.get("strip", "")
    return ""


def _slug(url: str, source: str) -> str:
    path = re.sub(r"^https?://[^/]+", "", url).strip("/")
    path = re.sub(r"\.md$", "", path)
    path = re.sub(src_strip(source), "", path) if src_strip(source) else path
    path = re.sub(r"/(index)?$", "", path)
    slug = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", path).strip("_")
    return (slug or "index")[:120]


def _walk(node, out: list[str]) -> None:
    for child in getattr(node, "children", []):
        name = getattr(child, "name", None)
        if name in BLOCK_TAGS:
            if name == "pre":
                # 关键：代码块必须用空分隔符取文本。用 "\n" 会把每个 span 拆成一行，
                # 结果是"每行一个词"的废文本（FastAPI 的 Pygments 高亮尤其严重）。
                code = child.get_text("", strip=False)
                code = re.sub(r"\n{3,}", "\n\n", code).strip("\n")
                if code:
                    out.append(f"\n```\n{code}\n```\n")
                continue
            text = child.get_text(" ", strip=True)
            if not text or len(text) < 2:
                continue
            text = re.sub(r"\s+", " ", text).replace("¶", "").strip()
            if any(rx.search(text) for rx in NOISE_RE):
                continue
            out.append(PREFIX.get(name, "") + text)
        elif name is not None:
            _walk(child, out)


def html_to_md(html: str, selectors: list[str] | None = None) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "aside", "noscript", "svg", "form", "button"]):
        tag.decompose()

    root = None
    for sel in list(selectors or []) + ["article", "main", "[role=main]", ".md-content", "#content"]:
        node = soup.select_one(sel)
        if node and len(node.get_text(strip=True)) > 300:
            root = node
            break
    root = root or soup.body or soup

    for tag in root.select("nav, .breadcrumbs, .table-of-contents, .toc, .pagination, .md-sidebar"):
        tag.decompose()

    lines: list[str] = []
    _walk(root, lines)

    text = "\n\n".join(lines)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    deduped: list[str] = []
    for ln in text.split("\n\n"):
        if deduped and ln.strip() == deduped[-1].strip():
            continue
        deduped.append(ln)
    return "\n\n".join(deduped).strip()


def collect_urls(client: httpx.Client, src: dict) -> list[str]:
    urls: list[str] = []
    for sm in src["sitemaps"]:
        r = client.get(sm, timeout=30)
        r.raise_for_status()
        locs = re.findall(r"<loc>(.*?)</loc>", r.text)
        for u in locs:
            if re.search(src["include"], u) and not re.search(src["exclude"], u):
                urls.append(u)
    return sorted(set(urls))


def fetch_one(client: httpx.Client, url: str, src: dict, dest: Path) -> tuple[str, int, str]:
    for attempt in range(3):
        try:
            r = client.get(url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            md = html_to_md(r.text, src["selectors"])
            if len(md) < 150:
                return url, 0, "too-short"
            title = ""
            m = re.search(r"^#\s+(.+)$", md, re.M)
            if m:
                title = m.group(1).strip()
            header = f"# {title or dest.stem}\n\n" if not md.startswith("# ") else ""
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(header + md + "\n", encoding="utf-8")
            time.sleep(0.2)  # 礼敬请
            return url, len(md), "ok"
        except Exception as e:
            if attempt == 2:
                return url, 0, f"{type(e).__name__}"
            time.sleep(1.5 * (attempt + 1))
    return url, 0, "unreachable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每个源最多抓多少页（调试用）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    manifest_path = out_root / "manifest.json"
    manifest: dict = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    limits = httpx.Limits(max_connections=args.workers * 2)
    total_ok = total_chars = 0
    with httpx.Client(headers=HEADERS, limits=limits, timeout=30) as client:
        for src in SOURCES:
            urls = collect_urls(client, src)
            if args.limit:
                urls = urls[: args.limit]
            print(f"[{src['name']}] sitemap 命中 {len(urls)} 页", flush=True)

            jobs = []
            for u in urls:
                dest = out_root / src["name"] / f"{_slug(u, src['name'])}.md"
                if dest.exists() and dest.stat().st_size > 200:
                    continue  # 断点续抓
                jobs.append((u, dest))

            ok = skip = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(fetch_one, client, u, src, d): (u, d) for u, d in jobs}
                for i, fut in enumerate(as_completed(futs), 1):
                    url, chars, status = fut.result()
                    if status == "ok":
                        ok += 1
                        total_chars += chars
                        manifest[url] = {"path": str(futs[fut][1].relative_to(ROOT)), "chars": chars}
                    else:
                        skip += 1
                        print(f"  ! {status} {url}", flush=True)
                    if i % 50 == 0:
                        print(f"  ... {i}/{len(jobs)}", flush=True)
            print(f"[{src['name']}] 新抓 {ok} 页，跳过 {skip} 页，已有 {len(urls) - len(jobs)} 页", flush=True)
            total_ok += ok

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n完成：新增 {total_ok} 页，累计 {len(manifest)} 页，正文 {total_chars / 1000:.0f}k 字符")
    print(f"输出目录：{out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
