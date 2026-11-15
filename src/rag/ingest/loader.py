"""语料加载：目录批量读取 .md/.txt。

doc_id 用「相对路径的 sha1 前 12 位」，不用文件名 —— 保证重命名文件名后
已标注的 gold block 仍然可对齐（只要目录结构不变）。
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from rag.types import Document

_TITLE = re.compile(r"^#\s+(.+)$", re.M)


def _doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _title(text: str, fallback: str) -> str:
    m = _TITLE.search(text)
    return m.group(1).strip() if m else fallback


def load_documents(corpus_dir: Path, globs: list[str]) -> list[Document]:
    if not corpus_dir.exists():
        raise FileNotFoundError(f"语料目录不存在: {corpus_dir}")

    files: list[Path] = []
    for g in globs:
        files.extend(sorted(corpus_dir.glob(g)))
    files = sorted({p for p in files if p.is_file()})

    docs: list[Document] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        source = p.relative_to(corpus_dir).as_posix()
        docs.append(
            Document(
                doc_id=_doc_id(source),
                source=source,
                title=_title(text, p.stem),
                text=text,
            )
        )
    return docs
