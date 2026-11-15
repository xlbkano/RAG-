"""配置加载与 config_hash（H1 可复现的锚点）。

config_hash 的作用：任何一次评测结果都必须能反查到唯一一份配置。
规则见 PRD §19：数字 ↔ run_id ↔ config_hash，断链即视为不可信。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()
REPO_ROOT = Path(__file__).resolve().parents[2]


class Config:
    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data
        self.path = path

    # ---------- 构造 ----------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
        if not p.is_absolute():
            p = REPO_ROOT / p
        with p.open("r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {}, p)

    # ---------- 读取 ----------
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                if default is _MISSING:
                    raise KeyError(f"config key not found: {dotted}")
                return default
            cur = cur[part]
        return cur

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data, default=str))

    # ---------- 派生 ----------
    def path_of(self, dotted: str) -> Path:
        """把配置里的相对路径解析成仓库内绝对路径。"""
        p = Path(self.get(dotted))
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def collection_name(self, strategy: str) -> str:
        return f"{self.get('milvus.collection_prefix')}_{strategy}"
