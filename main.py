#!/usr/bin/env python
"""PyCharm / 命令行一键启动入口。

在 PyCharm 里直接右键本文件 → Run 'main'，或在终端 `python main.py`。

为什么需要这个文件：项目是 src 布局（`src/rag/...`），`rag` 包不在默认
import 路径上，直接 `uvicorn rag.api.main:create_app` 会 ModuleNotFoundError。
这里在导入前把 `src` 插进 sys.path，因此**不依赖**IDE 里手动标记 Sources Root，
也不会往 site-packages 里装东西。

与 `python tasks.py serve` 的关系：两者最终都调 `rag.api.main:create_app`，
区别是 tasks.py 通过子进程转发、需要 PYTHONPATH 环境变量，适合 CI；
本文件是单进程，适合 IDE 调试（断点能落在 rag 包内部）。

调试提示：PyCharm 的 Debug 请**不要**加 `--reload`。reload 模式下 uvicorn 会
fork 子进程重新 import 应用，断点会失效或落在错误的进程上。
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# 必须在 import rag.* 之前完成。用 insert(0) 而不是 append：避免同名的
# 已安装包（例如别的项目 pip install 过名为 rag 的库）抢在前面。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_CONFIG = "configs/default.yaml"


def resolve_config(path: str | Path | None = None) -> Path:
    """相对路径一律相对仓库根解析，这样从任何工作目录启动结果都一致。"""
    p = Path(path or os.environ.get("RAG_CONFIG") or DEFAULT_CONFIG)
    return p if p.is_absolute() else ROOT / p


def build_app():
    """uvicorn 应用工厂。

    --reload 时 uvicorn 需要 import string 而不能传实例，此时配置路径通过
    环境变量 RAG_CONFIG 传给重载后的子进程（子进程会重新执行本模块顶层代码）。
    """
    from rag.api.main import create_app
    from rag.config import Config

    cfg_path = resolve_config()
    if not cfg_path.exists():
        raise SystemExit(f"配置文件不存在：{cfg_path}")
    return create_app(Config.load(cfg_path))


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="main",
        description="RAG 检索服务启动入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径")
    ap.add_argument("--reload", action="store_true", help="代码改动自动重启（调试时勿开）")
    ap.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    a = ap.parse_args(argv)

    cfg_path = resolve_config(a.config)
    os.environ["RAG_CONFIG"] = str(cfg_path)  # 供 --reload 的子进程读取

    import uvicorn

    # 端口被占时说人话，而不是抛一大段 OSError 堆栈。--reload 的子进程会继承
    # 同一端口，这里只在主进程检查。
    if not _port_available(a.host, a.port):
        print(
            f"[main] 端口 {a.host}:{a.port} 已被占用。\n"
            f"       换端口：python main.py --port 8123\n"
            f"       或先停掉占用进程：netstat -ano | findstr :{a.port}",
            file=sys.stderr,
        )
        return 1

    print(f"[main] 配置 {cfg_path}")
    print(f"[main] 服务 http://{a.host}:{a.port}  (/healthz /readyz /v1/search /v1/stats /docs)")
    print("[main] 首次 /v1/search 会加载嵌入模型并预热，约 10-15s；那之前的请求会看起来“卡住”")

    if a.reload:
        uvicorn.run(
            "main:build_app",
            factory=True,
            host=a.host,
            port=a.port,
            reload=True,
            reload_dirs=[str(SRC)],  # 只监听源码，避免 data/、results/ 抖动触发重启
            log_level=a.log_level,
        )
    else:
        uvicorn.run(build_app(), host=a.host, port=a.port, log_level=a.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
