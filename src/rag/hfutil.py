"""HuggingFace 模型定位。

踩过的坑：模型已经完整缓存在本地，但 sentence-transformers 每次实例化仍会去 Hub
解析 revision。网络一抖动（实测遇到 502 Bad Gateway）服务就直接起不来——本地明明
有模型，却因为一次 HEAD 请求失败而不可用。

做法：先问 huggingface_hub 要本地快照路径（local_files_only=True，纯本地、无网络），
拿不到再退回模型名走正常下载。这样首次部署走网络，之后完全离线可启动。
"""
from __future__ import annotations

import os


def local_snapshot_or_name(model: str) -> str:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model, local_files_only=True)
    except Exception:
        return model


def apply_endpoint(hf_endpoint: str | None) -> None:
    """国内直连 huggingface.co 会超时，改用镜像。已设置的环境变量不覆盖。"""
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
