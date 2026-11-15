"""指标单测。这里的每个数字都是手算的——指标算错整个评测就没有意义，
而这类错误在真实数据上极难发现（数字看起来总是"合理"）。
"""
from __future__ import annotations

import pytest

from rag.eval import metrics as M


def test_ranked_blocks_dedups_keeping_first_order():
    """父子块场景：同一个父块被切成 4 个子块，命中 4 条也只算 1 个块。"""
    assert M.ranked_blocks(["b1", "b1", "b2", "b1", "b3"]) == ["b1", "b2", "b3"]


def test_recall_uses_dedup_then_cut():
    """不去重会得出 recall=1.0（5 条命中里有 4 条是 b1），去重后正确结果是 1/3。

    这正是 PRD §3 担心的虚高：切得越碎，重复命中越多，块粒度 Recall 越假。
    """
    ranked_raw = ["b1", "b1", "b1", "b1", "b1"]
    gold = {"b1", "b2", "b3"}
    assert M.recall_at_k(M.ranked_blocks(ranked_raw), gold, 5) == pytest.approx(1 / 3)
    assert M.recall_at_k(ranked_raw, gold, 5) == pytest.approx(1 / 3), "单块 gold 时去重与否不影响"


def test_hit_and_mrr():
    assert M.hit_at_k(["x", "b2"], {"b2"}, 5) == 1.0
    assert M.hit_at_k(["x", "y"], {"b2"}, 5) == 0.0
    assert M.mrr_at_k(["x", "y", "b2"], {"b2"}, 10) == pytest.approx(1 / 3)
    assert M.mrr_at_k(["x", "y"], {"b2"}, 1) == 0.0


def test_ndcg_perfect_ranking_is_one():
    gains = {"b1": 2.0, "b2": 1.0}
    assert M.ndcg_at_k(["b1", "b2"], gains, 10) == pytest.approx(1.0)


def test_ndcg_penalizes_lower_position():
    gains = {"b1": 2.0}
    top = M.ndcg_at_k(["b1", "x", "y"], gains, 10)
    third = M.ndcg_at_k(["x", "y", "b1"], gains, 10)
    assert top == pytest.approx(1.0)
    assert 0 < third < top


def test_ndcg_ignores_irrelevant_above():
    """无关文档排在前面不扣分（它们增益为 0），但会把 gold 挤到低位置导致折损。"""
    gains = {"b1": 2.0}
    a = M.ndcg_at_k(["b1"], gains, 10)
    b = M.ndcg_at_k(["z", "b1"], gains, 10)
    assert a > b


def test_empty_gold_returns_zero_not_none():
    """无答案题不该参与 Recall 聚合，但函数本身返回 0 而不是 None，
    避免调用方把 None 混进平均时静默变成 nan。"""
    assert M.recall_at_k(["b1"], set(), 5) == 0.0
    assert M.hit_at_k(["b1"], set(), 5) == 0.0
    assert M.mrr_at_k(["b1"], set(), 5) == 0.0


def test_percentile_interpolates():
    xs = [1.0, 2.0, 3.0, 4.0]
    assert M.percentile(xs, 0.50) == pytest.approx(2.5)
    assert M.percentile(xs, 1.00) == pytest.approx(4.0)
    assert M.percentile([], 0.5) == 0.0


def test_leakage_detects_copied_query():
    from rag.eval.dataset import leakage

    block = "ConfigMap 是一种 API 对象，用来将非机密性的数据保存到键值对中。"
    copied = "ConfigMap 是一种 API 对象，用来将非机密性的数据保存到键值对中"
    paraphrased = "不涉及机密的应用配置想在集群里单独管理，该用哪个资源？"
    assert leakage(copied, block) >= 0.4
    assert leakage(paraphrased, block) < 0.4
