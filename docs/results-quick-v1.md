# 第一轮真实评测结果（quick_v1）

> 生成方式：`python tasks.py eval --suite evalsets/quick_v1.jsonl --specs quick`
> 汇总：`python scripts/report.py --out docs/results-quick-v1.md`
> 本文所有数字都能通过表格里的 `run_id` 在 `results/` 下找到对应的 `metrics.json` 与 `config.json`。

## 0. 一句话结论

在 44 条有答案的单跳题上，**Recall@5 从 dense 单路的 39.8% 提升到 混合检索+重排的 70.5%**。
拆开看：**混合检索贡献 +22.7pt，重排再贡献 +8.0pt**。但重排在 CPU 上 P50 达 3.8 秒，
是 250ms 预算的 15 倍——这个代价值得单独讨论（见 §5）。

## 1. 实验条件

| 项 | 值 |
| --- | --- |
| 语料 | Kubernetes 中文文档 1028 篇 + FastAPI 中文文档 143 篇（共 1171 篇 / 6.94M 字符） |
| 索引规模 | fixed512 7143 块 / recursive 9471 块 / parentchild 17269 块 |
| 评测块粒度 | 1024 token（与切分策略无关的固定网格，共 3844 个块） |
| 嵌入 | `BAAI/bge-small-zh-v1.5`（512 维，CPU，query 侧带指令前缀） |
| 重排 | `BAAI/bge-reranker-base`（278M，CPU，max_length=256） |
| 检索参数 | 每路召回 20 → 融合 → 重排池 15 → 取 top-10 |
| 评测集 | `evalsets/quick_v1.jsonl`：44 条有答案 + 6 条无答案，全部单跳题 |
| 硬件 | 16 核 CPU / 无 GPU / fp32 |
| 配置哈希 | `9bf34d0270dd` |

**这不是最终数字**：44 条样本、单 gold、单人标注。它的作用是给方向，不是给结论。
局限逐条写在 `evalsets/README.md`。

## 2. 主结果

最佳配置（fixed512 + hybrid_rrf + rerank）与各基线的对比：

| 配置 | Recall@5 | Hit@5 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| dense 单路 | 39.8% | 40.9% | 0.286 | 0.344 |
| BM25 单路 | 54.5% | 54.5% | 0.455 | 0.501 |
| 混合（RRF） | 62.5% | 63.6% | 0.444 | 0.497 |
| **混合 + 重排** | **70.5%** | **70.5%** | **0.527** | **0.571** |

### 增益拆解

```
dense 39.8%  ──混合检索 +22.7pt──▶  62.5%  ──重排 +8.0pt──▶  70.5%
```

只报 39.8% → 70.5% 而不说清各占多少，等于没解释。这里两个数字分开给了。

### 统计显著性（成对 bootstrap，2000 次重采样）

| 对比 | Δ Recall@5 | 95% CI | 判定 |
| --- | ---: | --- | --- |
| dense → hybrid（fixed512） | +22.7 pt | [+9.1, +36.4] | 显著 |
| dense → hybrid（recursive） | +29.5 pt | [+15.9, +43.2] | 显著 |
| dense → hybrid（parentchild） | +6.8 pt | [-4.5, +18.2] | 不显著 |
| hybrid → hybrid+rerank（fixed512） | +8.0 pt | [-5.7, +20.5] | 不显著 |

为什么必须做成对检验：44 条样本上单组 Recall@5 的 95% 置信区间约 ±13pt，两个独立区间
一比，10pt 以内的差异根本说不清。成对检验消掉了"两个配置都答对/都答错"的样本，
所以能看出 dense→hybrid 这类大效应，但**重排那 8pt 在 n=44 下仍然落在噪声区间内**。
要确认重排的增益，得等 200 条评测集。这一点不能含糊过去。

## 3. 负结果一：父子块（S4）在这套语料上最差

PRD 原假设 S4 会最好（子块精确匹配 + 父块提供上下文）。实测相反：

| 策略 | hybrid_rrf Recall@5 | +rerank |
| --- | ---: | ---: |
| fixed512 | 62.5% | 70.5% |
| recursive | 62.5% | 67.0% |
| parentchild | **50.0%** | 58.0% |

parentchild 的子块中位数只有 244 token（约 430 字符）。这类短块的稠密向量区分度不足，
BM25 侧也因文档过短导致分数不稳定，两路融合后反而互相拖累。数据佐证：parentchild 的
dense 单路（43.2%）其实高于 fixed512 的 dense（39.8%），说明**短块对稠密检索不差，
是融合把它拉下来了**。

同时排除了另一个可能：不是父块去重把候选吃掉了。实测 top-10 里平均块数为 10.0，
说明去重几乎没折叠（`avg_blocks_in_top10` 记录在 per_query 里）。

结论：**父子块的价值在"喂给 LLM 的上下文完整"，不在于召回**。要发挥它，得配合
"子块召回 + 父块回填"（PRD §20.11 记的缺口），而不是指望它提升 Recall。

## 4. 负结果二：重排对 parentchild + dense 是负作用

| 配置 | Recall@5 关→开 | MRR@10 关→开 |
| --- | --- | ---: |
| parentchild + dense | 43.2% → **42.0%**（-1.1pt） | 0.241 → 0.340（+0.099） |

Recall 掉了，MRR 明显升了。意思是：重排把原本刚好排在第 5 位的正确块挤到了第 6 位，
但同时把第 8 位的正确块提到了第 2 位。**重排在优化"排名质量"，不是优化"top-5 命中数"**——
当 Recall@5 的分母只有 5 时，个别位置交换就会让指标抖动。这也说明单看 Recall@5 会掩盖
重排的真实作用，所以报告里 MRR@10 必须一起给。

## 5. 延迟：重排在 CPU 上超预算 15 倍

| 配置 | P50 | 重排阶段均值 |
| --- | ---: | ---: |
| hybrid（无重排） | 47.5 ms | — |
| hybrid + rerank | **3834.7 ms** | 3790 ms |

每候选约 250ms（15 个候选 → 3.8s）。这台机器 fp32 实测约 150 GFLOPS，
278M 参数的 cross-encoder 处理一条 256-token 输入需要约 200ms 是**算力决定的，不是实现问题**。
已经做的优化：`max_length` 512→256（约 1.7 倍）、`doc_max_chars` 800→600、候选池 15。

结论（写进 PRD §20.14）：
- **CPU 上重排不适合放在在线路径**。250ms 预算在无 GPU 环境下不可达，差距是数量级的，调参救不回来。
- 上线只有三条路：GPU（估 25–40ms）、蒸馏小模型、或把重排做成异步二次精排。
- 因此默认配置 `rerank.enabled: false`，重排作为可开关的离线/评估能力，而不是默认链路。

## 6. 一个意外收获：重排分数可以拿来做拒答

| 配置 | 有答案题 top1 均分 | 无答案题 top1 均分 | 比值 |
| --- | ---: | ---: | ---: |
| hybrid（无重排） | 0.030 | 0.031 | 1.0（**完全无区分**） |
| hybrid + rerank | 0.869 | 0.095 | 9.1 |

RRF 的分数是 `Σ 1/(k+rank)`，只有相对意义，绝对值随候选数变化，**天然不能做阈值**。
cross-encoder 输出的是相关性概率，有绝对含义。这意味着"检索不到就拒答"这个能力
只有在重排之后才可能实现——6 条无答案题上 0.869 vs 0.095 的分离度是一个可用的起点，
但 6 条样本定不出阈值，需要专门的负样本集来标定（记入 M3）。

## 7. 与简历表述的对应

| 简历原表述 | 实测 | 应改为 |
| --- | --- | --- |
| Recall@5 从 61% 优化至 84% | dense 39.8% → hybrid+rerank 70.5% | **Recall@5 从 40% 提升到 71%** |

61/84 是 PRD v0.1 写下的目标值，从未被实测覆盖。现在实测值出来了，就按实测写。
按 §19 的纪律：70.5% 对应 `results/20261115-110545__fixed512__hybrid_rrf__rr`，
配置哈希 `9bf34d0270dd`，评测集 `quick_v1.jsonl`（sha1 记录在该 run 的 config.json 里）。

另外两个可以在简历里用的数字（都有 run 可查）：
- 混合检索相对稠密单路 **+22.7pt**，成对 bootstrap 95% CI [+9.1, +36.4]，显著；
- 端到端检索（不含重排）P95 **54.2ms**，其中 query 编码 11.2ms、稠密 13.8ms、稀疏 21.3ms。

---

# 附：自动生成的完整表格

### 数据集：quick_v1.jsonl　配置哈希：9bf34d0270dd

共 15 个 run，有答案题 44 条。

## 1. 主矩阵：切分策略 × 检索方式 × 重排

| 切分策略 | 检索方式 | rerank | Recall@5 | Hit@5 | MRR@10 | nDCG@10 | P50 (ms) | P95 (ms) | run_id |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| fixed512 | dense | 关 | 39.8% | 40.9% | 0.286 | 0.344 | 24.2 | 27.9 | `20261115-105853__fixed512__dense` |
| fixed512 | dense | 开 | 54.5% | 54.5% | 0.421 | 0.457 | 3851.0 | 3977.3 | `20261115-110216__fixed512__dense__rr` |
| fixed512 | sparse | 关 | 54.5% | 54.5% | 0.455 | 0.501 | 22.9 | 28.1 | `20261115-110219__fixed512__sparse` |
| fixed512 | hybrid_rrf | 关 | 62.5% | 63.6% | 0.444 | 0.497 | 47.5 | 54.2 | `20261115-110221__fixed512__hybrid_rrf` |
| fixed512 | hybrid_rrf | 开 | 70.5% | 70.5% | 0.527 | 0.571 | 3834.7 | 3937.4 | `20261115-110545__fixed512__hybrid_rrf__rr` |
| recursive | dense | 关 | 33.0% | 34.1% | 0.268 | 0.311 | 24.1 | 28.7 | `20261115-110546__recursive__dense` |
| recursive | dense | 开 | 53.4% | 54.5% | 0.409 | 0.448 | 3744.5 | 3785.9 | `20261115-110904__recursive__dense__rr` |
| recursive | sparse | 关 | 54.5% | 54.5% | 0.429 | 0.478 | 18.2 | 33.4 | `20261115-110906__recursive__sparse` |
| recursive | hybrid_rrf | 关 | 62.5% | 63.6% | 0.413 | 0.468 | 47.4 | 58.0 | `20261115-110909__recursive__hybrid_rrf` |
| recursive | hybrid_rrf | 开 | 67.0% | 68.2% | 0.493 | 0.549 | 3771.4 | 3814.1 | `20261115-111229__recursive__hybrid_rrf__rr` |
| parentchild | dense | 关 | 43.2% | 43.2% | 0.241 | 0.301 | 31.2 | 34.9 | `20261115-111230__parentchild__dense` |
| parentchild | dense | 开 | 42.0% | 43.2% | 0.340 | 0.393 | 3667.4 | 3796.2 | `20261115-111541__parentchild__dense__rr` |
| parentchild | sparse | 关 | 54.5% | 54.5% | 0.368 | 0.428 | 25.2 | 41.0 | `20261115-111544__parentchild__sparse` |
| parentchild | hybrid_rrf | 关 | 50.0% | 50.0% | 0.356 | 0.435 | 61.9 | 72.8 | `20261115-111547__parentchild__hybrid_rrf` |
| parentchild | hybrid_rrf | 开 | 58.0% | 59.1% | 0.419 | 0.487 | 3644.1 | 3824.2 | `20261115-111859__parentchild__hybrid_rrf__rr` |

## 2. 检索方式相对 dense 基线的差值（成对 bootstrap，重排关闭）

| 切分策略 | 对比 | Δ Recall@5 | 95% CI (pt) | 变好/变差/不变 | n |
| --- | --- | ---: | --- | --- | ---: |
| fixed512 | dense → sparse | +14.8 pt | [+0.0, +30.7] 不显著 | 10/4/30 | 44 |
| fixed512 | dense → hybrid_rrf | +22.7 pt | [+9.1, +36.4] 显著 | 11/1/32 | 44 |
| recursive | dense → sparse | +21.6 pt | [+3.4, +40.9] 显著 | 15/6/23 | 44 |
| recursive | dense → hybrid_rrf | +29.5 pt | [+15.9, +43.2] 显著 | 13/0/31 | 44 |
| parentchild | dense → sparse | +11.4 pt | [-6.8, +29.5] 不显著 | 10/5/29 | 44 |
| parentchild | dense → hybrid_rrf | +6.8 pt | [-4.5, +18.2] 不显著 | 5/2/37 | 44 |

## 3. 重排增益（同配置开关对比，成对 bootstrap）

| 切分策略 | 检索方式 | Δ Recall@5 | 95% CI (pt) | 变好/变差/不变 | Δ MRR@10 | n |
| --- | --- | ---: | --- | --- | ---: | ---: |
| fixed512 | dense | +14.8 pt | [+0.0, +28.4] 不显著 | 9/3/32 | +0.135 | 44 |
| fixed512 | hybrid_rrf | +8.0 pt | [-5.7, +20.5] 不显著 | 7/4/33 | +0.083 | 44 |
| recursive | dense | +20.5 pt | [+6.8, +34.1] 显著 | 10/1/33 | +0.142 | 44 |
| recursive | hybrid_rrf | +4.5 pt | [-4.5, +15.9] 不显著 | 4/2/38 | +0.080 | 44 |
| parentchild | dense | -1.1 pt | [-15.9, +13.6] 不显著 | 6/6/32 | +0.099 | 44 |
| parentchild | hybrid_rrf | +8.0 pt | [-8.0, +23.9] 不显著 | 9/5/30 | +0.063 | 44 |

## 4. 重排增益点估计

| 切分策略 | 检索方式 | Recall@5 关→开 | Δ Recall@5 | MRR@10 关→开 | Δ MRR@10 | Δ P95 (ms) |
| --- | --- | --- | ---: | --- | ---: | ---: |
| fixed512 | dense | 39.8% → 54.5% | +14.8 pt | 0.286 → 0.421 | +0.135 | +3949.4 |
| fixed512 | hybrid_rrf | 62.5% → 70.5% | +8.0 pt | 0.444 → 0.527 | +0.083 | +3883.2 |
| recursive | dense | 33.0% → 53.4% | +20.5 pt | 0.268 → 0.409 | +0.142 | +3757.2 |
| recursive | hybrid_rrf | 62.5% → 67.0% | +4.5 pt | 0.413 → 0.493 | +0.080 | +3756.1 |
| parentchild | dense | 43.2% → 42.0% | -1.1 pt | 0.241 → 0.340 | +0.099 | +3761.3 |
| parentchild | hybrid_rrf | 50.0% → 58.0% | +8.0 pt | 0.356 → 0.419 | +0.063 | +3751.4 |

## 5. 分题型

| 配置 | 题型 | n | Recall@5 | MRR@10 |
| --- | --- | ---: | ---: | ---: |
| fixed512/dense | single_hop | 44 | 39.8% | 0.286 |
| fixed512/dense/rr | single_hop | 44 | 54.5% | 0.421 |
| fixed512/sparse | single_hop | 44 | 54.5% | 0.455 |
| fixed512/hybrid_rrf | single_hop | 44 | 62.5% | 0.444 |
| fixed512/hybrid_rrf/rr | single_hop | 44 | 70.5% | 0.527 |
| recursive/dense | single_hop | 44 | 33.0% | 0.268 |
| recursive/dense/rr | single_hop | 44 | 53.4% | 0.409 |
| recursive/sparse | single_hop | 44 | 54.5% | 0.429 |
| recursive/hybrid_rrf | single_hop | 44 | 62.5% | 0.413 |
| recursive/hybrid_rrf/rr | single_hop | 44 | 67.0% | 0.493 |
| parentchild/dense | single_hop | 44 | 43.2% | 0.241 |
| parentchild/dense/rr | single_hop | 44 | 42.0% | 0.340 |
| parentchild/sparse | single_hop | 44 | 54.5% | 0.368 |
| parentchild/hybrid_rrf | single_hop | 44 | 50.0% | 0.356 |
| parentchild/hybrid_rrf/rr | single_hop | 44 | 58.0% | 0.419 |

## 6. 延迟拆解（阶段均值为该配置全部 query 的平均）

| 配置 | P50 | P95 | embed | dense | sparse | fuse | rerank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed512/dense | 24.2 | 27.9 | 10.1 | 13.4 | — | 0.0 | — |
| fixed512/dense/rr | 3851.0 | 3977.3 | 10.9 | 12.6 | — | 0.0 | 3820.9 |
| fixed512/sparse | 22.9 | 28.1 | — | — | 22.0 | 0.0 | — |
| fixed512/hybrid_rrf | 47.5 | 54.2 | 11.2 | 13.8 | 21.3 | 0.1 | — |
| fixed512/hybrid_rrf/rr | 3834.7 | 3937.4 | 11.3 | 13.2 | 21.3 | 0.1 | 3790.0 |
| recursive/dense | 24.1 | 28.7 | 10.4 | 13.8 | — | 0.0 | — |
| recursive/dense/rr | 3744.5 | 3785.9 | 10.8 | 14.4 | — | 0.0 | 3708.4 |
| recursive/sparse | 18.2 | 33.4 | — | — | 19.7 | 0.0 | — |
| recursive/hybrid_rrf | 47.4 | 58.0 | 10.2 | 14.6 | 23.2 | 0.1 | — |
| recursive/hybrid_rrf/rr | 3771.4 | 3814.1 | 10.7 | 14.3 | 24.5 | 0.1 | 3720.5 |
| parentchild/dense | 31.2 | 34.9 | 11.0 | 19.3 | — | 0.0 | — |
| parentchild/dense/rr | 3667.4 | 3796.2 | 10.7 | 17.5 | — | 0.0 | 3582.2 |
| parentchild/sparse | 25.2 | 41.0 | — | — | 27.1 | 0.0 | — |
| parentchild/hybrid_rrf | 61.9 | 72.8 | 10.4 | 18.6 | 31.8 | 0.1 | — |
| parentchild/hybrid_rrf/rr | 3644.1 | 3824.2 | 10.6 | 17.9 | 31.3 | 0.1 | 3583.9 |
