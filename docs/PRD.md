# RAG 检索服务 + 评估体系 —— 需求文档（PRD）

| 项 | 内容 |
| --- | --- |
| 版本 | v0.3（M2 重排 + 评测执行后回填） |
| 日期 | 2026-11-15 |
| 状态 | M2 完成：已有第一轮实测指标（quick_v1，44 条），目标数字被实测推翻并修正，见 §20.14–20.17 |
| 定位 | 个人作品集项目（面试可讲、可复现、可被追问） |

---

## 1. 项目定位

**一句话**：不是"能跑通的 RAG Demo"，而是**一套可被证伪的检索方案对比实验平台** —— 同一个语料、同一套 200 条评测集、一条命令产出「召回率 / 命中率 / 延迟」横向对比表，并说明每一次优化为什么有效。

### 1.1 为什么这个项目能拉开差距

绝大多数 RAG 项目的产出是"我用了 LangChain + Milvus + 向量检索，回答效果不错"。"不错"没有定义，无法被追问，也无法被复用。

本项目的产出是**一张消融实验表**：4 种切分 × 6 种检索策略 × 200 条标注 query，每一格是可复现的数字，并且能回答下面 5 个追问：

1. 混合检索比纯向量召回高多少？高在哪种 query 上？——分难度切片回答。
2. rerank 涨了召回，代价多少毫秒？——延迟-召回权衡曲线。
3. 小 chunk 召回高是不是因为评测粒度不公？——统一按父块粒度评测（§12.3）。
4. 200 条评测集怎么造的？有没有自己给自己放水？——标注规范 + 泄漏检测（§12.2）。
5. 61% → 84% 这两个数怎么来的？——配置文件 hash + 一键复现脚本。

### 1.2 反 Demo 的四条硬指标（验收红线）

| # | 硬指标 | 通过标准 |
| --- | --- | --- |
| H1 | 可复现 | `make all` 从零跑完 20 组实验，seed 固定，产出表格与本文附录数字一致（误差 < 0.5pt） |
| H2 | 有对照 | 任何"优化"必须给出 baseline 数字与 Δ，禁止只报最终值 |
| H3 | 有失败分析 | 面板可下钻到单条 query 的检索链路 trace，至少 15 条失败 case 归因 |
| H4 | 有代价曲线 | 召回提升必须同时给 P50/P95 延迟与索引成本，不只报收益 |

---

## 2. 范围

**做（P0）**
- 语料入库 → 4 种切分策略 → 稠密/稀疏双索引 → 混合检索 → rerank → LLM 生成（LangGraph 编排）
- 自建 200 条评测集（含标注规范、质检、泄漏检测）
- 评测执行器（离线批量跑）+ 指标计算 + 统计显著性
- 指标面板（对比表、延迟分布、分切片下钻、失败 case 诊断）
- FastAPI 服务化 + docker-compose 一键起

**不做（明确砍掉，避免摊大饼）**
- 多模态（图片/表格解析）→ 只在语料清洗阶段跳过
- 用户体系、权限、多租户
- 在线 A/B 分流、线上流量采集（只做离线评测，成本可控）
- 模型微调（embedding / reranker 均用现成权重）

---

## 3. 指标定义（必须先定义，否则数字无意义）

记 $Q$ 为评测 query 集合，$R_q$ 为 query $q$ 的**标注相关块集合**（rel ≥ 1），$T_q^K$ 为检索返回的前 K 个块。

| 指标 | 公式 | 用途 | 备注 |
| --- | --- | --- | --- |
| Recall@K | $\frac{1}{|Q|}\sum_q \frac{|R_q \cap T_q^K|}{|R_q|}$ | 主指标（简历口径） | 衡量"漏没漏" |
| HitRate@K（命中率） | $\frac{1}{|Q|}\sum_q \mathbb{1}[|R_q \cap T_q^K|>0]$ | 辅助指标 | 单条 query 有 1 个相关块即算命中，比 Recall 宽松，二者必须同时报 |
| MRR@10 | $\frac{1}{|Q|}\sum_q \frac{1}{\text{rank}_q^{\text{first}}}$ | 衡量"排得前不前" | rerank 收益主要体现在这里 |
| nDCG@10 | $\sum \frac{2^{rel_i}-1}{\log_2(i+1)}$，rel∈{0,1,2} | 分级相关性 | 支持 2 级标注 |
| 引用准确率 | 答案中可溯源到引用块的句子数 / 引用句总数 | 生成侧 | 抽样 50 条人工核 |
| Latency P50/P95/P99 | 200 次串行请求分位值，先 warmup 20 次 | 代价指标 | **必须拆分阶段**，见 §7.2 |

**评测单元约定（关键）**：所有 Recall 一律在**父块（parent block, ~1024 token）粒度**计算。若按子块（256 token）粒度算，小 chunk 策略会因"一条 query 的相关内容被切成 4 块、命中 1 块即 recall 虚高"而系统性占优，这是自欺欺人。父块粒度下四种切分策略可比。

---

## 4. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  接入层  FastAPI (async)                                     │
│  /v1/search  /v1/chat(SSE)  /v1/eval/*  /v1/metrics/*        │
├─────────────────────────────────────────────────────────────┤
│  编排层  LangGraph StateGraph                                │
│  analyze → route → [dense ∥ sparse] → fuse → grade           │
│           ↑                              ↓                   │
│           └──── rewrite (≤2轮) ←── 不足 ─┘                    │
│                              → rerank → generate → cite_check │
├─────────────────────────────────────────────────────────────┤
│  检索层  Hybrid Retriever                                    │
│  dense: bge-m3 (1024d, HNSW/IP)                              │
│  sparse: BM25 (Milvus SPARSE_INVERTED_INDEX)                 │
│  fuse: RRF (k=60) / weighted                                  │
│  rerank: bge-reranker-v2-m3 (cross-encoder, top50→top5)      │
├─────────────────────────────────────────────────────────────┤
│  索引层  Milvus 2.4+  (collection per chunking strategy)      │
│  切分层  fixed / recursive / semantic / parent-child          │
├─────────────────────────────────────────────────────────────┤
│  评测层  runner → metrics → leaderboard → dashboard           │
│  可观测  trace 表（每阶段耗时/候选数/分数）                     │
└─────────────────────────────────────────────────────────────┘
```

**设计取舍说明（面试会被问）**
- BM25 用 Milvus 原生稀疏向量，而不是再起一套 Elasticsearch：少一个中间件、少一次跨服务往返、同一事务内可做混合过滤。代价是稀疏索引的调参与 ES 不完全等价，需在面板上给出对比结论。
- 每种切分策略独立 collection（`rag_chunks_fixed512` / `_recursive` / `_semantic` / `_parentchild`），而不是同 collection 加 `strategy` 字段过滤：避免不同策略共享 HNSW 图导致索引参数互相干扰，评测才干净。代价是内存 ×4，语料 3k 文档量级可接受。

---

## 5. 数据与存储设计

### 5.1 Milvus Collection Schema

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | VARCHAR(64) PK | `{strategy}_{doc_id}_{seq}` |
| `doc_id` | VARCHAR(64) | 源文档 |
| `parent_id` | VARCHAR(64) | 父块 id，用于 small-to-big 聚合与统一评测粒度 |
| `text` | VARCHAR(8192) | 块正文 |
| `token_len` | INT32 | 供分桶统计 |
| `strategy` | VARCHAR(32) | 冗余字段，便于排查 |
| `dense` | FLOAT_VECTOR(1024) | bge-m3，已 L2 归一化 |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 词权重 |
| `meta` | JSON | `{source, section_path, page, lang, ingest_at}` |

索引：`dense` → HNSW(M=16, efConstruction=200), metric=IP；`sparse` → SPARSE_INVERTED_INDEX, metric=IP。检索参数 `ef` 在 64/128/256 三档做敏感性扫描，取 P95 延迟达标的最小值。

### 5.2 元数据表（PostgreSQL 或 SQLite，二选一，默认 SQLite 降低依赖）

- `documents`：doc_id, source, title, n_chunks, hash, created_at
- `eval_suites`：suite_id, name, version, n_queries, annotator, kappa
- `eval_queries`：qid, suite_id, query, type, difficulty, gold_parent_ids(json), notes
- `eval_runs`：run_id, suite_id, config_hash, config_json, git_sha, started_at, metrics_json
- `traces`：trace_id, run_id/qid, stage, ms, n_candidates, extra_json

`config_hash` 是 H1 可复现的锚点：配置 → 哈希 → 结果，三者可反查。

---

## 6. 功能需求

### FR-01 语料接入与索引（P0）
- 支持目录批量导入 `.md/.txt/.pdf`；PDF 只取文本层，扫描件跳过并记录跳过清单。
- 切分 → 双向量编码 → 写 Milvus，全流程幂等（按文件 hash 去重，支持 `--rebuild`）。
- **验收**：3,000+ 段落语料入库，4 个 collection 均可用；重复执行不产生重复数据。

### FR-02 四种切分策略（P0）
见 §8，每种策略参数落配置文件，可单独开关。

### FR-03 混合检索（P0）
- 稠密 + 稀疏并行召回，RRF 融合；支持 `mode=dense|sparse|hybrid_rrf|hybrid_weighted`。
- 支持 metadata 过滤（source/section）。
- **验收**：单次 hybrid 检索两路并行，额外耗时 ≤ 8ms（相比单路）。

### FR-04 重排（P0）
- cross-encoder 精排，`rerank_top_n`（20/50/100）可配；无 GPU 时降级为小模型（bge-reranker-base）并记录降级标记。
- **验收**：top50→top5 单次 ≤ 200ms(P95, CPU 8C)；GPU ≤ 30ms。

### FR-05 LangGraph 编排（P0）
见 §11。要求：任一节点异常不整体 500，返回降级结果 + `degraded=true`。

### FR-06 生成与引用（P1）
- 严格基于检索块作答，输出句级引用 `[1][2]`。
- 无相关上下文时回答"资料中未提及"，禁止编造（评测集含 10 条 no-answer query 专门测这个）。

### FR-07 评测集构建（P0，最重的一项）
见 §12。交付物：`evalsets/v1/queries.jsonl`（200 条）+ `annotation_guide.md` + `qa_report.md`（一致性/泄漏/分布）。

### FR-08 评测执行器（P0）
- `POST /v1/eval/run` 或 CLI `python -m rag.eval run --config configs/exp/hybrid_rrf_rerank.yaml`。
- 20 组配置并行跑（进程池），产出 `results/{run_id}/per_query.csv` + `metrics.json`。
- 支持断点续跑、失败重试、进度条。

### FR-09 指标面板（P0）
见 §14。要求能下钻到单条 query 的完整 trace。

### FR-10 对比报告导出（P1）
一键生成 Markdown/CSV 对比表，直接贴进 README 与简历附件。

---

## 7. 非功能需求

### 7.1 性能预算（单 query，检索路径）

| 阶段 | 目标 P95 | 备注 |
| --- | --- | --- |
| query 编码（dense+sparse） | 20ms | bge-m3，CPU 需 batch=1 优化 |
| dense search top50 | 15ms | ef=128 |
| sparse search top50 | 18ms | B300 万级以下 |
| RRF 融合 + 去重 | 3ms | 父块聚合取 max |
| rerank 50→5 | 180ms(CPU)/25ms(GPU) | **主要延迟来源** |
| 上下文组装 | 5ms | |
| **检索链路合计** | **≤ 250ms (CPU)** | baseline（纯稠密无 rerank）≈ 55ms |
| 端到端含 LLM | ≤ 3.5s P95 | 流式首 token ≤ 800ms |

### 7.2 延迟测量规范
- 串行 200 次、warmup 20 次、排除首次 CUDA/模型加载；
- 分阶段计时（`time.perf_counter`，写入 `traces` 表），面板展示瀑布图；
- 并发压测 1/8/32 QPS 三档，记录 P95 退化曲线（32 QPS 下 P95 ≤ 2× 单路）。

### 7.3 其他
- 冷启动 ≤ 60s（模型 + 索引就绪）。
- 全流程 CPU 可跑（8C16G），GPU 可选加速 —— 保证面试官能复现。
- 依赖锁定（`uv lock` / `requirements.txt` 带版本），Docker 镜像可构建。

---

## 8. 切分策略规格

| ID | 策略 | 参数 | 实现要点 | 预期 |
| --- | --- | --- | --- | --- |
| S1 | 固定窗口 | 512 token, overlap 0 | 最简单 baseline，按句号对齐避免截断词 | 基线 |
| S2 | 递归切分 | 512 token, overlap 64, 分隔符 `\n\n→\n→。` | 尊重段落结构 | 通常优于 S1 |
| S3 | 语义切分 | 句向量相邻相似度，95 分位为断点，上限 512 | 用 bge-m3 句向量 + percentile 阈值 | 语义连贯但块长方差大 |
| S4 | 父子块（small-to-big） | child 256 / parent 1024, overlap 32 | 子块建索引，命中后回溯父块喂 LLM 并作为评测单元 | **预期最优**，但要防"召回虚高"质疑 |

补充规格：
- 所有策略记录 `token_len` 分布（面板出直方图），块长异常（>1500 或 <50 token）单独统计比例，>5% 视为策略实现有问题。
- 表格/代码块不切断（S1/S2 需前置检测 ```` ``` ```` 与 `|` 表格）。
- 统一 tokenizer（bge-m3 tokenizer），避免不同 tokenizer 导致块长不可比。

---

## 9. 检索与融合规格

- RRF：$\text{score}(d)=\sum_{i\in\{\text{dense},\text{sparse}\}} \frac{w_i}{k+\text{rank}_i(d)}$，$k=60$，默认 $w=1.0$；另测 $w=(0.7,0.3)$、$(0.5,0.5)$、$(0.3,0.7)$ 三组加权。
- 候选池：两路各取 top50 → 融合后取 top50 → rerank → top5（可配）。
- 去重：同一 `parent_id` 的多个子块只保留最高分，命中数计入 trace（用于观察 small-to-big 的实际贡献）。
- 空结果兜底：hybrid 为空时降级到稠密放宽阈值重试一次。

---

## 10. 重排规格

- 模型：`bge-reranker-v2-m3`（主）/ `bge-reranker-base`（CPU 降级）。
- 输入 `(query, chunk_text)` 对，batch 8，`max_length=512` 截断。
- 阈值策略：rerank 分数 < 0.15 的块直接丢弃，全部低于阈值 → 触发 §11 的 rewrite 分支（而非硬答）。
- 敏感性实验：`rerank_top_n ∈ {20, 50, 100}`，观察 Recall@5 饱和点与延迟拐点，写入面板。

---

## 11. LangGraph 编排

**State（TypedDict）**：`query, rewritten, candidates[], fused[], rerank_scores[], rounds, answer, citations[], degraded, trace{stage:ms}, config`

```
analyze_query ──► route ──┬─► retrieve_dense ─┐
                          ├─► retrieve_sparse ─┴─► fuse ─► grade_context
                          └─► direct_answer (寒暄/无关)        │
                                                    ┌────────┴────────┐
                                              sufficient          insufficient
                                                    │                 │
                                                    ▼                 ▼
                                                 rerank      rewrite(rounds<2) ──► retrieve
                                                    │                 │
                                                    ▼                 └─► fallback_answer
                                                generate ─► cite_check ─► END
```

- `grade_context`：用 rerank 分数分布做轻量判定（top1 < 0.15 且均值 < 0.1 判不足），**不额外调 LLM**，省一次 400ms 往返。这是刻意的成本取舍。
- 最大改写轮数 2，超过直接 fallback 并标记 `degraded=true`。
- 每个节点写入 trace，面板可回放单条 query 的完整链路（回答 H3 失败分析）。

---

## 12. 评测体系（项目最核心，也是最能被追问的部分）

### 12.1 双轨评测
| 轨道 | 用途 | 规模 |
| --- | --- | --- |
| A. 公开基准 | 证明管线实现正确、可与公开数字对齐 | C-MTEB `T2Retrieval` dev 子集 500 条 |
| B. 自建域内集 | 证明在目标语料上的真实效果（简历口径来源） | 200 条，见 12.2 |

A 轨的意义：如果 A 轨 Recall 明显低于公开 baseline，说明是**你的实现有 bug**，而不是"任务难"。这一条能挡掉"数字是自己拍的"的质疑。

### 12.2 200 条评测集构建规范

**语料**：3,000+ 段落中文技术文档（Milvus / FastAPI / LangGraph 官方文档 + 20 篇 RAG 论文摘要）。选它因为：领域术语密集（BM25 有戏）、跨段落引用多（能造多跳 query）、可公开。

**类型分层（必须按配额造，不能顺手写）**

| 类型 | 配额 | 说明 | 考察点 |
| --- | --- | --- | --- |
| 单跳·事实 | 90 | 答案在单一块内 | 基础召回 |
| 单跳·同义改写 | 40 | 用不同措辞问同一事实（不用原文词） | 稠密检索 vs BM25 差异 |
| 多跳·跨段 | 40 | 需 2 个以上块才能回答 | 混合检索优势区 |
| 指代/省略 | 20 | "它的默认值是多少" | query 改写价值 |
| 无答案 | 10 | 语料中确实没有 | 拒答能力（防幻觉） |

**难度分层**：easy 80 / medium 80 / hard 40（hard = 相关块排名天然靠后或表述差异大）。

**标注流程（三段式，防自己放水）**
1. **候选生成**：对随机块用 LLM 反向生成问题（避免人只写"好检索"的问题），人工改写去口语化。
2. **相关性标注**：对每条 query 用 hybrid 检索 top20 + 人工全文搜索，标 rel ∈ {0,1,2}（2=直接回答，1=部分相关/背景，0=无关）。标注时**不看**自己系统的排名，避免锚定。
3. **交叉复核**：20% 抽样双人标（可找同行/或用第二次独立标注），Cohen's κ ≥ 0.70 才通过；不达标重标该批次。

**泄漏检测（必做，否则数字虚高）**
- query 与所有 gold 块的最大 4-gram 重叠率 < 0.4；超阈值视为"抄原文"，重写 query。
- 检测 query 是否含 gold 块的罕见专有名词（如函数名）—— 含则记录，分切片报告"含术语 vs 不含术语"的差异。

**质检报告 `qa_report.md` 必含**：类型/难度分布、gold 块数分布（均值/中位）、κ 值、泄漏检测通过率、块长分布。

### 12.3 评测粒度与防作弊约定
- Recall 统一在 parent block 粒度（§3）。
- 同一 query 的所有配置用**完全相同**的候选池深度与过滤条件。
- 报告数字必须附 `config_hash` 与 `git_sha`。
- 显著性：200 条样本下，Recall 差异需做配对 bootstrap（1000 次）给 95% CI；Δ < 2pt 且 CI 跨 0 的结论一律标注"不显著"。

---

## 13. 实验矩阵与消融计划

### 13.1 分组实验（共 20 组，分 4 阶段）

| 阶段 | 变量 | 组数 | 固定项 |
| --- | --- | --- | --- |
| E1 切分对比 | S1–S4 | 4 | dense only, no rerank, top5 |
| E2 检索策略 | dense / sparse / rrf / weighted×3 | 6 | 取 E1 最优切分 |
| E3 重排与深度 | no-rerank / rerank(20/50/100) | 4 | 取 E2 最优融合 |
| E4 编排与代价 | E3 最优 ± query rewrite、± 并发档位 | 6 | 计入端到端延迟 |

### 13.2 目标对比表（数字待实测填充，此处为目标值 + 对应配置）

> 规则：下表**只允许被实测覆盖，不允许保留目标值交付**。空着的格子必须在 M3 结束前填满，填不满就如实写"未达标 + 原因"。

| 配置 | 切分 | 检索 | Rerank | Recall@5 | HitRate@5 | MRR@10 | P95 延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline（简历 61% 的口径） | S1 fixed512 | dense only | 无 | 0.61 (目标) | — | — | ~55ms |
| B1 | S2 recursive | dense only | 无 | — | — | — | — |
| B2 | S4 parent-child | dense only | 无 | — | — | — | — |
| B3 | S4 | sparse(BM25) only | 无 | — | — | — | — |
| B4 | S4 | hybrid RRF | 无 | — | — | — | — |
| B5 | S4 | hybrid weighted 0.7/0.3 | 无 | — | — | — | — |
| **final（简历 84% 的口径）** | **S4 parent-child** | **hybrid RRF** | **top50→5** | **0.84 (目标)** | — | — | ~250ms |

关键叙事（面试要能顺着讲）：
- 61→84 的 23 个点拆解：切分贡献 ≈ ? pt、混合检索 ≈ ? pt、rerank ≈ ? pt（**必须有拆分，合起来才可信**）。
- 代价：P95 从 55ms → 250ms（4.5×），换 23pt 召回。若业务是交互式问答可接受，若是高并发检索 API 则需 GPU 或降 rerank_top_n —— 这个权衡要在面板上可视化，而不是回避。

### 13.3 必须有的负结果
至少报告 1 个"做了但没效果"的尝试（预期：语义切分 S3 因块长方差大，Recall 不如 S4；或 query rewrite 对单跳 query 无增益却增加 500ms）。**主动报负结果是最强的工程信号。**

---

## 14. 指标面板

技术选型：FastAPI 提供 JSON + 单页静态 HTML（Chart.js），不引入 Streamlit（保留可嵌入其他系统的能力）。

| 页面 | 内容 | 对应追问 |
| --- | --- | --- |
| P1 总览 | 4 张 KPI 卡（当前最优 Recall@5 / HitRate@5 / P95 / 索引条数）+ 最近 run 列表 | 现在什么水平 |
| P2 策略对比 | 策略 × 指标热力表，支持点选两列做 diff（Δ 与 CI） | 哪个方案好，好多少，显著吗 |
| P3 延迟分析 | 阶段瀑布 + P50/P95/P99 箱线 + 并发退化曲线 | 代价多少 |
| P4 召回曲线 | Recall@K / nDCG@K（K=1..20）多配置折线 | 是不是只是 top5 调参 |
| P5 切片下钻 | 按 类型/难度/含术语 分组的指标柱状（含样本量 n） | 强在哪、弱在哪 |
| P6 单条诊断 | 输入 qid 看：query、改写、双路候选、RRF 分数、rerank 分数、最终上下文、耗时 | 失败原因是什么 |
| P7 失败归因 | 15+ 条失败 case 的人工归因标签统计（切分错/召回漏/排序错/生成错） | 下一步优化什么 |

---

## 15. API 设计（摘要）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/search` | `{query, strategy, mode, top_k, rerank, filters}` → hits + `timings` |
| POST | `/v1/chat` | SSE 流式，返回 answer + citations + trace_id |
| POST | `/v1/eval/run` | 启动评测任务 → `job_id` |
| GET | `/v1/eval/jobs/{id}` | 进度 / 结果 / 失败明细 |
| GET | `/v1/eval/leaderboard?suite=v1` | 对比表数据（面板 P2 数据源） |
| GET | `/v1/traces/{trace_id}` | 单条链路（面板 P6） |
| POST | `/v1/admin/ingest` | 重建索引（带 rebuild 开关） |
| GET | `/healthz` `/readyz` | 存活 / 就绪（模型与索引加载完成） |

---

## 16. 目录结构（规划）

```
RAG/
├─ docs/PRD.md                 # 本文档
├─ configs/
│  ├─ exp/{e1..e4}_*.yaml      # 每组实验的完整配置（config_hash 来源）
│  └─ default.yaml
├─ evalsets/v1/                # queries.jsonl + annotation_guide.md + qa_report.md
├─ src/rag/
│  ├─ api/            # FastAPI 路由
│  ├─ ingest/         # loader, chunkers/{fixed,recursive,semantic,parentchild}.py
│  ├─ index/          # milvus client, schema, bm25 sparse encoder
│  ├─ retrieve/       # dense.py, sparse.py, fuse.py, rerank.py
│  ├─ graph/          # LangGraph nodes + state
│  ├─ eval/           # runner, metrics, bootstrap, leakage, report
│  └─ observe/        # trace 记录
├─ dashboard/         # 单页 HTML + Chart.js
├─ tests/             # 指标计算单测（用构造数据验公式）+ 泄漏检测单测
├─ docker-compose.yml # milvus + api
└─ Makefile           # make index / make eval-all / make report / make serve
```

单测重点：**指标函数必须有构造用例**（例如 5 条 query 手算 Recall 与实现比对）。否则面板上的数字一错全错。

---

## 17. 里程碑与验收

| 里程碑 | 交付 | 验收标准 |
| --- | --- | --- |
| M1 骨架与单路检索 | 语料入库、dense 检索、`/v1/search` 可用 | 3k 段落入库，单路 Recall@5 有数（不追求高） |
| M2 混合 + 重排 + 4 切分 | 6 种检索模式、4 个 collection、rerank | 20 组配置全部可跑通 |
| M3 评测集与评测管线 | 200 条标注集、执行器、指标、质检报告 | 表 13.2 填满；κ ≥ 0.7；泄漏检测 100% 通过 |
| M4 面板 + 编排 + 交付 | 7 个面板页、LangGraph、压测、README | H1–H4 全过；一条命令复现附录数字 |

---

## 18. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 评测集自建、自己标 → 数字虚高 | 面试被质疑致命 | 双轨（公开基准对齐）+ 泄漏检测 + κ + 不看排名标注 |
| rerank 在 CPU 太慢拖垮演示 | 演示体验差 | 保留 base 小模型降级路径；默认 top50 而非 100 |
| 混合检索提升不明显（<3pt） | 核心叙事崩塌 | 若真不显著，就如实报告并分析原因（语料术语密度不足），改为"检索策略选择取决于语料类型"的结论 —— 这也是有价值的工程结论 |
| Milvus 4 份 collection 内存吃紧 | 跑不动 | 分策略分批建索引；或用 partition + 共享索引做对照说明 |
| LLM 生成侧不稳定 | 引用准确率不可测 | 生成侧只做 P1，主指标全部落在检索侧（可控、可复现） |
| 只做检索不做生成被说"不完整" | 观感 | 交付最小可用生成链路 + 引用校验，但明确声明主指标不在生成侧 |

---

## 19. 附录：简历表述 ↔ 证据映射

| 简历语句 | 证据位置 | 面试可展开 |
| --- | --- | --- |
| "构建混合检索 RAG 服务" | §4 架构图 + `src/rag/retrieve/` + `/v1/search` | 为何用 Milvus 稀疏向量而非 ES；RRF 为何 k=60 |
| "自建 200 条评测集" | `evalsets/v1/` + §12.2 规范 + qa_report | 类型配额、κ、泄漏检测、为何按父块粒度评测 |
| "Recall@5 61% → 84%" | 表 13.2 + `results/{run_id}/metrics.json` + config_hash | 23pt 拆分（切分/混合/重排各多少）、bootstrap CI |
| "延迟横向对比" | 面板 P3 + §7.1 预算表 | rerank 占端到端 70% 延迟；top_n 的拐点 |
| （加分）"主动报告负结果" | §13.3 | 语义切分为何没赢、代价意识 |

**表述纪律**：最终 README 与简历中的每个数字，必须能在 `results/` 下找到对应 `run_id` 与配置文件。做不到就不写这个数字。

---

## 20. 实现阶段的修正（v0.1 → v0.2）

需求文档一旦开始实现就会暴露假设错误。以下四条是 M1 阶段对原设计的修正，**原条款保留、修正记录在此**，避免后人按过期文档实现。

### 20.1 评测粒度：从「父块」改为「策略无关的评测块网格」

原文 §3 要求"统一在父块粒度评测"。实现时发现这个说法有漏洞：S1 固定窗口与 S2 递归切分**没有父块层级**，只有 S4 有。若 S1/S2 按自身 chunk 粒度、S4 按父块粒度，等于各用各的尺子，仍不可比。

修正做法：对每篇文档额外铺一层固定 1024-token 的网格（`eval_block_id`，形如 `docid:b3`），与切分策略完全无关。所有 chunk 通过**字符区间最大重叠**映射到网格，Recall 一律在网格粒度统计。S4 的父块与网格高度重合，因此也自然满足原意。

实现位置：`src/rag/ingest/blocks.py`，单测 `test_chunking.py::test_eval_block_grid_is_strategy_independent`。

### 20.2 `meta` JSON 字段拆成标量字段

原 §5.1 把 source/title/section 放在 `meta` JSON 里。面板要按 source/section 过滤与分组（P5 切片下钻），而 JSON 字段无法建标量索引、也无法 group by。

修正：拆成三个独立 VARCHAR 字段（`source` / `title` / `section`），可直接用于过滤表达式与聚合。

### 20.3 部署形态：无 Docker → Milvus Lite

原方案假设 docker-compose 起 Milvus standalone。实际环境未安装 Docker 且 WSL 被安全策略拦截。

已验证 **Milvus Lite 3.2.2 同时支持 `FLOAT_VECTOR` 与 `SPARSE_FLOAT_VECTOR` 的写入与检索**（这是能否沿用"BM25 走 Milvus 稀疏向量"方案的前提），因此：
- 本地开发：`uri = data/milvus/rag.db`（文件库，无需 Docker）
- 线上部署：`uri = http://milvus:19530`（同一套代码，只换配置）

需要注意的 Lite 差异（已写进代码注释）：
1. 稠密索引用 `AUTOINDEX` 而非 `HNSW`（Lite 不支持 HNSW 参数调优，`ef` 需在 standalone 上才能扫）；
2. 新进程首次检索前必须显式 `load_collection`，否则报 `collection released`；
3. 首次 `search` 有约 300ms 段懒加载，必须预热（见 20.5）。

### 20.4 BM25 统计量按策略隔离存储

原 §5 只写了"全库 fit 一次"。但每个策略一个 collection，意味着 chunk 集合不同 → `df` / `avgdl` 不同 → IDF 不同。若共用一份统计文件，S4 的 ID 会套在 S1 的语料统计上，分数不可比。

修正：统计文件按策略分存 `data/bm25/bm25_{strategy}.json`。

### 20.5 新增三级预热（原 §7.3 只说"冷启动 ≤ 60s"）

实测发现三处冷启动叠加会让首请求达到 395ms（稳态仅 4.6ms，差 85 倍）：
jieba 词典加载 400ms、collection load 200ms、Milvus 首次 search 段懒加载 366ms。

修正：`Service.warmup()` 显式做三件事（切一次词、load 集合、打一次空检索）。预热后首请求 19.5ms，boot 841ms。

### 20.6 关于 §13.2 目标数字的现状

**目前表中所有数字仍是目标值，一个都没被实测覆盖。** M1 阶段只验证了链路，且用的是 `embedding.kind: hash` 占位嵌入（指标无意义），语料也只有 8 篇种子文档。

按 §19 纪律，在真实语料 + 真实嵌入 + 200 条评测集跑完之前，任何人（包括我）都不得把 61% / 84% 写进简历。下一步见 §17 的 M2/M3。

### 20.7 嵌入模型：bge-m3 → bge-small-zh-v1.5

原计划用 bge-m3（2.2GB / 1024 维）。实际环境为纯 CPU 且 `huggingface.co` 直连超时，2.2GB 下载成本与收益不匹配。

改用 `BAAI/bge-small-zh-v1.5`（95MB / 512 维 / 24M 参数）。

| 维度 | bge-m3 | bge-small-zh-v1.5 | 影响 |
| --- | --- | --- | --- |
| 体积 | 2.2GB | 95MB | 下载 20 倍差距 |
| 向量维度 | 1024 | 512 | 稠密索引内存减半 |
| query 编码（CPU） | 估 40–60ms | 实测 6.7ms | 检索链路主成本 |
| 中文检索质量 | 更强 | 够用（C-MTEB 检索子集差距约 3–5 分） | 结论方向不受影响 |
| 多语言 | 支持 | 中英为主 | 语料是中文技术文档，无影响 |

两个实现细节（踩过才有资格写）：
1. **查询侧指令前缀**：BGE 系列在 s2p（短查询检索长段落）场景要求只在 query 侧拼 `为这个句子生成表示以用于检索相关文章：`，文档侧不拼。两侧都拼或都不拼都会掉点。bge-m3 不需要指令，所以 `query_instruction` 置空即可。
2. **HF 镜像**：`HF_ENDPOINT=https://hf-mirror.com`，否则下载直接超时。配置里给了 `embedding.hf_endpoint` 字段，海外环境置空。

换回 bge-m3 的操作：改 `embedding.kind/model/dim` 三项 → `python tasks.py index-all`（带 `--rebuild`）。代码零改动，`build_embedder` 会校验配置维度与模型实际输出维度，不一致直接报错，避免"配置写 1024、模型出 512"这种只在写 Milvus 时才炸的隐性问题。

实测：模型加载 12.9s（含三级预热），在 §7.3 的 60s 预算内；稳态 hybrid_rrf P50 16.8ms。

### 20.8 新发现：种子语料做不出区分度，M3 扩语料是前置条件

8 篇种子文档共 30 个 chunk，四种配置的 top-1 几乎全部命中正确文档，Recall@5 基本都会是 1.0。**这不是检索好，是语料太小**：30 个 chunk 里取前 5，任何方法都不会太差。

结论：种子语料只能用于联调，不能用于评测。M3 必须先把语料扩到 3000 段量级，否则会得出"混合检索没有提升"的错误结论——而这个错误结论会被当作负结果写进报告，比没有结论更糟。

### 20.9 语料来源变更：Milvus / LangGraph → Kubernetes

原定"Milvus / FastAPI / LangGraph 官方中文文档"。实测后两条落空：

| 源 | 预期 | 实测 |
| --- | --- | --- |
| Milvus | `/docs/zh/` 有中文 | URL 与 sitemap 都存在（504 条中文路径），但**正文 100% 英文**。抽查 overview / schema / search / manage-collections 等 6 页，中文占比均为 0.0%——只翻译了导航壳 |
| LangGraph | 官方文档 | 无中文版，`sitemap.xml` 返回 404 |
| FastAPI | `/zh/` 中文 | ✅ 153 页，正文确为中文 |
| Kubernetes | 备选 | ✅ `zh-cn` 1036 页，正文确为中文 |

用英文正文配中文查询，等于给稠密检索凭空加一层跨语言难度，指标差异无法归因到检索策略本身。因此剔除 Milvus / LangGraph，改用 **FastAPI 中文（153 页）+ Kubernetes 中文（1036 页）**。

连带影响（做 M3 时必须按新语料调整，别照抄旧计划）：
1. **query 内容要换域**：原计划的"Milvus 索引参数 / LangGraph 状态机"类问题不存在了，改成 K8s 概念（调度、网络、存储、控制器）与 FastAPI 用法（依赖注入、参数校验、异步）。
2. **多跳 query 的来源更充足**：K8s 文档天然跨页引用（Deployment → ReplicaSet → Pod），造跨段 query 比 Milvus 更容易。
3. **20 篇 RAG 论文摘要暂不加**：与语料语言/领域都不一致，会引入噪声。若后期要加，单独立一个"论文语料"实验组，不与主实验混。

语料细节、抓取脚本与复现方式见 [docs/corpus.md](corpus.md)。

### 20.10 真实规模实测 vs §7.1 预算

语料落地后（1171 篇 / 6.94M 字符 / parentchild 12032 块），把 §7.1 的预算表逐项对了一遍：

| 阶段 | §7.1 预算 | 实测 P50 | 结论 |
| --- | --- | --- | --- |
| query 编码 | ≤ 20ms | 8.0ms | ✅ 达标，且与语料规模无关 |
| dense search top50 | ≤ 15ms | 16.3ms | ⚠️ 略超，且是**暴力检索的上界**（Lite 无 HNSW），换 standalone 会降 |
| sparse search top50 | ≤ 18ms | 13.1ms | ✅ |
| 融合 + 去重 | ≤ 3ms | 0.09ms | ✅ |
| 检索链路合计 | ≤ 250ms（含 rerank） | 38.7ms（不含 rerank） | ✅ 余量 ~200ms 留给 rerank |
| 冷启动 | ≤ 60s | 11.9s | ✅ |

两个结论：
1. **§7.1 的预算结构是对的**，250ms 里 rerank 该占大头（150–200ms），目前余量充足。
2. **Milvus Lite 的数字是上界**，因为它只用 FLAT/AUTOINDEX 暴力检索。线上换 standalone + HNSW 后 dense 阶段应降到个位数毫秒——但这一点必须在有 Docker 的环境里验证，不能直接写进报告。

### 20.11 已知缺口：S4 只做了一半

§8 对父子块的要求是"命中子块后回溯父块喂给 LLM"。现状：**检索侧完成，回溯侧未实现**——`Hit` 只带子块文本，没有回取父块。

影响：
- 对**评测无影响**（Recall 按 `eval_block_id` 算，本来就在父块粒度）；
- 对**生成有影响**：256 token 的子块上下文不足以支撑推理，且 CLI 输出看起来都是碎片。

处理：M2 里给 `Hit` 增加 `parent_text`（按 `parent_id` 批量回查 Milvus），由 `small_to_big` 开关控制。这条先记下来，避免把"检索结果看起来碎"误判成切分策略的问题。

### 20.12 token 估计器用真实 tokenizer 校准（评测口径的 bug 修复）

原 §8 的 token 计数是手写规则：「中文 1 token + ASCII 单词 1.3 token + 其余 0」。两个问题：

1. **ASCII 部分低估 28%–35%**。`PNGStreamingResponse` 算 1.3 token，真实是 6 token。
2. **标点计 0 token**，于是「------」这类表格分隔行在窗口累加时**免费**，评测块可以无限膨胀。实测出现过 **17153 字符**的块。

块过大的后果不是"看起来丑"，而是**指标失去意义**：一个 17k 字符的块横跨多个子主题，多种切分策略的 chunk 都能映射到它，Recall 虚高且没有区分度。

修法：随机抽 300 个语料窗口，用 bge-small-zh 的真实 tokenizer 数 token，对
（中文字符数，ASCII 字符数，其他非空白字符数）做最小二乘，得到

```
tokens ≈ 0.968 * 中文字符 + 0.220 * ASCII 字符 + 1.397 * 其他字符
R² = 0.98，单窗口相对误差中位数 3.5%，全语料总 token 偏差 -0.3%
```

修正后评测块 token 数分布：P50=1006、P90=1024、max=1036（此前 max 无上界）。块数从 2953 变为 3844。

**代价**：chunk 尺寸随之变化，`eval_block_id` 全部重算，三个策略必须重建索引。这也是为什么这条必须在正式评测之前修，而不是之后。

### 20.13 空稀疏向量导致建库失败

token 计数修正后，段落切分（`recursive`）下会出现独立的「纯标点块」（如表格分隔线 `|---|---|`）。这类块经 jieba 分词后没有词项，稀疏向量全零，Milvus 直接拒收整批：

```
code=6, message=sparse vector field 'sparse': sparse vector must contain at least one nonzero float32 weight
```

一次失败的建库要重跑十几分钟才暴露问题，所以修两道：

1. `chunker` 用 `has_indexable_text()` 丢弃无词项块（它们本来也不承载检索信息）；
2. `pipeline` 在写库前按实际分词结果再过滤一次，并打印 `dropped_empty` 计数。

第 2 道是兜底，不是冗余——切分逻辑以后还会改（S3 语义切分），不能指望它永远不产生空块。


### 20.14 重排：CPU 上超预算 15 倍，默认关闭

M2 接入 `bge-reranker-base`（278M，CPU/fp32）后实测：

| 配置 | P50 | 重排阶段 |
| --- | ---: | ---: |
| hybrid（无重排） | 47.5 ms | — |
| hybrid + rerank（池 15） | 3834.7 ms | 3790 ms |

每候选约 250ms。这台机器 fp32 实测约 150 GFLOPS，处理一条 256-token 输入约需 200ms，
**是算力决定的，不是实现问题**。已做的优化：max_length 512→256（约 1.7 倍）、
doc_max_chars 800→600、候选池 50→15。再往下压就只能牺牲重排质量。

结论：
1. §7.1 的「250ms 含重排」在无 GPU 环境下不可达，差距是数量级的，调参救不回来。
2. 上线路径只有三条：GPU（按同模型估 25–40ms）、蒸馏小模型、或把重排做成异步二次精排。
3. 因此 `rerank.enabled` 默认 `false`，重排作为**可开关的评估能力**而非默认链路。
   这个取舍要写进 README，否则"有 rerank 但默认关"看起来像没做完。

### 20.15 两个与预期相反的实测结果

**(1) 父子块召回最差。** §8 假设 S4 最好（子块精匹配 + 父块补上下文）。实测 Recall@5：
fixed512 62.5% / recursive 62.5% / parentchild **50.0%**（hybrid_rrf）。

归因：子块中位数仅 244 token（约 430 字符），短块的稠密向量区分度不足，BM25 侧也因文档过短
导致分数不稳，融合后互相拖累。佐证：parentchild 的 dense 单路（43.2%）反而**高于**
fixed512 的 dense（39.8%），说明短块适合稠密、不适合与 BM25 融合。
另排除了"父块去重吃掉候选"这个解释：实测 top-10 内平均块数 10.0，去重几乎没折叠。

修正认知：**父子块的价值在"喂给 LLM 的上下文完整"，不在召回**。要发挥它必须配合父块回填（§20.11）。

**(2) 重排对 parentchild + dense 是负作用**：Recall@5 43.2% → 42.0%（-1.1pt），
但 MRR@10 0.241 → 0.340（+0.099）。重排优化的是排名质量，不是 top-5 命中数；
Recall@5 分母只有 5 时，个别位置交换就会让指标抖动。所以报告必须同时给 MRR@10，
否则会得出"重排没用"的错误结论。

### 20.16 重排分数是唯一能做拒答阈值的分数

| 配置 | 有答案 top1 均分 | 无答案 top1 均分 |
| --- | ---: | ---: |
| hybrid（RRF） | 0.030 | 0.031 |
| hybrid + rerank | 0.869 | 0.095 |

RRF 分数是 `Σ 1/(k+rank)`，只有相对意义、随候选数变化，**天然不能当阈值**。
cross-encoder 输出相关性概率，有绝对含义。所以"检索不到就拒答"这个能力只有在重排之后才可能实现。
当前只有 6 条无答案样本，定不出阈值——M3 需要专门的负样本集来标定（PRD §12.4 原本只写了 10 条，偏少）。

### 20.17 §13.2 目标数字的实测覆盖情况

| 指标 | v0.1 目标 | 实测 | 状态 |
| --- | ---: | ---: | --- |
| Recall@5（目标配置 parentchild + hybrid + rerank） | 84% | 58.0% | ❌ 未达标 |
| Recall@5（实测最优：fixed512 + hybrid + rerank） | — | **70.5%** | — |
| Recall@5（dense 基线） | 61% | 39.8% | ❌ 基线本就低于目标 |
| P95（含 rerank） | 250ms | 3937ms | ❌ 差 15 倍 |

结论：**61% / 84% 两个数字都不成立**，简历改按实测写「Recall@5 从 40% 提升到 71%」。
目标配置也从 parentchild 改为 fixed512——因为 §20.15 的负结果推翻了原假设。
