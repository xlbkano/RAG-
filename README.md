# RAG 检索服务 + 评估体系

> 不是"能跑通的 RAG Demo"，而是一套**可被证伪的检索方案对比实验平台**。
> 同一语料、同一套评测集、一条命令产出「召回率 / 命中率 / 延迟」横向对比表。

**技术栈**：FastAPI · Milvus（稠密 + 稀疏双索引）· bge-small-zh-v1.5 · bge-reranker-base · BM25（自实现）· LangGraph（待接）
**核心**：混合检索（dense + BM25, RRF 融合）· Cross-encoder 重排 · 3 种切分策略横向对比 · 自建评测集 + 成对显著性检验

📄 **需求文档**：[docs/PRD.md](docs/PRD.md) —— 指标定义、评测集构建规范、消融矩阵、简历证据映射
📊 **第一轮评测结果**：[docs/results-quick-v1.md](docs/results-quick-v1.md) —— 含负结果与延迟账

---

## 一句话结果

在 44 条有答案的单跳题上，**Recall@5 从 dense 单路的 39.8% 提升到 混合检索 + 重排的 70.5%**。
拆解：混合检索 +22.7pt（成对 bootstrap 95% CI [+9.1, +36.4]，显著），重排再 +8.0pt（CI 跨 0，不显著）。

| 配置（fixed512） | Recall@5 | MRR@10 | P50 | P95 |
| --- | ---: | ---: | ---: | ---: |
| dense 单路 | 39.8% | 0.286 | 24 ms | 28 ms |
| BM25 单路 | 54.5% | 0.455 | 23 ms | 28 ms |
| 混合 RRF | 62.5% | 0.444 | 47 ms | 54 ms |
| 混合 + 重排 | **70.5%** | **0.527** | 3835 ms | 3937 ms |

三个必须一起看的结论：
1. **混合检索是主要贡献者**，不是重排。只报最终数字而不拆解，等于没解释。
2. **重排在 CPU 上 P50 3.8 秒**，是 250ms 预算的 15 倍——算力决定，调参救不回来，默认关闭。
3. **父子块策略在这套语料上召回最差**（50.0% vs fixed512 62.5%），与预期相反，属负结果。

---

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| M0 | 需求文档（含指标定义、评测规范、消融矩阵） | ✅ |
| M1 | 骨架：语料入库 → 3 策略切分 → 双索引 → 混合检索 → API | ✅ |
| M1.5 | 语料扩量：1171 篇 / 6.94M 字符，三策略建库 + 真实规模压测 | ✅ |
| M2 | 重排接入 · 评测执行器 · 50 条评测集 · 对比表 · 成对显著性检验 | ✅ |
| M3 | 200 条评测集（多跳 + 多 gold + 双人复核）+ S3 语义切分 | ⬜ 下一步 |
| M4 | 父块回溯 · LangGraph 编排 · 指标面板 · 压测 | ⬜ |

---

## 语料

| 源 | 页数 | 字符 | 内容 |
| --- | --- | --- | --- |
| FastAPI 中文文档 | 143 | 1.1M | 代码密集、参数说明、短段落 |
| Kubernetes 中文文档 | 1028 | 5.8M | 概念解释、长段落、术语密集 |
| **合计** | **1171** | **6.94M** | 官方文档，可公开、可复现 |

切分后：`fixed512` **7143** 块 / `recursive` **9471** 块 / `parentchild` **17269** 块；
评测块网格 **3844** 个（1024 token，与策略无关）。

抓取：`python scripts/fetch_corpus.py`（sitemap 驱动、断点续抓、并发 4 + 0.2s 间隔）。
来源取舍与证据见 [docs/corpus.md](docs/corpus.md)——**Milvus 的 `/docs/zh/` 实测正文 100% 英文**，
LangGraph 无中文版，故剔除，改用 Kubernetes 补足。

---

## 评测体系

### 评测集

`evalsets/quick_v1.jsonl`：50 条（44 有答案 + 6 无答案），单跳题，easy 20 / medium 26 / hard 4。

标注刻意避开自己系统的排序：先用**内容特征词**在原文定位答案位置（纯文本匹配，不经检索器），
再映射到评测块，最后用自己的话重写问题。加载后跑 4-gram 泄漏检测，query 照抄原文的比例
超过 0.4 就改写——实测 50 条里有 2 条超标，已重写。

```bash
python tasks.py check-suite --suite evalsets/quick_v1.jsonl   # gold 存在性 + 泄漏检测
```

### 指标

全部在**评测块粒度**算，且**先去重再取 top-k**：父子块下一个父块对应 4 个子块，
若不去重，命中 4 条子块会被记成"召回 4 条"，实际只召回 1 个块。这条处理在 `rag/eval/metrics.py`，
并有单测锁住（`test_recall_uses_dedup_then_cut`）。

指标：Recall@1/3/5/10、Hit@k、MRR@10、nDCG@10（分级增益 2/1/0）。

### 显著性检验

44 条样本上单组 Recall@5 的 95% CI 约 ±13pt，两个独立区间一比，10pt 以内的差异说不清。
所以报告里所有"某配置比某配置好多少"都用**成对 bootstrap**（2000 次重采样）给差值区间，
区间跨 0 就标"不显著"，不给点估计下结论。实现见 `scripts/report.py::paired_diff`。

### 跑评测

```bash
python tasks.py eval --suite evalsets/quick_v1.jsonl --specs quick   # 15 组，约 25 分钟（含重排）
python tasks.py report --out docs/results-quick-v1.md                # 汇总成对比表
```

产物：`results/{run_id}/{config.json, metrics.json, per_query.jsonl}`。
每个数字都能顺着 `run_id` → `config_hash` → 评测集 `sha1` 回溯，这是 §19 立的纪律。

---

## 延迟实测（真实语料，单机 CPU）

| 策略 | 模式 | P50 | P95 | 阶段拆分（P50） |
| --- | --- | --- | --- | --- |
| fixed512 (7143 块) | dense | 24.2 ms | 27.9 ms | embed 10.1 + dense 13.4 |
| fixed512 | sparse (BM25) | 22.9 ms | 28.1 ms | sparse 22.0 |
| fixed512 | hybrid_rrf | 47.5 ms | 54.2 ms | embed 11.2 + dense 13.8 + sparse 21.3 |
| parentchild (17269 块) | hybrid_rrf | 61.9 ms | 72.8 ms | embed 10.4 + dense 18.6 + sparse 31.8 |
| fixed512 | hybrid + rerank | 3834.7 ms | 3937.4 ms | **rerank 3790** |

读法：
- **query 编码固定约 10ms**，与语料规模无关；检索开销随块数线性涨（7.1k → 17.3k 块，dense 13.4 → 18.6ms）。
- Milvus Lite 是暴力检索（无 HNSW），所以这些是**上界**：换 standalone + HNSW 会更低。
- 不含重排的链路 P95 54.2ms，相对 250ms 预算余量近 200ms；但重排一开就爆到 3.9s。
  **250ms 含重排的预算在无 GPU 环境下不可达**（实测 fp32 约 150 GFLOPS，278M 模型每候选约 250ms）。

---

## 已知缺口

| 缺口 | 影响 | 计划 |
| --- | --- | --- |
| 评测集只有 50 条、单 gold、单人标注 | 效应量小于 ~13pt 测不出来；多跳能力未覆盖 | M3 扩到 200 条 + 双人复核 |
| S4 父块回溯未实现 | 命中子块后只返回子块文本，未回取父块喂给 LLM | M4 |
| S3 语义切分未实现 | 四策略对比缺一角 | M3 |
| 重排 CPU 延迟 3.8s | 不适合在线路径，只能离线/评估用 | GPU 或蒸馏（未验证） |
| 指标面板未做 | 目前只能看 CLI 与 markdown | M4 |
| LangGraph 编排未接 | 无 query 分析/路由/自反思 | M4 |

---

## 快速开始

```bash
pip install -r requirements.txt

# 0. 抓语料（约 5 分钟，1171 页）
python scripts/fetch_corpus.py

# 1. 建索引（三个策略各一个 collection，CPU 约 24 分钟）
python tasks.py index-all

# 2. 看状态
python tasks.py stats

# 3. 命令行检索
python tasks.py search "Pod 的三种探针分别用来做什么" --mode hybrid_rrf --strategy fixed512

# 4. 起 API（两种等价方式）
python tasks.py serve
# 或：python main.py --port 8000      # 单进程，适合 IDE 调试
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Pod 删不掉怎么办","mode":"hybrid_rrf","top_k":5,"rerank":true}'

# 5. 评测
python tasks.py check-suite --suite evalsets/quick_v1.jsonl
python tasks.py eval --suite evalsets/quick_v1.jsonl --specs quick
python tasks.py report

# 6. 单测
python tasks.py test
```

> Windows 上没有 make，所以任务入口是 `tasks.py`（跨平台，无额外依赖）。

### 在 PyCharm 里运行

项目是 src 布局（`src/rag/...`），`rag` 包不在默认 import 路径上。`main.py` 在导入前
把 `src` 插进 `sys.path`，所以**不需要**手动标记 Sources Root，也不需要设 PYTHONPATH：

1. 右键 `main.py` → **Run 'main'**，或直接点右上角运行配置 **RAG API**（配置在 `.run/RAG API.run.xml`，随仓库共享）。
2. 打开 <http://127.0.0.1:8000/docs> 可交互调接口。
3. 调试断点请**不要**勾 `--reload`：reload 模式下 uvicorn 会另起子进程重新 import，
   断点会失效。改代码热重启时再用 `python main.py --reload`。

```bash
python main.py                     # 127.0.0.1:8000
python main.py --port 8123         # 换端口（被占用时会直接提示，不抛堆栈）
python main.py --config configs/default.yaml --log-level debug
python main.py --reload            # 只监听 src/ 的改动
```

首次 `/v1/search` 要加载嵌入模型并预热（实测约 13–45s），这期间的请求会看起来"卡住"；
`/healthz` 不受影响，`/readyz` 才反映索引与模型是否真就绪。

---

## 环境适配说明

| 项 | 情况 | 处理 |
| --- | --- | --- |
| Docker | 本机未安装、WSL 被安全策略拦截 | 用 **Milvus Lite**（本地文件库，无需 Docker） |
| 稠密/稀疏 | 已验证 Milvus Lite 3.2.2 同时支持 `FLOAT_VECTOR` 与 `SPARSE_FLOAT_VECTOR` 检索 | 本地与线上同一套代码，只换 URI：`data/milvus/rag.db` ↔ `http://milvus:19530` |
| GPU | 无 | 嵌入与重排走 CPU；重排 `bge-reranker-base` 降级并限制候选池 |
| 嵌入模型 | **bge-small-zh-v1.5**（95MB / 512 维）而非 bge-m3（2.2GB / 1024 维） | CPU 上够快、索引内存省一半；换 m3 只改配置 + `--rebuild` |
| 模型下载 | huggingface.co 直连超时，走 `hf-mirror.com`；模型缓存后仍会向 Hub 解析 revision，网络抖动会让服务起不来 | `rag/hfutil.py` 优先用本地快照（`local_files_only=True`），首次联网、之后完全离线可启动 |

---

## 关键设计决策

**1. 评测粒度必须与切分策略无关**
如果按各策略自己的 chunk 算 Recall，父子块策略会因"相关内容被切成 4 块、命中 1 块即算命中"
而系统性虚高。所以每篇文档额外铺一层固定 1024-token 网格（`eval_block_id`），
所有 chunk 通过字符区间最大重叠映射到网格，Recall 一律在网格粒度统计。

**2. token 预算用真实 tokenizer 校准，不是拍脑袋**
早期规则「中文 1 token + ASCII 单词 1.3 token + 其余 0」把 ASCII 低估 30%、把标点计 0，
导致纯标点表格行在窗口累加时"免费"，实测出现过 17153 字符的评测块——块一大，
各种切分的 chunk 都能映射到它，Recall 虚高且失去区分度。现用 300 个语料窗口对
bge tokenizer 做最小二乘拟合：`0.968×中文 + 0.220×ASCII字符 + 1.397×其他`，
R²=0.98、总偏差 -0.3%，块 token 数从"无上界"收敛到 P50=1006 / max=1036。

**3. 一个策略一个 collection**
不同切分策略共享同一 collection 时，HNSW 图与稀疏倒排表会被混在一起，索引参数互相干扰。
代价是内存 ×N，本量级可接受。

**4. BM25 自己算而不是用 Milvus 内置 Function**
IDF 依赖全库统计量，自己算才能在面板上解释"为什么这个词被降权"，中文切词行为也可控。
核心技巧：doc 侧存 `idf × tf_norm`、query 侧存 `1.0`，内积正好等于 BM25 打分；
若 query 侧也乘 idf 就变成 idf²，高频词被过度惩罚。

**5. 关闭重排时也走同一条代码路径**
`IdentityReranker` 存在的意义不是占位，而是让 `rerank=off` 这一列和 `rerank=on`
共用同一段编排代码，避免两套分支的实现差异污染对比。

---

## 目录结构

```
RAG/
├─ main.py                     启动入口（PyCharm 右键可跑；自动把 src 加入 sys.path）
├─ .run/RAG API.run.xml        PyCharm 共享运行配置
├─ tasks.py                    跨平台任务入口（CI / 命令行）
├─ docs/PRD.md                 需求文档（v0.3，含实现阶段修正记录）
├─ docs/results-quick-v1.md    第一轮评测结果与负结果分析
├─ docs/corpus.md              语料来源与取舍（含 Milvus 英文正文的证据）
├─ configs/default.yaml        全局配置（实验配置从此复制）
├─ evalsets/quick_v1.jsonl     50 条评测集
├─ evalsets/README.md          标注方法、局限、扩到 200 条的规范
├─ results/{run_id}/           每次评测的 config/metrics/per_query
├─ data/corpus/                抓取的中文文档 1171 篇（不入库，脚本重建）
├─ scripts/
│  ├─ fetch_corpus.py          语料抓取
│  ├─ probe_block.py           按内容定位评测块（标注用，不经检索器）
│  └─ report.py                汇总 results/ 出对比表 + 成对 bootstrap
├─ src/rag/
│  ├─ textutil.py              token 估算（已校准）/ 切分 / 累加窗口
│  ├─ hfutil.py                本地快照优先的模型加载
│  ├─ ingest/{loader,chunker,blocks,pipeline}.py
│  ├─ index/{bm25,store}.py
│  ├─ embed/                   哈希占位 / ST 后端
│  ├─ rerank/                  cross-encoder + identity（保证同路径对比）
│  ├─ retrieve/search.py       RRF / 加权融合 / 父块去重 / 重排编排
│  ├─ eval/{metrics,dataset,runner,corpus_index}.py
│  ├─ api/main.py              FastAPI
│  ├─ service.py               装配 + 三级预热
│  └─ cli.py                   命令行
├─ tests/                      37 个单测 + fixtures/corpus 8 篇测试语料
└─ tasks.py                    跨平台任务入口
```

---

## 简历表述（按实测更新）

| 原表述 | 实测 | 结论 |
| --- | --- | --- |
| Recall@5 从 61% 优化至 84% | 39.8% → **70.5%** | 按实测写：**「Recall@5 从 40% 提升到 71%」** |

61/84 是 PRD v0.1 的目标值，从未被实测覆盖。70.5% 对应
`results/20261115-110545__fixed512__hybrid_rrf__rr`，配置哈希 `9bf34d0270dd`。
按 §19 纪律：任何数字若不能在 `results/` 下找到 run 与配置，就不写进简历。

> 也要准备回答"为什么不是 84%"：50 条单跳题、单 gold、单人标注，且 gold 是 1024-token 块——
> 分母更严。扩到 200 条并允许多 gold 后数字会变，届时再更新。
