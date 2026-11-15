# 语料说明

## 来源

| 源 | 页面数 | 语言 | 抓取方式 | 内容特点 |
| --- | --- | --- | --- | --- |
| FastAPI 中文文档 | 153 | 中文 | `fastapi.tiangolo.com/zh/sitemap.xml` | 代码密集、参数说明、短段落 |
| Kubernetes 中文文档 | 1036 | 中文 | `kubernetes.io/zh-cn/sitemap.xml` | 概念解释、长段落、术语密集 |

两者都是官方文档、可公开引用、可复现抓取，且分属两种文档生成器（mkdocs-material / Docsy），
能顺带检验切分与抽取对不同 HTML 结构的鲁棒性。

复现：

```bash
python scripts/fetch_corpus.py              # 全量
python scripts/fetch_corpus.py --limit 10   # 试抓
```

脚本特性：sitemap 驱动（可复现）、断点续抓（已存在且 >200 字节则跳过）、
并发 4 + 每请求 0.2s 间隔 + 失败退避（不压对方服务器）、
`data/corpus/manifest.json` 记录 url → 文件 → 字符数。

## 为什么没有 Milvus 和 LangGraph

PRD 初稿设想"Milvus / FastAPI / LangGraph 官方文档"。实测结果：

- **Milvus**：`milvus.io/docs/zh/*` 路径存在，sitemap 里也有 504 条中文路径，但**正文是纯英文**。
  抽查 6 个页面，中文占比全部为 **0.0%**（`overview` / `schema` / `search` / `manage-collections` 等）。
  站点只翻译了导航壳，没翻译正文。
- **LangGraph**：没有中文版，且 `sitemap.xml` 返回 404。

用英文正文配中文查询，等于凭空给稠密检索加一层跨语言难度，指标会被语言差异污染，
无法归因到检索策略本身。所以剔除，改用同为中文的 Kubernetes 文档补足规模。

若后续要做跨语言检索实验（中文 query → 英文文档），Milvus 英文文档是合适的对照组，
但那属于独立实验，不与主实验混在一起。

## 目录约定

```
data/corpus/
├─ manifest.json          url → 文件 → 字符数
├─ fastapi/*.md           153 篇
└─ kubernetes/*.md        1036 篇
```

文件名由 URL 派生（去掉语言前缀与 `/docs/`），重名时靠路径前缀区分
（`fastapi/` 与 `kubernetes/` 各自独立目录），不会互相覆盖。

`doc_id` 用「相对路径的 sha1 前 12 位」，所以**改文件名不会破坏已标注的 gold block 对齐**，
只有移动目录才会。这一条在多轮标注中很重要。

## 与测试的关系

`data/corpus/` 是评测语料，不进版本库（`.gitignore` 已排除，靠脚本重建）。

单测与 API 集成测试用的是 `tests/fixtures/corpus/` 下 8 篇人工撰写的小文档——
它们体量小、术语可控，测试断言依赖具体内容（比如"BM25 检索必须命中 BM25 那篇"），
用真实语料反而会因内容变动而脆断。
