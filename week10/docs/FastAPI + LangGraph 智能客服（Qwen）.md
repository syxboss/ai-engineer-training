## 概览
- 基于 FastAPI 与 LangGraph，按“意图 → 检索/工具/人工”进行路由；大模型使用 `ChatTongyi(qwen-turbo)`。
- 参考现有模式：向量检索（FAISS）、DashScope 嵌入、结构化路由与检查点保存；保持与 tests 目录一致的库选择与调用方式。
- 当检索无结果时，将用户问题记录到 SQLite 并切换人工客服；订单查询通过数据库工具执行。

## 代码结构
- `work/app.py`：FastAPI 入口与路由（/chat、/health），启动与中间件（请求 ID、日志）。
- `work/config.py`：环境变量与默认配置（模型、FAISS 索引目录、DB 路径、检查点类型、人工客服接口 URL）。
- `work/statee.py`：LangGraph 状态类型（TypedDict），含 query、intent、kb_answer、order_summary、human_handoff、sources、route。
- `work/prompts.py`：意图识别、检索作答、订单说明等提示词。
- `work/tools.py`：
  - `retrieve_kb(query)`：FAISS 检索，返回内容与原始文档；参考 `tests/web.py:24-33`。
  - `getdb(order_text)`：订单查询（安全 SQL + 参数）；参考 `tests/router.py:213-218`。
  - `exec_sql(sql, params)`：SQLite 执行；参考 `tests/router.py:221-244`。
  - `record_unanswered(query)`：写入 SQLite 表 `unanswered_questions`。
  - `handoff_to_human(payload)`：对接默认人工客服接口（可返回占位说明或转发 HTTP）。
- `work/graph.py`：构建 LangGraph 流程图、节点与条件边，接入检查点与日志。
- `work/config.py` 中统一依赖 `DASHSCOPE_API_KEY`、`KB_INDEX_DIR`、`ORDERS_DB_PATH`、`CHECKPOINT_DB_PATH`、`HUMAN_SUPPORT_URL`。

## LangGraph 设计
- 检查点：默认 `InMemorySaver()`；若配置 `CHECKPOINT_DB_PATH`，启用 `SqliteSaver`（若库存在）。参考 `tests/web.py:45-50` 与 `tests/model-qwen.py:33-37` 的用法。
- 状态（`State`，TypedDict）：
  - `query` 输入文本；`intent`（course | presale | postsale | order | human | direct）；
  - `kb_answer`、`sources`（RAG）；`order_summary`（订单）；`human_handoff`（人工接口结果）；`route`（最终路由）。
- 节点：
  - `intent_node`：规则 + LLM 识别（结构化输出），映射到五类：课程咨询/售前/售后 → RAG；订单查询 → DB；人工 → 人工客服；其他 → direct。
    - 参考结构化输出用法：`tests/router.py:21-25`。
  - `kb_node`：调用 `retrieve_kb`（k=2），若有结果，构造 `kb_answer`（严格依据 Content）；参考 `tests/router.py:103-122`。
  - `no_kb_then_handoff_node`：无检索结果时写库 `record_unanswered` 并调用 `handoff_to_human`。
  - `order_node`：解析订单号并调用 `getdb` + `exec_sql`，将结果转自然语言；参考 `tests/router.py:263-272`。
  - `direct_node`：对非检索/非订单给出简要答复；参考 `tests/router.py:274-283`。
- 边：
  - `START → intent_node`。
  - `intent_node` 条件分支：
    - `course|presale|postsale → kb_node` → 若空 → `no_kb_then_handoff_node`，否则 → `END`。
    - `order → order_node → END`。
    - `human → no_kb_then_handoff_node → END`。
    - 其他 → `direct_node → END`。

## FastAPI 接口
- `POST /chat`：
  - 请求：`{ query: str, user_id?: str, thread_id?: str }`。
  - 执行：构建消息上下文（含 `thread_id`）→ `graph.ainvoke` 或 `astream`。
  - 返回：`{ route, answer, sources?, order?, human? }`；`route` 为最终路由，`answer` 为 `kb_answer`/`order_summary`/`human_handoff`/`direct_answer`。
- `GET /health`：返回服务健康状态与依赖（模型/索引/DB）。

## 提示词与意图映射
- 意图识别：仅输出之一 `course/presale/postsale/order/human/direct`，优先命中关键词（课程/售前/售后/订单/支付/退款/人工/客服/转人工），否则 LLM 判断。
- 检索作答：
  - 只依据“参考资料”的 `Content` 字段；语气自然但不改变含义；与 `tests/web.py:35-43` 一致。
- 订单说明：将状态、金额、更新时间、时间线转为自然语言；参考 `tests/router.py:246-261`。

## 检索与数据库
- FAISS：默认从 `KB_INDEX_DIR` 加载（缺省 `../tests/faiss_index`），DashScope 嵌入 `text-embedding-v4`；参考 `tests/rag-train.py:21-24` 与 `tests/web.py:20-22`。
- SQLite：
  - `ORDERS_DB_PATH` 指向订单库；安全 SQL 使用参数化（`%s` → `?`）；参考 `tests/router.py:221-244`。
  - 额外建表 `unanswered_questions(id, user_id, text, created_at)`，在 `record_unanswered` 中按需初始化。

## 日志与可观测性
- 使用 `logging`，统一格式：`%(asctime)s | %(levelname)s | %(name)s | req=%(request_id)s | %(message)s`。
- 在 FastAPI 中通过中间件注入 `request_id`（UUID）并写入上下文；关键节点与工具均记录输入/输出与耗时。
- 对接检查点的 `thread_id` 与日志的 `request_id` 保持一致，便于关联排障。

## 安全与健壮性
- 所有数据库调用参数化；拒绝长语句或非预期表访问；捕获异常写入告警日志并返回友好提示。
- 外部人工客服接口（如 `HUMAN_SUPPORT_URL`）网络错误时回退为默认占位响应。
- 对输入做清洗（空白、零宽字符、合并空格）；参考 `tests/router.py:50-55`。

## 验证与运行
- 依赖：设置 `DASHSCOPE_API_KEY`；默认使用 `../tests/faiss_index`；可选 `ORDERS_DB_PATH` 与 `CHECKPOINT_DB_PATH`。
- 运行：`uvicorn work.app:app --reload`。
- 验证：
  - 课程/售前/售后类问题 → 命中检索并返回来源；无结果 → 记录 SQLite 并返回人工转接。
  - 带订单号或关键词的问题 → 返回订单状态描述（若无 DB 则使用 mock）。
  - 明确“人工/转人工” → 直接走人工节点。

## 参考与一致性
- 检查点与检索工具风格：`tests/web.py:45-50`、`tests/rag-ask.py:45-50`、`tests/model-qwen.py:33-37`。
- RAG 语气与来源拼接：`tests/web.py:24-43`、`tests/router.py:103-122`。
- 订单查询/参数化 SQL/自然语言转换：`tests/router.py:197-272`。