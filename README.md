# astrbot_plugin_group_digest

这是一个基于 AstrBot 的**群聊理解增强与长期记忆插件**。  
本插件围绕群聊事实归档、话题切片、语义输入增强、增量更新与可复用记忆，形成了一条可持续运行的群聊理解链路。

## 项目定位

一句话定位：

- **按群隔离的群聊理解中间层**
- **带长期记忆能力的日报与主动发言引擎**

当前主线目标：

1. 稳定产出 `/group_digest`、`/group_digest_today`、`/group_digest_debug_today`。
2. 在 scheduler 场景下复用同一条分析链路输出 `suggested_bot_reply`。
3. 让语义分析不再只依赖“最近 N 条消息”，而是利用切片与历史上下文覆盖长时段信息。
4. 将可复用的群内知识（如黑话解释）沉淀成轻量长期记忆。

## 当前能力总览

### 1. 对外能力

1. `/group_digest`：生成昨日日报。
2. `/group_digest_today`：生成今日截至当前时刻日报。
3. `/group_digest_debug_today`：输出今日调试统计信息。
4. scheduler 定时主动发言：按群独立生成分析结果，并发送 `suggested_bot_reply`。

### 2. 群聊理解能力

1. **消息事实层**：按群按天 JSONL 归档，append-only。
2. **topic 生命周期层**：有效消息驱动，支持 `created/active/closed`。
3. **topic slice 层**：topic close 后落盘，形成可复用中间表示。
4. **语义输入组装层**：`retrieved slices + current-day slices + tail raw messages + optional slang contexts`。
5. **黑话学习层（轻量版）**：统计预筛 -> retrieval 取证 -> 条件 LLM 解释 -> 解释复用注入。
6. **缓存/增量层**：checkpoint + delta 增量更新 + 回退全量重算。

### 3. 长期记忆能力（当前已落地）

1. 原始消息长期归档：`messages/<group_id>/<YYYY-MM-DD>.jsonl`。
2. 话题切片长期归档：`topic_slices/<group_id>/<YYYY-MM-DD>.jsonl`。
3. 黑话解释长期归档：`slang/<group_id>/slang.jsonl`。
4. 向量记忆（可选）：semantic_unit 与 topic_slice/core embedding 持久化到 Qdrant。

## 架构主线

当前稳定主链路：

1. 消息采集与过滤
2. 事实归档（JSONL）
3. topic 生命周期推进与切片沉淀
4. 语义输入组装（slice-aware / retrieval-aware / slang-aware）
5. LLM 语义分析
6. 缓存命中 / 增量更新 / 全量重算
7. 命令返回或 scheduler 主动发送

这条主链路由以下核心模块协同完成：

- `main.py`：插件入口与依赖注入
- `services/storage.py`：消息事实存储
- `services/group_topic_segment_manager.py`：topic 生命周期
- `services/topic_slice_store.py`：slice 持久化
- `services/semantic_input_builder.py`：语义输入唯一主入口
- `services/digest_service.py`：日报编排核心
- `services/llm_analysis_service.py`：LLM 调用与结构化解析
- `services/report_cache_store.py`：checkpoint/cache 存储
- `services/incremental_update_service.py`：delta 增量辅助
- `services/scheduler_service.py`：定时主动发言

## Topic 与切片机制（当前实现）

### 1. 有效消息过滤

低信息消息不进入 topic 主流程，例如：

- 纯附和
- 纯语气
- 纯应答
- 短句且缺少实体/动作/时间信息

### 2. topic 状态机

1. 有效消息以序列方式推进。
2. 每两条有效消息构造一个 semantic unit。
3. topic core 创建后保持稳定，不做滚动漂移更新。
4. 长时间无有效消息后 close，并落盘为 slice。

### 3. transfer buffer

1. 若 embedding 可用，semantic unit 与 topic core 相似度过低时进入 transfer buffer。
2. buffer 达到阈值后触发 topic transfer，创建新 topic core。
3. 若 embedding 关闭或失败，自动降级为仅时间规则，不影响主链路。

### 4. lifecycle sweep / prune

1. 后台 sweep 在“无新消息”时也能推进 topic close。
2. 已 `closed + persisted` 的 runtime topic 可按保留时长 prune。
3. prune 只清理内存态，不影响已落盘 slices。

## 语义输入增强（SemanticInputBuilder）

`SemanticInputBuilder` 是语义输入唯一主入口。

当前 full-window / incremental 都通过它组装材料：

1. historical retrieved slices（Qdrant retrieval，可降级）
2. current-day closed slices
3. tail raw effective messages
4. related slang contexts（可配置）

关键点：

- 无 retrieval / 无 slices / 无 slang 命中时自动回退，不中断链路。
- 对 slices 与 slang contexts 都有固定长度硬兜底，防止 prompt 无上限膨胀。
- source 与组装元数据会写日志，便于排查“本次到底喂了什么语义材料”。

## 黑话学习（RAG-enhanced minimal）

这版黑话学习借鉴参考项目思路，但做了轻量适配：

1. **候选发现（低成本）**：`SlangCandidateMiner` 在 slices 上做统计预筛。
2. **历史取证（RAG）**：优先检索当前群近期 `topic_slice` 语境。
3. **条件解释（LLM）**：证据不足不解释，避免硬猜。
4. **解释复用（store）**：`SlangStore` 按群落盘并复用，减少重复推断。
5. **主链路注入**：由 `SemanticInputBuilder` 按相关性注入解释上下文。

明确不做：

- 复杂黑话生命周期治理
- 重型全局 hook pipeline
- 人格/好感度/情绪等 self-learning 体系

## 缓存与增量更新

同一天、同群、同模式下：

1. 无新有效消息：`cache_hit`
2. 少量新增：`incremental_update`
3. 条件不满足或增量失败：`full_rebuild`

缓存判定包含：

- 消息 checkpoint（count/last_ts/fingerprint）
- provider 与 prompt 签名
- 语义输入相关签名（含切片/检索/黑话注入后的上下文变化）

## 存储目录

默认在 AstrBot `data` 目录下：

- 消息事实层：`messages/<group_id>/<YYYY-MM-DD>.jsonl`
- topic slices：`topic_slices/<group_id>/<YYYY-MM-DD>.jsonl`
- slang 解释：`slang/<group_id>/slang.jsonl`
- 群 origin 映射：`group_origins.json`
- 报告缓存：`report_cache.json`
- 旧数据兼容回读：`messages.json`（只读，不再扩写）

## 关键配置（按能力分组）

### A. 日报与主动发言

- `use_llm_topic_analysis`
- `analysis_provider_id`
- `max_messages_for_analysis`
- `enable_scheduled_proactive_message`
- `scheduled_send_hour` / `scheduled_send_minute` / `scheduled_send_timezone`
- `scheduled_group_whitelist_enabled` / `scheduled_group_whitelist`

### B. topic 生命周期

- `new_topic_gap_seconds`
- `topic_close_gap_seconds`
- `single_message_topic_timeout_seconds`
- `transfer_similarity_threshold`
- `transfer_buffer_size`
- `enable_topic_lifecycle_sweep`
- `topic_lifecycle_sweep_interval_seconds`
- `topic_runtime_closed_prune_seconds`

### C. embedding 与向量存储

- `enable_topic_embedding`
- `embedding_api_key` / `embedding_model` / `embedding_base_url` / `embedding_timeout_seconds`
- `enable_qdrant_embedding_store`
- `qdrant_url` / `qdrant_api_key`
- `qdrant_semantic_unit_collection` / `qdrant_topic_slice_collection`
- `qdrant_vector_size` / `qdrant_distance_metric` / `qdrant_timeout_seconds`

### D. retrieval 语义增强

- `enable_topic_slice_retrieval`
- `topic_slice_retrieval_recent_days`
- `topic_slice_retrieval_limit`
- `topic_slice_retrieval_query_message_count`

### E. 黑话学习与注入

- `enable_slang_contexts`
- `enable_slang_learning`
- `slang_store_path`
- `slang_recent_days`
- `slang_injection_limit`
- `max_slang_context_chars`
- `slang_candidate_min_term_frequency`
- `slang_candidate_min_slice_coverage`
- `slang_candidate_max_candidates`
- `slang_candidate_current_day_boost`
- `slang_retrieval_recent_days`
- `slang_retrieval_limit`
- `slang_min_context_items_for_inference`
- `slang_max_inference_per_build`
- `slang_reinfer_min_evidence_increase`

## 降级策略

任何增强能力不可用时，主链路都应继续工作：

1. embedding 不可用：topic 逻辑退化为时间规则。
2. Qdrant 不可用：跳过向量持久化/检索，回退本地语义输入。
3. 黑话链路证据不足或 LLM 不可用：仅复用已有解释或不注入黑话。
4. LLM 分析失败：按 `fallback_to_stats_only` 决定降级为统计输出。

## 当前边界（明确不做）

1. 不引入完整 self-learning agent 架构。
2. 不引入 persona/affection/mood/goal-driven 对话系统。
3. 不引入重型 RAG 平台和复杂 rerank 流水线。
4. 不改变现有命令与 scheduler 对外语义。

## 测试状态

当前仓库测试覆盖包含：

- digest 主链路
- scheduler 主链路
- topic lifecycle/sweep/prune
- semantic input builder（slice/retrieval/scheduled query）
- embedding store（Qdrant no-op/degrade）
- slang pipeline（候选/解释/复用/注入）

当前执行：`pytest -q tests` 通过。

## 许可证

本项目采用 **GNU AGPL v3** 许可证。
