# Agent 后端维护说明

核对日期：2026-09-14。本文适用于独立副本，不沿用原项目目录前缀。个人贡献边界见 README，文件来源见 SOURCE_INVENTORY。

## 1. 接手顺序

1. 按 README 安装依赖，使用空白模板配置自己的模型服务。
2. 阅读 `api_server.py` 的请求类和路由，理解 JSON、multipart 与 SSE。
3. 阅读 `agent_core.py` 的 `load_resources`、`ask_tutor`、`stream_tutor_events`、`check_report`、`stream_report_events`。
4. 阅读 `standard_qa.py`、`quiz_schema.py`，区分固定回答、普通模型回答、测验 JSON。
5. 阅读 `prompts/` 和基础知识库，明确正式实验是连续扫描时间载波法。
6. 按需阅读 RAG、PDF、视觉和教学图片模块，再运行对应测试。

## 2. 运行资源与配置

- `agent_core.py` 使用当前根目录下的 `docs/04_实验知识库初版.md` 和 `prompts/`；这些是运行输入，不只是说明书。
- `rag_retriever.py`、`rag_ingest.py` 即使 RAG 关闭也被导入，因此不能删源码；ChromaDB 与 sentence-transformers 在真正使用时延迟导入。
- 配置加载顺序为进程环境、`agent_demo/.env`、根目录 `.env`；现有 `load_dotenv` 不覆盖已设值。建议只维护根目录 `.env`，避免旧 Demo 配置抢先生效。
- 文本变量为 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`；视觉变量为 `ZHIPU_API_KEY`、`ZHIPU_BASE_URL`、`ZHIPU_MODEL_NAME`，开关为 `ENABLE_VISION`。
- `AI_AGENT_RAG_ENABLED` 默认关闭；`AI_AGENT_IMAGES_ENABLED` 默认启用。根目录 `.env.example` 明确给出安全默认值，不含真实密钥。
- API 不提供关闭所有输出留存的请求字段；多个入口会保存到 `outputs/`。不要向公开演示服务发送私人报告。

## 3. 教学与输出约定

标准阶段为 `pre_lab`、`during_experiment`、`result_evaluation`、`unknown`。旧 `measurement` 映射到现场阶段；旧数据处理、计算、误差分析和报告阶段映射到结果评价。内部阶段解析依次使用显式阶段、规则、模型分类、未知回退；离题和社交有短路路径。

普通 tutor 先尝试标准问答的严格别名匹配。命中后跳过生成模型、RAG 和图片增强；显式引导策略在非安全问题上可绕过固定回答。未命中时根据规则和必要的分类选择 `direct`、`guided`、`hybrid`。三级引导不等于后端保存了长期学习档案，状态仍依赖客户端回传的文字历史。

只有显式 `stage=pre_lab` 且本轮包含出题意图时触发测验生成。最终生成 3 道选择题、2 道简答题，每题含答案或评分要点，服务器补充 `quiz_id`。客户端需解析 `answer_markdown` 内的 JSON，不应把它当普通 Markdown 展示。

报告检查返回 Markdown 诊断和建议分，不负责最终成绩。可附加 `quiz_context`，最大 32768 字符；习题反馈独立于原 100 分量规。相关上下文不进入 RAG 查询和原报告输入存档，但生成的反馈本身可能反映其内容，因此输出仍应视为敏感材料。

## 4. API 差异速查

| 能力 | 普通 JSON / 非流式 | 流式 |
| --- | --- | --- |
| 文本 tutor | 接收教学策略、引导等级 | 同样接收 |
| 视觉 tutor | multipart，不接收教学覆盖字段 | multipart，接收教学覆盖字段 |
| 文本 report-check | 接收 `quiz_context` | 接收 `quiz_context` |
| PDF upload | 不接收 `quiz_context` | 无同名流式路由 |
| 视觉 report-check | 不接收 `quiz_context` | 接收 `quiz_context` |

普通响应主要包含 `success`、`mode`、`stage`、`display_type`、`answer_markdown`、`saved`、`warning`，tutor 还含 `need_student_reply`。错误分支需同时检查 HTTP 状态和 `success` / `error_code`。

SSE 的 `delta.text` 供增量展示，`done.answer_markdown` 为完整最终内容。前置校验失败可能直接返回 JSON，不一定进入 SSE。内部教学策略不会新增到公开响应字段；普通文本返回的 `stage` 保持请求值，视觉流式 `meta.stage` 使用 `generation`。

视觉关闭时相关接口返回 HTTP 404、`VISION_DISABLED`。非流式单图上限 5 MB，流式单图上限 15 MB；非流式 PDF 10 MB、流式视觉 PDF 30 MB。非流式最多 8 张嵌入图或扫描页，流式最多 12 张。扫描页依赖 Poppler 栅格化，不是 OCR。

## 5. 当前风险与未实现事项

- **知识与算法不同步**：原 data_process 主流程 `skip_mask=True`，但部分教学文本仍要求 mask。报告可能因此要求不该必需的材料。本次只提取，不修正文案或算法。
- **部署边界**：没有身份认证、细粒度访问控制和完整限流；原 CORS 仅列出本地常用端口。`session_id` 不保证数据隔离。
- **证据边界**：RAG 来源只注入 Prompt，没有结构化来源响应；模型是否展示引用和图片不能保证。
- **作者声明与公开选择**：Seasoner 已确认独立负责后端、团队允许单独公开，四张教学图片与全部知识资料为本人原创。作者选择暂不添加许可证；不将此项目标注为 MIT 或其他已许可开源项目。
- **依赖与模型**：requirements 未锁死全部版本；模型名是否可用取决于账户，真实服务、embedding 下载与扫描 PDF 尚未随本次离线验证验收。
- **不在范围内**：Word、OCR、设备控制、FFT 执行、真实自动评分、管理后台及持久化会话。

## 6. 独立副本维护

这次没有改动任何提取的 Python、Prompt、基础知识库、标准答案或图片。README、AGENTS、本文及配置模板为独立目录重新整理，修正了旧路径与过时题型说明。原项目文档没有变化。

初次提取没有复制 `.git`、真实 `.env`、历史输出或索引。后续本机运行已创建私密 `.env`、虚拟环境和合成运行输出，它们仍被忽略，不属于提交材料。独立目录现已初始化全新 Git 历史，远程为 `https://github.com/wlt205287-hub/michelson-ai-tutor.git`，本地分支为 `codex/initial-publication`，计划推送至远程 `main`。提交署名为 Seasoner，使用作者提供的 GitHub noreply 邮箱。本记录更新时尚未上传成功，远程检查遇到域名解析失败；发布验证见 VERIFICATION 第 11 节。
