# AGENTS.md - Michelson AI Tutor

## 1. 范围

这是独立 Agent 后端仓库，源码位于根目录，不再使用 `source/ai_agent/` 前缀。作者独立负责后端；原实验平台的桌面端、采集和算法模块不在此仓库内。先阅读 README、PROJECT_NOTES、FILE_MANIFEST 和 VERIFICATION。

## 2. 开发约束

- Python 3.10+，沿用类型注解、pathlib、logging、FastAPI 与 unittest。
- 保留现有 HTTP 路由、请求字段、响应包络和 SSE `meta/delta/done/error` 协议，除非任务明确要求变更。
- 核心资源通过 `__file__` 相对定位；不要添加对原团队仓库或本机绝对路径的依赖。
- 不执行相机采集、FFT、相位展开、三维重建，不宣称已实现这些能力。
- 不随意修改基础知识库、标准答案或 Prompt。变更教学内容须有明确目标与相关回归测试。
- 题型以 `quiz_schema.py` 为准：固定 3 道选择题、2 道简答题，无填空题。格式错误最多修复一次，校验前不能把题目流式输出给客户端。
- 标准问答只做严格别名匹配，不改成模糊检索；标准答案资产存在哈希回归测试。
- RAG 与图片增强是可选补充，异常或不可用时应降级，不替代基础知识库和安全边界。
- 阶段与教学策略是内部决策，对外 `stage` 的兼容透传及视觉流式 `generation` 例外必须保留。不要新增隐式持久化状态。
- `quiz_context` 只用于报告检查 Prompt，不进入 RAG 查询、日志或原始报告输入存档；习题反馈不计入原 100 分量规。
- 原实验主流程已停用 mask；本仓库教学资料仍有残留要求，可能误报。不能把遗留要求解释为正式必需步骤。修复时需同步 Prompt、知识库与测试，不要悄悄扩大任务范围。

## 3. 安全与公开

- 不读取、输出或提交真实密钥、学生报告、聊天历史、图片 base64 和运行时索引。
- `.env.example` 只能包含空密钥或占位值。配置示例不代表某模型当前可用。
- 不全局忽略 JSON 或图片，避免漏交标准问答、manifest 和业务图片。
- `outputs/` 可能包含请求正文，本地输出不等于脱敏数据。
- 教学图片和资料须核实公开授权；未经确认不添加开源许可证。
- 服务默认只监听本机。没有鉴权和完整限流，不将其作为公网生产服务发布。

## 4. 验证与交付

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s test_cases -p "test_*.py"
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

默认测试不能调用真实模型或下载 embedding。`test_vision_smoke.py` 只供显式手工运行，不用 pytest 收集。SOURCE_INVENTORY 记录初次提取哈希，后续差异见 VERIFICATION；目前仅 RAG 量规来源标签按作者要求做了人名中性化。将来正常开发可以改代码，但要说明与提取快照的差异。完成任务后报告实际测试结果，不把 mock 通过说成真实模型验收通过。
