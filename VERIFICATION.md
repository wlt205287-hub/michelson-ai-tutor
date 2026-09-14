# 独立副本验证记录

日期：2026-09-14。此记录针对提取快照，不是生产发布认证，也不是模型准确率评测。

后续进展：第 7 节补齐全新环境安装与真实 HTTP 启动验收，第 8 节记录三个真实模型示例，第 9 节记录静态展示材料整理，第 10 节记录作者确认与来源元数据中性化。第 1 至 6 节保留初次提取的历史记录。

## 1. 来源与变更范围

- 原项目 HEAD：`39d1a5354c71a5e750be1136aa8d70e4a4809adc`。
- 从工作区白名单复制 57 个文件，全部记录在 `SOURCE_INVENTORY.csv`。
- 复制后、测试后分别比对原文件及副本 SHA-256，57 个文件均一致。
- 未修改原项目已跟踪文件；未复制 `.git`、真实 `.env`、历史输出、私人报告、虚拟环境或索引。
- 仅新增独立交付文档、空密钥模板、忽略规则、测试依赖声明与合成请求；未重构业务代码或修改 Prompt。

## 2. 本地验证环境

| 项目 | 实际版本 |
| --- | --- |
| Python | 3.12.4，Windows，本机已有 Anaconda Python |
| FastAPI | 0.136.3 |
| Pydantic | 2.13.3 |
| OpenAI SDK | 2.30.0 |
| python-dotenv | 0.21.0 |
| pypdf | 6.12.2 |
| python-multipart | 0.0.29 |
| pdf2image | 1.17.0 |
| Pillow | 10.3.0 |
| httpx | 0.27.0 |

本机用于测试的环境没有 Uvicorn、ChromaDB 和 sentence-transformers；未为本次整理联网安装依赖。上述列表只记录已验证环境，不是推荐版本或 lock 文件。

## 3. 自动化测试

发现方式等同于：

```powershell
python -m unittest discover -s test_cases -p "test_*.py"
```

本次实际通过 Python 包装该发现过程执行：移除进程中的两类 API Key，关闭默认视觉与 RAG，设置 Hugging Face 离线标志，并拦截非本机回环的 socket 连接。模型、检索和渲染行为仍使用原测试中的 mock；没有改动测试源码。

```text
Ran 194 tests in 3.026s
OK
```

初次验证对所有 socket.connect 的拦截影响了 Windows asyncio 事件循环的本机 socketpair，导致测试环境报错；仅放行 `127.0.0.1` 和 `::1` 后重跑通过。未通过修改业务代码来消除该环境错误。

覆盖范围包括 API/SSE、CLI、阶段与教学策略、标准答案、测验格式及修复、RAG 降级、教学图片、PDF 文本/图片和视觉客户端。`test_vision_smoke.py` 的真实在线流程未执行。

## 4. 进程内 HTTP 与资源检查

使用 FastAPI TestClient，不启动 TCP 服务，不使用真实模型：

- `/health` 返回预期 JSON。
- `/docs` 与 `/openapi.json` 返回 HTTP 200。
- `load_resources()` 从独立目录成功读取资源。
- 标准问答命中返回资产中的完整固定答案；模型客户端构造设置为遇调用即失败，实际未触发；保存函数被 mock。
- 4 个静态教学图片路由均返回 HTTP 200，Pillow 可验证图片格式。
- 3 个合成 JSON 请求均通过当前 Pydantic 请求结构校验。

本机 Starlette 对现有 httpx TestClient 发出弃用警告，但上述调用及 unittest 均通过。本次未升级依赖或更改客户端；后续锁定依赖时应复核兼容性。

## 5. 文件安全检查

- 白名单复制确保没有带入真实环境配置、历史报告和输出。
- 对导出文本扫描长格式 `sk-`、GitHub token 和 PEM 私钥头，未发现匹配；该扫描不是完整密钥审计，不能证明所有形式的秘密均不存在。
- 测试在独立目录生成的 `outputs/` 已清理；没有删除原项目运行数据。
- 公开资产的授权仍未核实，见 PUBLICATION_CHECKLIST。

## 6. 未验证事项

- 未在全新虚拟环境执行完整 `pip install -r requirements.txt`，未验证所有间接依赖组合。
- 未启动真实 Uvicorn 进程。TestClient 验证的是 ASGI 应用，不代表 TCP、代理或公网部署验收。
- 未调用真实文本或视觉模型，未验证账户权限、模型可用性、费用、响应时延和回答准确性。
- 未下载 embedding、构建真实 ChromaDB 索引，未验证本机 Poppler 扫描页渲染。
- 未修复 mask 遗留要求，未验证教学效果或正式评分可靠性。
- 未初始化 Git、创建 GitHub 仓库、提交或上传任何文件。

结论：独立副本的资源、核心导入、现有 mock 回归和进程内接口已验证。完整安装、真实服务运行、模型效果及公开授权仍需按发布清单确认。

## 7. 后续第一步：全新环境与真实服务验收

日期：2026-09-14。在独立目录创建全新 `.venv`，未复用系统 site-packages，未修改原团队项目。从 `.env.example` 生成本地 `.env`，未读取或记录真实密钥。

执行 `.venv/Scripts/python.exe -m pip install -r requirements-dev.txt --index-url https://pypi.org/simple` 成功。随后 `pip check` 输出 `No broken requirements found.`。

关键版本：Python 3.12.4、FastAPI 0.141.1、Starlette 1.6.0、Pydantic 2.13.5、OpenAI SDK 3.13.0、Uvicorn 0.52.4、python-dotenv 1.2.3、pypdf 6.18.1、Pillow 12.3.0、httpx 0.28.1、httpx2 2.12.0、ChromaDB 1.5.9、sentence-transformers 6.0.1、PyTorch 2.14.0、transformers 5.17.0。

在新环境中，禁用 dotenv 读取、清除进程内 API Key、阻止外网连接，再次执行全部 unittest：`Ran 194 tests in 3.188s`，结果 `OK`。

真实 Uvicorn 使用独立虚拟环境启动，仅监听 `127.0.0.1:8000`。实际 HTTP 验证：

- `GET /health` 返回 `status=ok`、`service=ai_agent`。
- `GET /docs` 返回 HTTP 200。
- `GET /openapi.json` 正常解析，包含 10 个路径定义。

服务日志位于已忽略的 `outputs/server.stdout.log` 与 `outputs/server.stderr.log`，日志和测试输出不进入公开交付。

截至本节记录时，真实模型请求尚未执行，等待作者确认本地配置完成。RAG 包已安装，尚未下载 embedding 或构建真实索引；视觉模型与 Poppler 渲染仍未验收。未修改业务代码、Prompt、基础知识库或原依赖声明，未初始化 Git 或上传。

## 8. 三个真实模型示例

2026-09-14，作者确认本地配置完成后，重启独立后端加载配置，依次发送 `examples/` 中三个合成请求。未读取或展示密钥，未发送真实学生数据。

| 场景 | HTTP | success | 单次请求总耗时 | 内容核对 |
| --- | --- | --- | --- | --- |
| 引导式问答 | 200 | true | 27.91 秒 | 返回相位与光程差公式，继续以问题引导学生推导往返光程 |
| 预习出题 SSE | 200 | true | 68.49 秒 | meta、delta、done 完整；QuizEnvelope 再校验通过，3 道选择题和 2 道简答题 |
| 报告检查 SSE | 200 | true | 84.23 秒 | meta、delta、done 完整；返回问题、原因、建议、建议分与教师复查说明 |

耗时是三个不同请求各自的一次观测，不是平均延迟、首字延迟或性能基准。三个本地 API 请求不等于恰好三次上游模型调用，内部可能有分类或格式修复调用。

完整合成响应、SSE 事件和汇总存于 `outputs/model_demo_20260914_151447/`，另生成便于阅读的 `chat.answer.md`、`report.answer.md`、`quiz.quiz.json`。这些仍位于忽略目录，没有自动加入公开展示资料。

报告检查明确复现了 mask 遗留问题：输出要求说明 mask 是否经过平滑、阈值和形态学处理，与原 data_process 主流程已停用 mask 不一致。因此本次确认的是配置、真实模型调用、响应和测验结构正常，不代表所有教学判断正确，也未修复该知识冲突。合成报告的建议分不能作为评分准确率证据。

题目二次校验首次受到 Windows 管道默认编码影响，显式指定 UTF-8 后通过；未重新调用模型或修改 Schema 来绕过问题。视觉与真实 RAG 索引验收仍不在本次三个文本示例范围内。

## 9. 静态展示材料整理

2026-09-14，按作者要求暂不修复 mask，将第 8 节的合成输入与真实输出按白名单整理至 [showcase/README.md](showcase/README.md)。共 15 个展示文件，包括总览、3 个案例说明、3 个请求、3 个原始响应、2 个可读回答、1 个解析后的题目 JSON，以及请求汇总和 SSE 检查结果。

同步更新主 README 的案例入口、FILE_MANIFEST 文件用途和 PUBLICATION_CHECKLIST 已完成项。本轮没有调用模型、修改业务代码或重新运行全量业务测试；既有 194 项测试记录见第 7 节。

本轮专项验证：

- 3 个展示输入与原 examples 文件逐字节一致，3 个原始响应与本地本轮记录逐字节一致。
- 2 个可读 Markdown 的模型正文与响应中的 answer_markdown 完全一致，仅标题、风险提示和起止标记属于新增说明。
- 所有展示 JSON 均可解析，展示题目通过原 QuizEnvelope 校验，数量为 3 道选择题和 2 道简答题。
- 出题 SSE 为 3 个事件，报告 SSE 为 1380 个事件；两者增量拼接和 done 均与最终响应一致，error 事件数为 0。公开证据只保留统计，不复制完整运行日志。
- 改动文档的本地链接均有效，展示文件未发现已检查的长格式 token、私钥、绝对本地路径或邮箱模式；这不是全面隐私或版权审计。
- Git 只读忽略检查确认展示文件未被当前规则忽略，.env、outputs 与 .venv 仍被忽略；没有初始化 Git 或提交。
- 57 个提取源文件哈希保持不变，原始 Prompt、知识库和标准答案未修改。

内容风险如实保留：报告仍有 mask 遗留要求；题目 q3 的两个干扰项数学含义相同但字符串不同。Schema 通过不等于语义质量通过。材料未改写为理想答案，也未据单次观测声称准确率、教学效果或平均性能。

## 10. 作者确认与来源元数据中性化

2026-09-14，作者明确确认四张教学图片、基础知识库、标准问答与 RAG 文档均为本人原创，而不只是整理或改写他人材料。作者要求来源文件名不含人名，指定公开署名为 Seasoner，选择暂不添加许可证。相关状态已同步至审核记录与 README，未创建 LICENSE。

唯一的提取文件变更位于 `rag_docs/grading/2026_表面形貌实验报告检查量规.md` 的 `source_file`：来源标签现为 `实验方案 - 副本.md; 主报告模板.docx`。知识正文、评分量规、mask 要求和 Python 业务代码都未改。原团队仓库未改，SOURCE_INVENTORY 保留初次提取的旧哈希用于追溯。

专项验证结果：

- 56 个提取文件仍与初次哈希一致，1 个文件只有上述预期的来源元数据差异。
- 排除 `source_file` 行并统一换行后，新旧文件全文一致；没有趁机修改教学内容。
- 离线执行 `test_rag_config`、`test_rag_ingest`、`test_rag_retriever`，12 项测试全部通过，耗时 0.035 秒；测试屏蔽 dotenv 读取和网络连接。
- 未重新调用真实模型，未初始化 Git、提交或上传。

如果曾基于旧文档构建 RAG 索引，后续使用前需按既有流程重建，使来源元数据同步；本次不下载模型或构建索引。

## 11. 首次 Git 提交前验证

2026-09-14，独立目录已初始化 Git，本地分支 `codex/initial-publication`，远程 `origin` 为 `https://github.com/wlt205287-hub/michelson-ai-tutor.git`。使用作者指定的 Seasoner 署名与 GitHub noreply 邮箱，仅设置仓库本地配置，不修改原团队仓库。

本轮重新执行全量 unittest，禁用 dotenv 读取、清除进程内 API Key、阻止非本机回环连接：`Ran 194 tests in 2.288s`，结果 `OK`。没有调用真实模型、下载 embedding 或修改业务代码。

实际暂存范围为 86 个文件，包含源码、测试、教学资源与合成展示；真实 `.env`、`.venv/`、`outputs/`、`rag_index/` 和缓存均不在提交材料内。原始来源哈希记录不改写，仍保留来源元数据中性化这一预期差异。

提交前尝试读取远程引用时出现 `Could not resolve host: github.com`。本节不宣称首次提交或推送已经成功；本地提交与远程上传是两个独立步骤，最终结果须以实际 Git 输出核对。仓库可见性不由本地 Git 配置改变。
