# 独立交付文件清单

本目录是原项目 `source/ai_agent/` 的独立副本，不含桌面端。57 个原始文件按白名单提取，保留字节内容和内部相对路径，详见 [SOURCE_INVENTORY.csv](SOURCE_INVENTORY.csv)。新编写的交付文档、配置和合成示例不在该来源清单中。

后续变更：按作者要求仅对 RAG 量规的 `source_file` 做人名中性化，当前 56 个源文件仍与初次提取哈希相同，1 个文件存在这一项预期差异。SOURCE_INVENTORY 保留初次提取记录，不重写旧哈希掩盖变更；详见 VERIFICATION 第 10 节。

## 1. 运行代码

| 文件 | 用途 | 必需程度 |
| --- | --- | --- |
| `api_server.py` | HTTP、SSE、上传、静态文件服务 | HTTP 服务必需 |
| `agent_core.py` | 资源加载、路由、模型调用、教学编排、报告检查 | 必需 |
| `quiz_schema.py` | 题目结构校验 | 必需，核心直接导入 |
| `standard_qa.py` | 标准问答加载与严格别名匹配 | 必需，核心直接导入 |
| `image_assets.py` | 图片 manifest 读取、安全校验和选图 | 必需，关闭图片功能也不可删除模块 |
| `document_loader.py` | PDF 文字与图像预处理 | HTTP 服务直接导入 |
| `vision_client.py` | 视觉模型请求与流式输出 | HTTP 服务直接导入，功能默认关闭 |
| `rag_config.py` | 可选 RAG 配置 | 必需，关闭 RAG 也不可删除模块 |
| `rag_ingest.py` | 入库命令与索引 manifest 校验 | 必需，检索模块导入其校验函数 |
| `rag_retriever.py` | 检索、可靠性判断与降级 | 必需，核心直接导入 |
| `requirements.txt` | 原后端完整依赖声明 | 必需；包括可选 RAG 的依赖 |
| `agent_demo/agent.py` | 交互式文本问答与报告检查 | 可选 CLI；不依赖 HTTP 服务 |

运行代码均相对于自身路径加载资源。CLI 与测试以自身父目录加入导入路径，不需要原仓库根目录、`app/` 或 `source/data_process/`。

## 2. 运行资源

| 文件或目录 | 用途 | 交付要求 |
| --- | --- | --- |
| `docs/04_实验知识库初版.md` | 稳定基础知识库 | 必须保留文件名及路径 |
| `prompts/*.txt`，共 16 个 | 系统边界、阶段、教学策略、测验、报告、视觉提示词 | 完整保留；不要因功能关闭而任意删文件 |
| `assets/standard_qa.json` | 固定问答，5 组答案、101 个别名 | 完整保留；测试校验既有 SHA-256 |
| `assets/image_manifest.json` | 教学图片描述与匹配规则 | 与图片文件保持一致 |
| `assets/images/michelson_optical_path.jpg` | 光路图 | 作者确认本人原创；公开署名 Seasoner，暂不添加许可证 |
| `assets/images/compensation_plate.jpg` | 补偿板示意图 | 同上 |
| `assets/images/equal_inclination_rings.jpg` | 等倾圆环原理图 | 同上 |
| `assets/images/fringe_in_out.gif` | 条纹吞吐动态图 | 同上 |
| `rag_docs/guidance/` | 3 篇指导资料 | RAG 可选，交付副本保留以便构建和测试 |
| `rag_docs/faq/` | 1 篇常见问题资料 | 同上 |
| `rag_docs/grading/` | 1 篇报告检查量规 | 同上 |
| `rag_docs/examples/.gitkeep` | 空案例目录占位 | 无真实学生案例 |

RAG 的 4 个分类目录均保留原有 `.gitkeep`。图片与知识文件的存在是运行检查结果，不等于已经完成版权授权审查。

作者后续已明确确认上述四张图片与全部知识文本为本人原创；该声明已记入 PUBLICATION_AUDIT，不由文件存在与否推断权属。

## 3. 测试与维护材料

| 文件或目录 | 用途 |
| --- | --- |
| `test_cases/test_*.py`，共 12 个文件 | 原有测试源码；其中 `test_vision_smoke.py` 为真实服务手工脚本，其余由 unittest 发现执行 |
| `docs/16_标准问答内容准备规范.md` | 固定问答内容维护规范，原样保留 |
| `README.md` | 面向招聘方和初次运行者，包含个人贡献、架构及启动方式 |
| `AGENTS.md` | 面向编码助手的独立仓库约束 |
| `PROJECT_NOTES.md` | 当前实现、接口细节、接手顺序和限制 |
| `FILE_MANIFEST.md` | 本文件，用于理解文件用途与提取边界 |
| `SOURCE_INVENTORY.csv` | 来源路径、大小、SHA-256，可核对 57 个源文件是否原样提取 |
| `VERIFICATION.md` | 本次验证结果与未验证项 |
| `PUBLICATION_CHECKLIST.md` | 公开前的授权、安全和展示材料检查 |
| `PUBLICATION_AUDIT.md` | 候选文件技术审核结果、作者公开确认与素材/来源信息待确认项；不含真实密钥 |
| `.gitignore`、`.env.example` | 安全忽略规则与空密钥配置模板 |
| `requirements-dev.txt` | 明确补充测试使用的 httpx 依赖 |
| `examples/*.json` | 3 个新写的合成 HTTP 请求，无真实模型结果与学生数据 |
| `showcase/README.md`、`showcase/cases/` | 三个案例说明，关联个人实现、观察结果与已知问题 |
| `showcase/inputs/`、`showcase/responses/` | 从指定合成演示白名单复制的 3 个输入和 3 个真实响应，不含私密配置或学生报告 |
| `showcase/results/` | 问答与报告可读快照、解析后的题目 JSON；报告保留 mask 风险提示，原始正文不修改 |
| `showcase/evidence/` | 单次请求汇总与 SSE 增量一致性检查，不含完整运行日志 |

## 4. 明确排除的内容

- 原项目 `.git/`：不复制原提交历史；独立目录自行初始化 Git 并配置作者指定的 GitHub 远程，不在原项目内创建嵌套仓库。
- 所有真实 `.env`、旧 `config.example.env`：不读取真实配置；重新编写空密钥模板。
- `outputs/`、`rag_index/`、虚拟环境、Python 缓存：用户输入、输出或可再生成数据。
- 唯一经单独审阅的输出交付为 `showcase/` 中的三组合成案例；这不代表可以整体提交 `outputs/`，也不包含真实学生报告或配置。
- `test_cases/regression_*/`：历史模型输出，不属于单元测试依赖。
- `test_cases/questions.json`、`report_samples.json`、`evaluation_template.csv`、`manual_test_template.md`：不作为本次公开数据集交付，自动化测试不依赖它们。
- `test_cases/run_regression_10.py`：历史在线回归脚本，本次用合成请求与现有 unittest 作为入口。
- `temp_standardQA/`、`assets/images.1/`：临时整理目录与旧图片副本，不属于当前运行资产。
- 原 `README.md`、`AGENTS.md`、`PROJECT_NOTES.md`、CLI 旧运行说明及其余历史设计文档：包含旧路径或过时说明，不原样作为独立仓库文档；新说明按代码核对。
- 桌面程序、相机代码、算法模型、个人信息与实验项目数据：不属于 Agent 独立运行边界。

## 5. 后续同步

原仓库更新不会自动同步到本副本。后续逐项比较源文件再决定同步，不要把整个旧目录覆盖回来，否则可能重新引入 `.env`、运行输出和旧文档。业务代码变化后，原提取清单只作为历史快照，不再代表新版本文件哈希。
