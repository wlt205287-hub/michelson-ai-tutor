# 案例 2：结构化预习出题

## 场景与输入

在 `pre_lab` 阶段明确请求“生成预习题”，侧重点为理论公式。接口为 `POST /api/agent/chat/stream`，完整请求见 [quiz.json](../inputs/quiz.json)。

> 请生成预习题，侧重点：理论公式。

## 实际输出与校验

- HTTP 200、`success=true`，返回 `route=quiz_generate`。
- 题目为 3 道选择题和 2 道简答题，包含服务器补充的 `quiz_id` 与 `stage=pre_lab`。
- 选择题含 4 个选项、从 0 开始的正确答案索引及解释；简答题含参考答案和 3 至 5 个评分要点。
- 从最终响应解析出的题目再次通过现有 `QuizEnvelope` 校验。
- 收到 `meta → delta → done`，其中 delta 仅有 1 个，包含完整校验后的 JSON 字符串。总耗时 68.49 秒。

完整结果见 [题目 JSON](../results/quiz.json)、[原始响应](../responses/quiz.json) 和 [流式检查证据](../evidence/stream-checks.json)。注意响应外层仍写 `display_type=markdown`，但 `answer_markdown` 内是 JSON，需要二次解析，不能直接当普通 Markdown 渲染。

## 示例题目

q2 问参考镜沿光轴移动距离 d 时的光程差变化，四个选项为 d、2d、d/2、4d；模型给出的正确答案索引为 1，即第二项 2d，并用往返光程解释。q4 要求推导 `f0=2v/λ`，附带参考答案与五个评分要点。

这些示例摘自完整结果，不是另写的标准答案。

## 对应个人实现与质量边界

- [quiz_schema.py](../../quiz_schema.py)：固定题型数量、字段类型、唯一 ID、答案范围等校验。
- [agent_core.py](../../agent_core.py)：出题意图门控、生成与最多一次格式修复、完成校验后输出。
- [quiz_prompt.txt](../../prompts/quiz_prompt.txt)：题目生成约束。

**结构通过不等于内容质量完全合格。** 原结果 q3 的第二项 `h = λ/(2π) Φ_rel` 与第四项 `h = λ Φ_rel / (2π)` 数学含义相同，但字符串不同，因此通过了当前字符串去重校验。这两项都是干扰项，不改变第一项为正确答案，但降低题目质量。这里保留原结果，不偷偷替换成更好的选项。

本次没有证据说明触发过修复重试；代码包含该机制不等于这个示例实际走过该分支。测验还返回了答案与评分要点，因此不应把本接口当作隐藏标准答案的正式考试服务。

[返回演示总览](../README.md)
