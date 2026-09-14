# 案例 1：引导式问答

## 场景与输入

学生想理解反射式干涉中相位与高度换算的 `4π` 因子。本例为合成提问，不包含学生身份或历史作业。

```json
{
  "session_id": "public-demo-chat",
  "stage": "pre_lab",
  "student_input": "为什么反射式干涉中高度变化与相位变化之间有 4π 因子？请引导我理解。",
  "conversation_history": [],
  "teaching_policy": "guided",
  "guidance_level": 1
}
```

接口：`POST /api/agent/chat`。完整输入见 [chat.json](../inputs/chat.json)。

## 实际输出与观察

模型先给出相位与光程差公式，再提出具体问题：

> 请你先写出：反射面高度变化 $h$ 时，对应光程差变化量 $\Delta L$ 等于多少？

响应的 `success=true`、`mode=tutor`、`stage=pre_lab`、`need_student_reply=true`。完整内容见 [回答 Markdown](../results/tutor.md) 与 [原始 JSON 响应](../responses/chat.json)。本次总耗时 27.91 秒。

## 对应个人实现

- [api_server.py](../../api_server.py)：接收本轮教学策略与引导等级，返回统一响应包络。
- [agent_core.py](../../agent_core.py)：教学策略解析与 Prompt 装配，将模型回答转为客户端可用结果。
- [teaching_guided_prompt.txt](../../prompts/teaching_guided_prompt.txt)：约束引导式回答的侧重点。

这个案例展示请求级策略覆盖确实影响了本轮回答，不证明自动分类在所有输入上正确。只录制了一轮，没有演示后续三级引导，也没有服务端长期学习档案。不能把 `need_student_reply` 当作学习效果指标。

[返回演示总览](../README.md)
