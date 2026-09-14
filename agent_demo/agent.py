from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from openai import OpenAIError

    from agent_core import (
        ask_tutor,
        build_frontend_payload,
        build_report_prompt,
        call_model,
        call_model_stream,
        check_report,
        create_client,
        load_resources,
        normalize_model_output,
        save_output,
        update_conversation_state,
    )
except ImportError as exc:
    print("依赖导入失败，请先运行：pip install -r requirements.txt")
    print(f"错误信息：{exc}")
    sys.exit(1)


def read_multiline_input() -> str:
    """读取多行报告片段，输入 END 单独成行结束。"""
    print("请输入报告片段。输入完成后，单独输入 END 并回车：")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def choose_mode() -> str:
    """读取用户选择的运行模式。"""
    print("请选择运行模式：")
    print("1：预习问答 / 操作答疑 / 引导式回答")
    print("2：报告检查")
    choice = input("请输入 1 或 2：").strip()
    if choice not in {"1", "2"}:
        raise ValueError("模式选择无效，请输入 1 或 2。")
    return choice


def _print_payload(payload: dict[str, Any], streamed: bool) -> None:
    """Print a structured payload in the CLI without exposing local paths."""
    if payload.get("success") and streamed:
        print()
    else:
        print(payload.get("answer_markdown") or payload.get("message", "Agent 未返回内容。"))

    if payload.get("saved"):
        print("\n本轮输出已保存到 outputs 目录")


def run_tutor_dialogue(
    client: object,
    resources: dict[str, str],
    system_prompt: str | None = None,
) -> None:
    """运行预习问答 / 操作答疑多轮命令行循环。"""
    conversation_history: list[dict[str, str]] = []
    conversation_state: dict[str, Any] = {
        "stage": "unknown",
        "topic": "unknown",
        "step": "first_turn",
        "turn_count": 0,
        "weak_points": [],
    }

    print("已进入预习问答 / 操作答疑 / 引导式回答模式。")
    print("可以连续提问或追问，输入 exit、quit 或 退出 可结束对话。")

    while True:
        student_input = input("\n学生：").strip()
        if student_input.lower() in {"exit", "quit"} or student_input == "退出":
            print("多轮对话已结束。")
            break
        if not student_input:
            print("输入不能为空，请重新输入。")
            continue

        print("\n========== Agent 输出 ==========\n")
        payload = ask_tutor(
            student_input=student_input,
            stage=str(conversation_state.get("stage", "unknown")),
            conversation_history=conversation_history,
            save=True,
            stream=True,
            conversation_state=conversation_state,
            client=client,
            resources=resources,
        )
        _print_payload(payload, streamed=True)

        if payload.get("success"):
            answer = str(payload.get("answer_markdown", ""))
            conversation_history.append({"role": "student", "content": student_input})
            conversation_history.append({"role": "agent", "content": answer})
            update_conversation_state(conversation_state, answer)


def main() -> None:
    try:
        resources = load_resources()
        mode = choose_mode()

        if mode == "1":
            client = create_client()
            run_tutor_dialogue(client, resources)
            return

        if mode == "2":
            client = create_client()
            report_content = read_multiline_input()
            if not report_content:
                raise ValueError("报告片段不能为空。")

            print("\n========== Agent 输出 ==========\n")
            payload = check_report(
                report_content=report_content,
                stage="report",
                save=True,
                stream=True,
                client=client,
                resources=resources,
            )
            _print_payload(payload, streamed=True)
            return

    except FileNotFoundError:
        print("文件错误：必要资源文件缺失，请检查 docs/ 和 prompts/ 目录。")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"配置错误：{exc}")
        sys.exit(1)
    except OpenAIError:
        print("API 调用失败，请检查 API Key、模型名、网络连接或账户权限。")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户中断运行。")
        sys.exit(1)
    except Exception as exc:
        print(f"运行失败：{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
