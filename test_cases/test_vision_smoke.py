"""手动 smoke test：触发 /chat/vision 与 /report-check/vision 端点，验证真实模型回路。

用法：
    1. 先启动后端 uvicorn；
    2. 把本机的一张干涉条纹照片（PNG/JPEG）放在 test_cases/sample_image.png；
       如果没有，本脚本会自动生成一张测试图。
    3. python test_vision_smoke.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

BASE = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent
SAMPLE_PATH = HERE / "sample_image.png"


def get_or_create_sample() -> bytes:
    if SAMPLE_PATH.exists():
        print(f"[INFO] 使用现有样本 {SAMPLE_PATH.name}")
        return SAMPLE_PATH.read_bytes()
    print(f"[INFO] 未发现 {SAMPLE_PATH.name}，生成 400x300 灰阶条纹测试图")
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    for x in range(0, 400, 16):
        draw.rectangle([x, 0, x + 8, 300], fill=(40, 40, 40))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def warmup() -> None:
    """触发一次 /chat 让 load_dotenv 把 ENABLE_VISION 加载进 os.environ。"""
    try:
        httpx.post(
            f"{BASE}/api/agent/chat",
            json={"session_id": "warmup", "stage": "unknown", "student_input": "你好"},
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] warmup 失败：{exc}")


def test_chat_vision(image_bytes: bytes) -> None:
    print("\n=== 路径 B：POST /api/agent/chat/vision ===")
    resp = httpx.post(
        f"{BASE}/api/agent/chat/vision",
        data={
            "session_id": "smoke",
            "stage": "unknown",
            "student_input": "这张图里你看到了什么？请简短描述图像特征。",
            "conversation_history": "[]",
        },
        files={"image": ("test.png", image_bytes, "image/png")},
        timeout=90,
    )
    print(f"HTTP {resp.status_code}")
    payload = resp.json()
    print(f"success={payload.get('success')} route={payload.get('route')} "
          f"error_code={payload.get('error_code')}")
    if payload.get("vision_info"):
        print(f"vision_info={payload['vision_info']}")
    if payload.get("success"):
        md = payload.get("answer_markdown", "")
        print(f"\nanswer_markdown (前 300 字):\n{md[:300]}")
    else:
        print(f"message: {payload.get('message')}")


def test_report_check_vision(image_bytes: bytes) -> None:
    print("\n=== 路径 C：POST /api/agent/report-check/vision (单图) ===")
    resp = httpx.post(
        f"{BASE}/api/agent/report-check/vision",
        data={"session_id": "smoke", "stage": "report"},
        files={"file": ("report.png", image_bytes, "image/png")},
        timeout=120,
    )
    print(f"HTTP {resp.status_code}")
    payload = resp.json()
    print(f"success={payload.get('success')} route={payload.get('route')} "
          f"error_code={payload.get('error_code')}")
    if payload.get("vision_info"):
        print(f"vision_info={payload['vision_info']}")
    if payload.get("document_info"):
        print(f"document_info={payload['document_info']}")
    if payload.get("success"):
        md = payload.get("answer_markdown", "")
        print(f"\nanswer_markdown (前 300 字):\n{md[:300]}")
    else:
        print(f"message: {payload.get('message')}")


def main() -> int:
    try:
        httpx.get(f"{BASE}/health", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 后端不可达 {BASE}: {exc}")
        return 2

    warmup()
    image_bytes = get_or_create_sample()
    print(f"[INFO] 图像 {len(image_bytes)} 字节\n")

    test_chat_vision(image_bytes)
    test_report_check_vision(image_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
