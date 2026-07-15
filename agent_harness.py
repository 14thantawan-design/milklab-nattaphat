"""MilkLab Agent Harness (S2).

Usage:
    python agent_harness.py --cmd "บันทึกขายนมหมี 2 ขวด ขวดละ 65"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from google import genai
from google.genai import types

TOOL_SCHEMA = [
    {
        "name": "log_sale",
        "description": "บันทึกการขายลง Google Sheets และส่ง notification",
        "parameters": {
            "type": "object",
            "properties": {
                "menu": {"type": "string", "description": "ชื่อเมนู"},
                "qty": {"type": "integer", "description": "จำนวนที่ขาย"},
                "price": {"type": "number", "description": "ราคาต่อหน่วย"},
            },
            "required": ["menu", "qty", "price"],
        },
    },
    {
        "name": "query_sales",
        "description": "ดูยอดขายของวันที่ระบุ",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "วันที่ format YYYY-MM-DD"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "send_alert",
        "description": "ส่ง message แจ้งเตือนผ่าน Bot",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    },
]


def log_trace(event_type: str, detail: str):
    """ฟังก์ชันช่วยบันทึก Log ลงไฟล์ agent_trace.log ตามฟอร์แมตของอาจารย์"""
    # ตั้งเวลาบวก 7 ชั่วโมงให้เป็นเวลาประเทศไทย
    tz_th = timezone(timedelta(hours=7))
    timestamp = datetime.now(tz_th).strftime("%Y-%m-%d %H:%M")
    log_line = f"{timestamp} | {event_type} | {detail}\n"
    
    with open("agent_trace.log", "a", encoding="utf-8") as f:
        f.write(log_line)
    print(log_line.strip())


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """TODO 1: ส่ง cmd ไป Gemini พร้อม TOOL_SCHEMA ขอให้ตอบเป็น JSON {tool, args}"""
    if not api_key:
        api_key = os.environ.get(
            "GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("ไม่พบ GEMINI_API_KEY ในระบบ")

    client = genai.Client(api_key=api_key)

    prompt = f"""
    วิเคราะห์คำสั่งนี้: "{cmd}"
    
    นี่คือเครื่องมือที่คุณสามารถใช้ได้:
    {json.dumps(TOOL_SCHEMA, ensure_ascii=False)}
    
    กรุณาเลือกเครื่องมือที่เหมาะสมและตอบกลับเป็น JSON ล้วนๆ โดยไม่ต้องมีเครื่องหมาย ```json
    รูปแบบ: {{"tool": "ชื่อเครื่องมือ", "args": {{"พารามิเตอร์": "ค่า"}}}}
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",  # แก้ไข: ใช้ JSON mode อย่างเดียว
            ),
        )

        result_text = response.text.strip()
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        return json.loads(result_text)

    except Exception as e:
        raise RuntimeError(f"Parse error: {e}")


def dispatch_tool(tool_call: dict) -> str:
    """TODO 2: ตรวจสอบความถูกต้อง (Validate) ก่อนเรียกใช้งานจริง"""
    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})

    if tool_name == "log_sale":
        menu = args.get("menu", "").strip()
        qty = args.get("qty", 0)
        price = args.get("price", 0)

        # Guardrail (ป้องกันข้อมูลมั่ว)
        if not menu:
            raise ValueError("menu must not be empty")
        if qty <= 0:
            raise ValueError("quantity must be positive")
        if price < 0:
            raise ValueError("price cannot be negative")

        return "row appended"

    elif tool_name == "query_sales":
        date = args.get("date")
        if not date:
            raise ValueError("date is required")
        return "รายการขาย 5 บรรทัด"

    elif tool_name == "send_alert":
        msg = args.get("message")
        if not msg:
            raise ValueError("message is required")
        return "alert sent"

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="คำสั่งภาษาไทย")
    args = parser.parse_args()

    # TODO 3: รันการทำงานและบันทึก log ครบทั้ง 4 แบบ
    log_trace("user_input", args.cmd)

    try:
        tool_call = parse_command(args.cmd)
        log_trace("llm_response", json.dumps(tool_call, ensure_ascii=False))

        result = dispatch_tool(tool_call)
        log_trace("tool_result", result)

    except Exception as e:
        log_trace("tool_error", f"{type(e).__name__} {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
