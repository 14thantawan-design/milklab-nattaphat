"""WashLab Agent Harness (S4 Pivot).

Usage:
    python agent_harness.py --cmd "บันทึกซักผ้าเครื่อง 10 kg 2 รอบ รอบละ 40 บาท"
    python agent_harness.py --cmd "ซักผ้าเครื่อง 20 kg ราคาเท่าไร"
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from google import genai
from google.genai import types

import sales_logger


# ข้อมูลบริการขนาดเล็กสำหรับ Agent
# ตั้งใจเก็บเฉพาะข้อมูลที่จำเป็นเพื่อลดความสับสนของโมเดล
SERVICE_DB = {
    "ซักผ้า": {
        "10 kg": {
            "price": 40,
            "duration": "ประมาณ 30 นาที"
        },
        "15 kg": {
            "price": 50,
            "duration": "ประมาณ 35 นาที"
        },
        "20 kg": {
            "price": 60,
            "duration": "ประมาณ 40 นาที"
        },
    },
    "อบผ้า": {
        "10 kg": {
            "price": 40,
            "duration": "ประมาณ 30 นาที"
        },
        "15 kg": {
            "price": 50,
            "duration": "ประมาณ 35 นาที"
        },
    },
}


TOOL_SCHEMA = [
    {
        "name": "log_service",
        "description": (
            "บันทึกการใช้บริการซักหรืออบผ้าของ WashLab "
            "ลง Google Sheets และส่ง notification"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "ประเภทบริการ เช่น ซักผ้า หรือ อบผ้า"
                },
                "machine": {
                    "type": "string",
                    "description": "ขนาดเครื่อง เช่น 10 kg, 15 kg หรือ 20 kg"
                },
                "cycles": {
                    "type": "integer",
                    "description": "จำนวนรอบที่ใช้บริการ"
                },
                "price": {
                    "type": "number",
                    "description": "ราคาต่อรอบ"
                },
            },
            "required": [
                "service",
                "machine",
                "cycles",
                "price"
            ],
        },
    },
    {
        "name": "query_service",
        "description": (
            "ค้นหาข้อมูลบริการของ WashLab เช่น "
            "ราคา ระยะเวลา และขนาดเครื่อง"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "ประเภทบริการ เช่น ซักผ้า หรือ อบผ้า"
                },
                "machine": {
                    "type": "string",
                    "description": "ขนาดเครื่อง เช่น 10 kg, 15 kg หรือ 20 kg"
                },
            },
            "required": [
                "service",
                "machine"
            ],
        },
    },
]


def log_trace(event_type: str, detail: str):
    """บันทึก Agent trace ลงไฟล์ agent_trace.log"""

    tz_th = timezone(timedelta(hours=7))
    timestamp = datetime.now(tz_th).strftime("%Y-%m-%d %H:%M")

    log_line = f"{timestamp} | {event_type} | {detail}\n"

    with open("agent_trace.log", "a", encoding="utf-8") as f:
        f.write(log_line)

    print(log_line.strip())


def parse_command(cmd: str, api_key: str | None = None) -> dict:
    """ส่งคำสั่งภาษาไทยให้ Gemini เลือก Tool และสร้าง arguments."""

    if not api_key:
        api_key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        if not api_key:
            raise RuntimeError(
                "ไม่พบ GEMINI_API_KEY หรือ GOOGLE_API_KEY ในระบบ"
            )

    client = genai.Client(api_key=api_key)

    prompt = f"""
คุณคือ Agent ของ WashLab ร้านซักอบผ้าแบบ Self-Service

วิเคราะห์คำสั่งของผู้ใช้:
"{cmd}"

เครื่องมือที่สามารถใช้ได้:
{json.dumps(TOOL_SCHEMA, ensure_ascii=False)}

กฎสำคัญ:
- เลือกเฉพาะ tool ที่มีอยู่ใน TOOL_SCHEMA
- ห้ามสร้างชื่อ tool ใหม่
- ห้ามแต่งราคาเองถ้าผู้ใช้ไม่ได้ระบุราคาในการบันทึก
- machine ต้องอยู่ในรูปแบบ เช่น "10 kg", "15 kg", "20 kg"
- service ใช้คำว่า "ซักผ้า" หรือ "อบผ้า"
- ตอบเป็น JSON เท่านั้น

รูปแบบ:
{{"tool": "ชื่อเครื่องมือ", "args": {{"พารามิเตอร์": "ค่า"}}}}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        result_text = (response.text or "").strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()

        return json.loads(result_text)

    except Exception as e:
        raise RuntimeError(f"Parse error: {e}")


def dispatch_tool(tool_call: dict) -> str:
    """ตรวจสอบข้อมูลก่อนเรียกใช้งาน Tool จริง."""

    tool_name = tool_call.get("tool")
    args = tool_call.get("args", {})

    if tool_name == "log_service":
        service_name = str(args.get("service", "")).strip()
        machine_size = str(args.get("machine", "")).strip()
        cycles = args.get("cycles", 0)
        price = args.get("price", 0)

        # Guardrails
        if service_name not in SERVICE_DB:
            raise ValueError(
                "service ต้องเป็น ซักผ้า หรือ อบผ้า"
            )

        if machine_size not in SERVICE_DB[service_name]:
            raise ValueError(
                f"ไม่พบเครื่องขนาด {machine_size} "
                f"สำหรับบริการ {service_name}"
            )

        if not isinstance(cycles, int) or cycles <= 0:
            raise ValueError(
                "cycles must be a positive integer"
            )

        if not isinstance(price, (int, float)) or price < 0:
            raise ValueError(
                "price cannot be negative"
            )

        expected_price = SERVICE_DB[service_name][machine_size]["price"]

        if float(price) != float(expected_price):
            raise ValueError(
                f"ราคาของ {service_name} เครื่อง {machine_size} "
                f"ควรเป็น {expected_price} บาทต่อรอบ"
            )

        total_amount = cycles * price

        # บันทึก Google Sheets
        timestamp = sales_logger.log_service_to_sheets(
            service_name,
            machine_size,
            cycles,
            price,
            total_amount
        )

        # แจ้งเตือน Telegram
        sales_logger.send_telegram_alert(
            service_name,
            machine_size,
            cycles,
            total_amount,
            timestamp
        )

        return (
            f"บันทึก {service_name} เครื่อง {machine_size} "
            f"{cycles} รอบ สำเร็จเมื่อ {timestamp} "
            f"(Total: {total_amount} บาท)"
        )

    elif tool_name == "query_service":
        service_name = str(args.get("service", "")).strip()
        machine_size = str(args.get("machine", "")).strip()

        if service_name not in SERVICE_DB:
            raise ValueError(
                "service ต้องเป็น ซักผ้า หรือ อบผ้า"
            )

        if machine_size not in SERVICE_DB[service_name]:
            available = ", ".join(
                SERVICE_DB[service_name].keys()
            )

            raise ValueError(
                f"ไม่มีเครื่อง {machine_size} "
                f"สำหรับ {service_name}. "
                f"ขนาดที่มี: {available}"
            )

        service_info = SERVICE_DB[service_name][machine_size]

        return (
            f"{service_name} เครื่อง {machine_size}: "
            f"ราคา {service_info['price']} บาทต่อรอบ, "
            f"ใช้เวลา {service_info['duration']}"
        )

    else:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="WashLab Agent Harness"
    )

    parser.add_argument(
        "--cmd",
        required=True,
        help="คำสั่งภาษาไทยเกี่ยวกับบริการ WashLab"
    )

    args = parser.parse_args()

    log_trace("user_input", args.cmd)

    try:
        tool_call = parse_command(args.cmd)

        log_trace(
            "llm_response",
            json.dumps(tool_call, ensure_ascii=False)
        )

        result = dispatch_tool(tool_call)

        log_trace(
            "tool_result",
            result
        )

        print(f"\n{result}")

    except Exception as e:
        log_trace(
            "tool_error",
            f"{type(e).__name__} {str(e)}"
        )

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
