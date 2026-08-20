"""WashLab Caption Generator (S4 Pivot).

Usage:
    python caption_generator.py --service "ชื่อบริการ" --n 1

Reads GOOGLE_API_KEY from env.
Generates a Thai promotional caption for a WashLab laundry service.
"""

import os
import sys
import argparse

from dotenv import load_dotenv
from google import genai


SERVICE_DB = {
    "ซักผ้า 10 kg": {
        "price": 40,
        "duration": "ประมาณ 30 นาที",
        "note": "เหมาะสำหรับผ้าปริมาณน้อย"
    },
    "ซักผ้า 15 kg": {
        "price": 50,
        "duration": "ประมาณ 35 นาที",
        "note": "เหมาะสำหรับผ้าปริมาณปานกลาง"
    },
    "ซักผ้า 20 kg": {
        "price": 60,
        "duration": "ประมาณ 40 นาที",
        "note": "เหมาะสำหรับผ้าปริมาณมากหรือผ้านวม"
    },
    "อบผ้า 10 kg": {
        "price": 40,
        "duration": "ประมาณ 30 นาที",
        "note": "เหมาะสำหรับผ้าปริมาณน้อย"
    },
    "อบผ้า 15 kg": {
        "price": 50,
        "duration": "ประมาณ 35 นาที",
        "note": "เหมาะสำหรับผ้าปริมาณปานกลาง"
    },
    "น้ำยาซักผ้า": {
        "price": 10,
        "duration": "-",
        "note": "จำหน่ายเป็นซอง ลูกค้าสามารถนำมาเองได้เช่นกัน"
    },
    "default": {
        "price": "-",
        "duration": "-",
        "note": "บริการซักอบผ้าแบบ Self-Service ของ WashLab"
    }
}


PROMPT_TEMPLATE = """\
คุณคือ social media manager ของ WashLab ร้านซักอบผ้าแบบ Self-Service

จงเขียนแคปชั่นภาษาไทยเพื่อโปรโมตบริการ: {service}
รายละเอียด: {details}

เขียน 3 รูปแบบ (Friendly, Minimal, Gen-Z)

เงื่อนไขสำคัญ:
- มี call-to-action
- ห้ามใช้ em dash
- ใช้เฉพาะข้อมูลบริการที่ให้มา
- ห้ามแต่งราคา ระยะเวลา หรือบริการเพิ่มเติมเอง
- เขียนให้สั้นและอ่านง่าย
- รวมทั้ง 3 รูปแบบต้องมีความยาวไม่เกิน 280 ตัวอักษร
"""


def generate_caption(service: str, api_key: str | None = None) -> str:
    """Generate a Thai caption for a WashLab laundry service."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")

    client = genai.Client(api_key=key)

    service_info = SERVICE_DB.get(service, SERVICE_DB["default"])

    details_text = (
        f"ราคา {service_info['price']} บาท, "
        f"ระยะเวลา {service_info['duration']}, "
        f"หมายเหตุ: {service_info['note']}"
    )

    max_retries = 3

    for attempt in range(max_retries):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT_TEMPLATE.format(
                service=service,
                details=details_text
            ),
        )

        caption = response.text or ""

        if len(caption) <= 280:
            print(
                f"  [System] สร้างสำเร็จในรอบที่ {attempt + 1} "
                f"(ความยาว {len(caption)} ตัวอักษร)"
            )
            return caption

        print(
            f"  [System] รอบที่ {attempt + 1} ล้มเหลว: "
            f"แคปชันยาวเกินไป ({len(caption)}/280 ตัวอักษร) "
            "กำลังสร้างใหม่..."
        )

    print(
        "  [System] คำเตือน: สร้างครบ 3 รอบแล้ว "
        "แต่ความยาวยังเกิน 280 ตัวอักษร"
    )

    return caption


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate captions for WashLab."
    )

    parser.add_argument(
        "--service",
        type=str,
        help="ชื่อบริการที่ต้องการโปรโมต"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="จำนวนเซ็ตแคปชันที่ต้องการสร้าง (default: 1)"
    )

    args = parser.parse_args()

    if not args.service:
        print(
            'กรุณาใส่ชื่อบริการ โดยใช้ flag --service '
            '(เช่น python caption_generator.py --service "ซักผ้า 10 kg")'
        )
        return 1

    print(
        f"กำลังสร้างแคปชันสำหรับบริการ: "
        f"'{args.service}' จำนวน {args.n} ชุด...\n"
    )

    for i in range(args.n):
        print(f"=== ผลลัพธ์ชุดที่ {i + 1} ===")
        caption = generate_caption(args.service)
        print(f"\n{caption}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
