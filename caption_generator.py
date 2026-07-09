"""MilkLab Caption Generator (S1).

Usage:
    python caption_generator.py --menu "ชื่อเมนู" --n 1

Reads GOOGLE_API_KEY from env. Generates a Thai caption for a milk menu item.
"""

import os
import sys
import argparse

from dotenv import load_dotenv
from google import genai

MENU_DB = {
    "นมสดโอริโอ้ปั่น": {"price": 55, "ingredients": "นมสดแท้ 100%, คุกกี้โอริโอ้บดกรุบกรอบ"},
    "นมหมีฮอกไกโด": {"price": 65, "ingredients": "นมหมีกระป๋องแท้, หัวนมฮอกไกโดเข้มข้น"},
    "ปังปิ้งเนยนม": {"price": 30, "ingredients": "ขนมปังหนานุ่ม, เนยสดแท้, นมข้นหวานฉ่ำๆ"},
    "default": {"price": "-", "ingredients": "สูตรลับเฉพาะของ MilkLab"}
}


PROMPT_TEMPLATE = """\
คุณคือ social media manager ของร้าน MilkLab° ร้านนมสดกลางคืน

จงเขียนแคปชั่นภาษาไทยเพื่อโปรโมตเมนู: {menu}
รายละเอียด: {details}

เขียน 3 รูปแบบ (Cute, Minimal, Gen-Z) 
เงื่อนไขสำคัญ:
- มี call-to-action
- ห้ามใช้ em dash
- นำรายละเอียดเมนูไปแต่งด้วย
- บังคับ: เขียนให้สั้นที่สุด รวมทั้ง 3 รูปแบบต้องมีความยาวไม่เกิน 280 ตัวอักษร!
"""


def generate_caption(menu: str, api_key: str | None = None) -> str:
    """Generate a Thai caption for the given milk menu item with length validation."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")
    client = genai.Client(api_key=key)

    menu_info = MENU_DB.get(menu, MENU_DB["default"])
    details_text = f"ราคา {menu_info['price']}บ., ส่วนผสม: {menu_info['ingredients']}"

    max_retries = 3
    for attempt in range(max_retries):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT_TEMPLATE.format(menu=menu, details=details_text),
        )
        caption = response.text or ""

        # ตรวจสอบความยาวไม่เกิน 280 ตัวอักษร
        if len(caption) <= 280:
            print(
                f"  [System] สร้างสำเร็จในรอบที่ {attempt + 1} (ความยาว {len(caption)} ตัวอักษร)")
            return caption
        else:
            print(
                f"  [System] รอบที่ {attempt + 1} ล้มเหลว: แคปชันยาวเกินไป ({len(caption)}/280 ตัวอักษร) กำลังสร้างใหม่...")

    # ถ้าพยายามครบ 3 รอบแล้วยังยาวเกิน ให้คืนค่ารอบสุดท้ายไปเลยพร้อมคำเตือน
    print("  [System] คำเตือน: สร้างครบ 3 รอบแล้วแต่ความยาวยังเกิน 280 ตัวอักษร")
    return caption


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate captions for MilkLab.")
    parser.add_argument("--menu", type=str, help="ชื่อเมนูที่ต้องการโปรโมต")
    parser.add_argument("--n", type=int, default=1,
                        help="จำนวนเซ็ตแคปชันที่ต้องการสร้าง (default: 1)")
    args = parser.parse_args()

    if not args.menu:
        print("กรุณาใส่ชื่อเมนู โดยใช้ flag --menu (เช่น python caption_generator.py --menu นมสด)")
        return 1

    print(f"กำลังสร้างแคปชันสำหรับเมนู: '{args.menu}' จำนวน {args.n} ชุด...\n")

    for i in range(args.n):
        print(f"=== ผลลัพธ์ชุดที่ {i + 1} ===")
        caption = generate_caption(args.menu)
        print(f"\n{caption}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())


# Lab 1.3
