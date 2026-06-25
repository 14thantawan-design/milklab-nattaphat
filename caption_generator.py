"""MilkLab Caption Generator (S1).

Usage:
    python caption_generator.py

Reads GOOGLE_API_KEY from env. Generates a Thai caption for a milk menu item.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai


PROMPT_TEMPLATE = """\
คุณคือ social media manager ของร้าน MilkLab° ร้านนมสดกลางคืน

จงเขียนแคปชั่นภาษาไทยเพื่อโปรโมตเมนู: {menu}
โดยบังคับให้เขียนออกมา 3 รูปแบบ ดังต่อไปนี้:

1. Cute: โทนน่ารัก สดใส อ้อนๆ ใส่ emoji น่ารักๆ
2. Minimal: โทนสั้น กระชับ เรียบง่าย ไม่ต้องมี emoji เยอะ
3. Gen-Z: โทนวัยรุ่นเทสดี ใช้ศัพท์วัยรุ่นฮิตๆ กวนๆ สนุกสนาน

เงื่อนไข:
- ทุกรูปแบบต้องมี call-to-action ปิดท้าย เช่น สั่งเลย หรือ ทักแชท
- ห้ามใช้ em dash
"""


def generate_caption(menu: str, api_key: str | None = None) -> str:
    """Generate a Thai caption for the given milk menu item."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set in env or argument")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=PROMPT_TEMPLATE.format(menu=menu),
    )
    return response.text or ""


def main() -> int:
    load_dotenv()
    menu = input("เมนูที่จะโปรโมต: ").strip()
    if not menu:
        print("กรุณาใส่ชื่อเมนู")
        return 1
    caption = generate_caption(menu)
    print()
    print(caption)
    return 0


if __name__ == "__main__":
    sys.exit(main())
