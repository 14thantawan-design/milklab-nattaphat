"""MilkLab Sales Logger (S2).

Usage:
    python sales_logger.py --menu "นมหมีฮอกไกโด" --qty 2 --price 65

Appends sales data to Google Sheets and sends a Telegram notification.
"""

import os
import sys
import argparse
import json
from datetime import datetime, timezone, timedelta  # แก้ไข: เพิ่ม timezone และ timedelta
import requests
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# โหลดค่ากุญแจต่างๆ จาก Environment
load_dotenv()

# ⚠️ แปะ Spreadsheet ID ของคุณที่ก๊อปปี้มาลงในเครื่องหมายคำพูดนี้แทนของเดิมครับ
SPREADSHEET_ID = "1mH93q8xykyLV90fp3I9Cruy53gB-p4phCIEOSnw_dGk"


def log_sale_to_sheets(menu: str, qty: int, price: float, total: float) -> str:
    """Append a sales record row to Google Sheets."""
    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")

    # 1.3 เงื่อนไขที่ 4: Handle case Sheets ไม่ accessible (กุญแจหาย/ไม่มีสิทธิ์)
    if not creds_json:
        print("[Error] ไม่พบกุญแจ GOOGLE_SHEETS_CREDENTIALS ในระบบ!",
              file=sys.stderr)
        print("กรุณาตรวจสอบการตั้งค่า Secret ใน GitHub Codespaces", file=sys.stderr)
        sys.exit(1)

    try:
        creds_dict = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)

        # แก้ไข: ตั้งค่าเวลาให้เป็นเวลาประเทศไทย (+7 ชั่วโมง)
        tz_th = timezone(timedelta(hours=7))
        timestamp = datetime.now(tz_th).strftime("%Y-%m-%d %H:%M:%S")
        
        values = [[timestamp, menu, qty, price, total]]
        body = {"values": values}

        # สั่ง Append ข้อมูลต่อท้ายแถวสุดท้ายอัตโนมัติ ใน Sheet1 คอลัมน์ A ถึง E
        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range="Sheet1!A:E",
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()

        return timestamp

    except HttpError as err:
        print(
            f"[Error] ไม่สามารถเข้าถึง Google Sheets ได้เนื่องจากระบบปฏิเสธ: {err.reason}", file=sys.stderr)
        print("กรุณาตรวจสอบว่า 1. แชร์ Sheet ให้ Email บอตหรือยัง 2. ใส่ SPREADSHEET_ID ถูกต้องไหม", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(
            f"[Error] เกิดข้อผิดพลาดที่ไม่คาดคิดระบบ Google Sheets: {e}", file=sys.stderr)
        sys.exit(1)


def send_telegram_alert(menu: str, qty: int, total: float, timestamp: str):
    """Send a sales notification to Telegram chat."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print(
            "[Warning] ข้ามการแจ้งเตือน: ไม่พบค่า TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID")
        return

    message = (
        f"🔔 *บันทึกยอดขายใหม่ (MilkLab°)*\n"
        f"📅 เวลา: {timestamp}\n"
        f"🥛 เมนู: {menu}\n"
        f"🔢 จำนวน: {qty} แก้ว\n"
        f"💰 รวมเงินทั้งสิ้น: {total} บาท\n"
        f"บันทึกเข้าคลาวด์เรียบร้อยแล้ว! ✅"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(
                f"[Warning] Telegram ส่งไม่สำเร็จ (Status Code: {response.status_code})")
    except Exception as e:
        print(f"[Warning] ไม่สามารถติดต่อ Telegram Server ได้: {e}")


def main() -> int:
    # 1.3 เงื่อนไขที่ 1: อ่าน Command-line Arguments ด้วย argparse
    parser = argparse.ArgumentParser(description="MilkLab Sales Logger Bot.")
    parser.add_argument("--menu", type=str, required=True,
                        help="ชื่อเมนูเครื่องดื่ม")
    parser.add_argument("--qty", type=int, required=True,
                        help="จำนวนที่ขายได้")
    parser.add_argument("--price", type=float,
                        required=True, help="ราคาต่อหน่วย")
    args = parser.parse_args()

    # คำนวณยอดรวมสุทธิ
    total_amount = args.qty * args.price

    print(f"กำลังบันทึกข้อมูลเมนู '{args.menu}' จำนวน {args.qty} แก้ว...")

    # 1.3 เงื่อนไขที่ 2: บันทึกข้อมูลลง Google Sheets
    ts = log_sale_to_sheets(args.menu, args.qty, args.price, total_amount)
    print(f" บันทึกข้อมูลสำเร็จเมื่อเวลา {ts}")

    # 1.3 เงื่อนไขที่ 3: ส่งการแจ้งเตือนผ่านบอต Telegram
    send_telegram_alert(args.menu, args.qty, total_amount, ts)
    print(" ส่งการแจ้งเตือนเข้า Telegram เรียบร้อย!")

    return 0


if __name__ == "__main__":
    sys.exit(main())