"""WashLab Service Logger (S4 Pivot).

Usage:
    python sales_logger.py --service "ซักผ้า" --machine "10 kg" --cycles 2 --price 40

Appends WashLab service usage data to Google Sheets
and sends a Telegram notification.
"""

import os
import sys
import argparse
import json
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# โหลดค่ากุญแจต่าง ๆ จาก Environment
load_dotenv()

# ใช้ Spreadsheet เดิม แต่เปลี่ยน schema ของข้อมูลเป็น WashLab
SPREADSHEET_ID = "1mH93q8xykyLV90fp3I9Cruy53gB-p4phCIEOSnw_dGk"


def log_service_to_sheets(
    service_name: str,
    machine_size: str,
    cycles: int,
    price: float,
    total: float
) -> str:
    """Append a WashLab service record to Google Sheets."""

    creds_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")

    if not creds_json:
        print(
            "[Error] ไม่พบกุญแจ GOOGLE_SHEETS_CREDENTIALS ในระบบ!",
            file=sys.stderr
        )
        print(
            "กรุณาตรวจสอบการตั้งค่า Secret ใน GitHub Codespaces",
            file=sys.stderr
        )
        sys.exit(1)

    try:
        creds_dict = json.loads(creds_json)

        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )

        service = build("sheets", "v4", credentials=creds)

        # เวลาประเทศไทย UTC+7
        tz_th = timezone(timedelta(hours=7))
        timestamp = datetime.now(tz_th).strftime("%Y-%m-%d %H:%M:%S")

        # Schema ใหม่:
        # เวลา | บริการ | ขนาดเครื่อง | จำนวนรอบ | ราคาต่อรอบ | ยอดรวม
        values = [[
            timestamp,
            service_name,
            machine_size,
            cycles,
            price,
            total
        ]]

        body = {"values": values}

        service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID,
            range="Sheet1!A:F",
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()

        return timestamp

    except HttpError as err:
        print(
            f"[Error] ไม่สามารถเข้าถึง Google Sheets ได้: {err.reason}",
            file=sys.stderr
        )
        print(
            "กรุณาตรวจสอบว่าแชร์ Sheet ให้ Service Account แล้ว "
            "และใส่ SPREADSHEET_ID ถูกต้อง",
            file=sys.stderr
        )
        sys.exit(1)

    except Exception as e:
        print(
            f"[Error] เกิดข้อผิดพลาดใน Google Sheets: {e}",
            file=sys.stderr
        )
        sys.exit(1)


def send_telegram_alert(
    service_name: str,
    machine_size: str,
    cycles: int,
    total: float,
    timestamp: str
):
    """Send a WashLab service notification to Telegram."""

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print(
            "[Warning] ข้ามการแจ้งเตือน: "
            "ไม่พบ TELEGRAM_BOT_TOKEN หรือ TELEGRAM_CHAT_ID"
        )
        return

    message = (
        f"🧺 *บันทึกการใช้บริการใหม่ (WashLab)*\n"
        f"📅 เวลา: {timestamp}\n"
        f"🫧 บริการ: {service_name}\n"
        f"⚙️ ขนาดเครื่อง: {machine_size}\n"
        f"🔢 จำนวน: {cycles} รอบ\n"
        f"💰 รวมเงินทั้งสิ้น: {total} บาท\n"
        f"บันทึกข้อมูลเรียบร้อยแล้ว ✅"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            print(
                f"[Warning] Telegram ส่งไม่สำเร็จ "
                f"(Status Code: {response.status_code})"
            )

    except Exception as e:
        print(
            f"[Warning] ไม่สามารถติดต่อ Telegram Server ได้: {e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="WashLab Service Logger Bot."
    )

    parser.add_argument(
        "--service",
        type=str,
        required=True,
        help='ประเภทบริการ เช่น "ซักผ้า" หรือ "อบผ้า"'
    )

    parser.add_argument(
        "--machine",
        type=str,
        required=True,
        help='ขนาดเครื่อง เช่น "10 kg", "15 kg" หรือ "20 kg"'
    )

    parser.add_argument(
        "--cycles",
        type=int,
        required=True,
        help="จำนวนรอบที่ใช้บริการ"
    )

    parser.add_argument(
        "--price",
        type=float,
        required=True,
        help="ราคาต่อรอบ"
    )

    args = parser.parse_args()

    if args.cycles <= 0:
        print("[Error] จำนวนรอบต้องมากกว่า 0")
        return 1

    if args.price < 0:
        print("[Error] ราคาต้องไม่ติดลบ")
        return 1

    total_amount = args.cycles * args.price

    print(
        f"กำลังบันทึกบริการ '{args.service}' "
        f"เครื่อง {args.machine} จำนวน {args.cycles} รอบ..."
    )

    timestamp = log_service_to_sheets(
        args.service,
        args.machine,
        args.cycles,
        args.price,
        total_amount
    )

    print(f"บันทึกข้อมูลสำเร็จเมื่อเวลา {timestamp}")

    send_telegram_alert(
        args.service,
        args.machine,
        args.cycles,
        total_amount,
        timestamp
    )

    print("ส่งการแจ้งเตือนเข้า Telegram เรียบร้อย!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
