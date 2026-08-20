---
title: WashLab RAG
emoji: 🧺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# WashLab° Solopreneur Pivot (Session 4)

โปรเจกต์สำหรับวิชา 31-407-106-406 : AI for Solopreneurs

โปรเจกต์นี้ Pivot จาก **MilkLab°** ไปเป็น **WashLab** ร้านซักอบผ้าแบบ Self-Service โดยปรับระบบ Caption Generator, Service Logger, Agent Harness และ RAG Chatbot ให้เหมาะกับ domain ร้านซักผ้า

## เกี่ยวกับ WashLab

WashLab เป็นร้านซักอบผ้าแบบ Self-Service ที่ให้ข้อมูลเกี่ยวกับบริการซักและอบผ้า ราคา ขนาดเครื่อง ระยะเวลาใช้งาน วิธีชำระเงิน และคำถามที่พบบ่อยผ่าน RAG Chatbot

## ไฟล์หลัก

| ไฟล์                   | Session | คำอธิบาย                                                              |
| ---------------------- | ------- | --------------------------------------------------------------------- |
| `caption_generator.py` | S1 / S4 | สร้างแคปชั่นโปรโมตบริการของ WashLab                                   |
| `sales_logger.py`      | S2 / S4 | บันทึกข้อมูลการใช้บริการลง Google Sheets และส่ง Telegram notification |
| `agent_harness.py`     | S2 / S4 | รับคำสั่งภาษาไทยและเลือก tool สำหรับบริการ WashLab                    |
| `app.py`               | S3 / S4 | Streamlit RAG Chatbot สำหรับตอบคำถามลูกค้า                            |
| `washlab_kb.md`        | S4      | Knowledge Base ของบริการ ราคา ขนาดเครื่อง และ FAQ                     |
| `PIVOT.md`             | S4      | อธิบายแนวคิดและรายละเอียดการ Pivot จาก MilkLab° เป็น WashLab          |

## เครื่องมือ

* Python 3.11
* Gemini API
* Streamlit
* Sentence Transformers
* FAISS
* Google Sheets API
* Telegram Bot API

## WashLab RAG Chatbot

แชทบอทตอบคำถามเกี่ยวกับบริการของ **WashLab** โดยอ้างอิงข้อมูลจาก `washlab_kb.md`

ตัวอย่างคำถาม:

* ซักผ้า 10 kg ราคาเท่าไร?
* ซักผ้านวมควรใช้เครื่องขนาดไหน?
* อบผ้าใช้เวลากี่นาที?
* ต้องเอาน้ำยาซักผ้ามาเองไหม?
* ร้านเปิดกี่โมง?
* จ่ายเงินด้วย QR Code ได้ไหม?

ระบบถูกออกแบบให้ตอบจากข้อมูลใน Knowledge Base และหลีกเลี่ยงการเดาข้อมูลที่ไม่มีอยู่ในระบบ
