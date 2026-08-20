"""WashLab Hybrid Conversational RAG Chatbot.

- WashLab-specific facts come from washlab_kb.md
- General questions can be answered with Gemini general knowledge
- Conversation history is used for follow-up questions
- Runs on Hugging Face Gradio + ZeroGPU
"""

import os
from functools import lru_cache

import gradio as gr
import spaces
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.6-flash"
KB_PATH = "washlab_kb.md"


APP_CONTEXT = """
คุณกำลังสนทนาอยู่ใน WashLab RAG Chatbot

WashLab เป็นร้านซักอบผ้าแบบ Self-Service
Chatbot นี้มีหน้าที่ช่วยตอบข้อมูลของ WashLab
และสามารถพูดคุยหรือตอบคำถามทั่วไปได้ด้วย

ข้อมูลเฉพาะของร้าน เช่น ราคา ขนาดเครื่อง เวลาเปิดร้าน
บริการ วิธีชำระเงิน และข้อกำหนดของร้าน
ต้องอ้างอิงจาก WashLab Knowledge Base เท่านั้น
"""


def get_client():
    """Create Gemini client from Hugging Face secret."""

    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


def split_kb_sections(text: str) -> list[str]:
    """
    Split Markdown KB by ## headings.

    Keeps each heading together with its content,
    which works better than splitting every paragraph.
    """

    sections = []
    current_section = []

    for line in text.splitlines():

        if line.startswith("## "):

            if current_section:
                section = "\n".join(current_section).strip()

                if section:
                    sections.append(section)

            current_section = [line]

        elif current_section:
            current_section.append(line)

    if current_section:
        section = "\n".join(current_section).strip()

        if section:
            sections.append(section)

    # Fallback if KB has no ## headings
    if not sections:
        sections = [
            chunk.strip()
            for chunk in text.split("\n\n")
            if chunk.strip()
        ]

    return sections


@lru_cache(maxsize=1)
def load_index():
    """Load WashLab KB and build semantic search index."""

    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {KB_PATH}"
        )

    with open(KB_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_kb_sections(text)

    if not chunks:
        raise ValueError(
            "Knowledge Base ไม่มีข้อมูล"
        )

    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Normalize vectors so Inner Product acts like cosine similarity
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        np.asarray(
            embeddings,
            dtype="float32"
        )
    )

    return model, index, chunks


def history_to_text(history, max_messages: int = 8) -> str:
    """
    Convert Gradio history to readable text.

    Supports both newer message format
    and older tuple-style history.
    """

    if not history:
        return "(ยังไม่มีบทสนทนาก่อนหน้า)"

    lines = []

    for item in history[-max_messages:]:

        # New Gradio message format
        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if not isinstance(content, str):
                continue

            if role == "user":
                lines.append(
                    f"ลูกค้า: {content}"
                )

            elif role == "assistant":
                lines.append(
                    f"WashLab: {content}"
                )

        # Older Gradio tuple format
        elif (
            isinstance(item, (list, tuple))
            and len(item) >= 2
        ):

            user_text = item[0]
            assistant_text = item[1]

            if isinstance(user_text, str):
                lines.append(
                    f"ลูกค้า: {user_text}"
                )

            if isinstance(assistant_text, str):
                lines.append(
                    f"WashLab: {assistant_text}"
                )

    return (
        "\n".join(lines)
        or "(ยังไม่มีบทสนทนาก่อนหน้า)"
    )


def recent_user_messages(history, limit: int = 2) -> list[str]:
    """Get recent user messages for better follow-up retrieval."""

    if not history:
        return []

    messages = []

    for item in reversed(history):

        if isinstance(item, dict):

            if (
                item.get("role") == "user"
                and isinstance(item.get("content"), str)
            ):
                messages.append(
                    item["content"]
                )

        elif (
            isinstance(item, (list, tuple))
            and len(item) >= 1
            and isinstance(item[0], str)
        ):
            messages.append(
                item[0]
            )

        if len(messages) >= limit:
            break

    messages.reverse()

    return messages


def build_search_query(message: str, history) -> str:
    """
    Build semantic-search query from current message + recent context.

    No fixed list of user questions is required.
    """

    previous_messages = recent_user_messages(
        history,
        limit=2
    )

    parts = previous_messages + [message]

    return "\n".join(parts)


def retrieve_top_k(
    query: str,
    model,
    index,
    chunks: list[str],
    k: int = 6
) -> list[str]:
    """Retrieve the most relevant WashLab KB sections."""

    query_vector = model.encode(
        [query],
        normalize_embeddings=True
    )

    actual_k = min(
        k,
        len(chunks)
    )

    _, indices = index.search(
        np.asarray(
            query_vector,
            dtype="float32"
        ),
        actual_k
    )

    results = []

    for i in indices[0]:

        if 0 <= i < len(chunks):
            results.append(
                chunks[i]
            )

    return results


def generate_hybrid_answer(
    message: str,
    history,
    context_chunks: list[str]
) -> str:
    """
    Hybrid answer:
    - WashLab facts must come from KB.
    - General questions may use Gemini general knowledge.
    """

    client = get_client()

    if client is None:
        return (
            "ระบบยังไม่ได้ตั้งค่า GOOGLE_API_KEY"
        )

    history_text = history_to_text(
        history
    )

    kb_context = "\n\n".join(
        context_chunks
    )

    prompt = f"""
คุณคือผู้ช่วย AI ของ WashLab

[บริบทของระบบ]
{APP_CONTEXT}

[บทสนทนาก่อนหน้า]
{history_text}

[ข้อมูลของ WashLab ที่ค้นคืนจาก Knowledge Base]
{kb_context}

[ข้อความล่าสุดของผู้ใช้]
{message}

ให้ตอบโดยใช้หลักการต่อไปนี้:

1. ก่อนตอบ ให้พิจารณาก่อนว่าผู้ใช้กำลังถาม
   - ข้อมูลเฉพาะของ WashLab
   - คำถามทั่วไปเกี่ยวกับการซักผ้า
   - หรือคำถาม/บทสนทนาทั่วไป

2. ถ้าคำถามเกี่ยวกับ WashLab โดยเฉพาะ
   เช่น ราคา บริการ ขนาดเครื่อง เวลาเปิดร้าน
   วิธีชำระเงิน ข้อจำกัด หรือนโยบายของร้าน
   ให้ใช้ข้อเท็จจริงจาก Knowledge Base เท่านั้น

3. ถ้าข้อมูลเฉพาะของ WashLab ไม่มีอยู่ใน Knowledge Base
   ให้บอกอย่างตรงไปตรงมาว่า
   "ตอนนี้ Knowledge Base ของ WashLab ยังไม่มีข้อมูลเรื่องนี้ครับ"
   ห้ามแต่งข้อมูลของร้านขึ้นมาเอง

4. ถ้าเป็นคำถามทั่วไป
   สามารถใช้ความรู้ทั่วไปของคุณตอบได้ตามปกติ

5. ถ้าเป็นคำแนะนำทั่วไปเกี่ยวกับการซักผ้า
   สามารถตอบได้ แต่ถ้าอาจทำให้เข้าใจว่าเป็นกฎของ WashLab
   ให้ระบุว่าเป็น "คำแนะนำทั่วไป"
   และไม่ใช่นโยบายเฉพาะของร้าน

6. ใช้บทสนทนาก่อนหน้าเพื่อเข้าใจคำถามต่อเนื่อง
   เช่น "แล้วอันนั้นล่ะ", "แล้วอบล่ะ", "ตัวใหญ่กว่านี้ล่ะ"

7. ถ้าผู้ใช้ใช้คำอย่าง
   "อันนี้", "ที่นี่", "ร้านนี้", "บอทนี้"
   โดยไม่มีบริบทอื่นขัดแย้ง
   ให้เข้าใจว่าหมายถึง WashLab หรือ WashLab Chatbot
   ตามบริบทของการสนทนา

8. อย่าปฏิเสธคำถามทั่วไปเพียงเพราะ
   คำตอบไม่ได้อยู่ใน Knowledge Base

9. ตอบภาษาไทยให้เป็นธรรมชาติ เป็นกันเอง
   กระชับ แต่มีข้อมูลเพียงพอ

คำตอบ:
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.35
            ),
        )

        return (
            response.text
            or "ขออภัยครับ ไม่สามารถสร้างคำตอบได้"
        )

    except Exception as exc:

        return (
            "เกิดข้อผิดพลาดในการเรียก Gemini API: "
            f"{exc}"
        )


@spaces.GPU(duration=60)
def chat(message, history):
    """Main Hybrid Conversational RAG function."""

    if not message or not message.strip():
        return (
            "พิมพ์คำถามมาได้เลยครับ 😊"
        )

    try:

        model, index, chunks = load_index()

        search_query = build_search_query(
            message,
            history
        )

        context = retrieve_top_k(
            search_query,
            model,
            index,
            chunks,
            k=6
        )

        return generate_hybrid_answer(
            message,
            history,
            context
        )

    except Exception as exc:

        return (
            "เกิดข้อผิดพลาดในการโหลดระบบ: "
            f"{exc}"
        )


demo = gr.ChatInterface(
    fn=chat,
    title="🧺 WashLab RAG Chatbot",
    description=(
        "ผู้ช่วย AI สำหรับ WashLab "
        "ตอบข้อมูลเฉพาะร้านจาก Knowledge Base "
        "และสามารถพูดคุยหรือให้คำแนะนำทั่วไปได้"
    ),
    examples=[
        "WashLab คืออะไร?",
        "มีบริการอะไรบ้าง?",
        "ซักผ้า 10 kg ราคาเท่าไร?",
        "ซักผ้านวมควรใช้เครื่องขนาดไหน?",
        "แล้วอบล่ะ?",
        "ผ้าขาวควรซักยังไง?",
        "สวัสดี ทำอะไรได้บ้าง?",
    ],
)


if __name__ == "__main__":
    demo.queue()
    demo.launch()
