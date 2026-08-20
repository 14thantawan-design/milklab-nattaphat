"""WashLab Conversational RAG Chatbot - Gradio + ZeroGPU."""

import os
from functools import lru_cache

import gradio as gr
import spaces
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# Gemini API
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def split_kb_sections(text: str) -> list[str]:
    """
    แบ่ง Knowledge Base ตามหัวข้อ Markdown ##

    วิธีนี้ทำให้หัวข้อ เช่น 'บริการซักผ้า'
    อยู่กับราคา/ขนาดเครื่องของหัวข้อนั้น
    ไม่ถูกแยกคนละ chunk
    """
    parts = text.split("\n## ")

    sections = []

    for part in parts[1:]:
        section = "## " + part.strip()

        if section.strip():
            sections.append(section)

    # fallback ถ้าไฟล์ไม่มีหัวข้อ ##
    if not sections:
        sections = [
            chunk.strip()
            for chunk in text.split("\n\n")
            if chunk.strip()
        ]

    return sections


@lru_cache(maxsize=1)
def load_index():
    """Load WashLab KB and build FAISS index."""

    kb_path = "washlab_kb.md"

    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {kb_path}"
        )

    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_kb_sections(text)

    if not chunks:
        raise ValueError(
            "Knowledge Base ไม่มีข้อมูล"
        )

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # normalize เพื่อใช้ cosine similarity
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


def get_last_user_message(history) -> str:
    """ดึงคำถามก่อนหน้าของผู้ใช้จาก Gradio history."""

    if not history:
        return ""

    for item in reversed(history):

        # Gradio รุ่นใหม่
        if isinstance(item, dict):
            if (
                item.get("role") == "user"
                and isinstance(item.get("content"), str)
            ):
                return item["content"]

        # รองรับ Gradio รุ่นเก่า
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text = item[0]

            if isinstance(user_text, str) and user_text.strip():
                return user_text

    return ""


def history_to_text(history, max_items: int = 6) -> str:
    """แปลงประวัติแชทเป็นข้อความให้ Gemini ใช้ตีความคำถามต่อเนื่อง."""

    if not history:
        return "(ยังไม่มีบทสนทนาก่อนหน้า)"

    lines = []

    for item in history[-max_items:]:

        # Gradio รุ่นใหม่
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

        # รองรับ Gradio รุ่นเก่า
        elif isinstance(item, (list, tuple)) and len(item) >= 2:

            user_text = item[0]
            bot_text = item[1]

            if isinstance(user_text, str):
                lines.append(
                    f"ลูกค้า: {user_text}"
                )

            if isinstance(bot_text, str):
                lines.append(
                    f"WashLab: {bot_text}"
                )

    return "\n".join(lines) or "(ยังไม่มีบทสนทนาก่อนหน้า)"


def build_search_query(message: str, history) -> str:
    """
    ทำคำถามภาษาพูดให้เหมาะกับ Vector Search
    โดยไม่เปลี่ยนความหมายของคำถาม
    """

    text = message.strip()
    lower = text.lower()

    # คำถามสั้นมากที่โดยปกติหมายถึงหน้า WashLab นี้
    vague_questions = {
        "อันนี้อะไร",
        "นี่อะไร",
        "นี่คืออะไร",
        "อันนี้คืออะไร",
        "ร้านอะไร",
        "ร้านนี้คืออะไร",
        "ที่นี่คืออะไร",
        "คืออะไร",
    }

    if lower in vague_questions:
        return (
            "WashLab คือร้านอะไร "
            "เป็นร้านซักอบผ้าแบบใด "
            "และให้บริการอะไรบ้าง"
        )

    # ภาษาพูดเกี่ยวกับบริการซัก
    if "ซัก" in lower and any(
        word in lower
        for word in [
            "อะไรบ้าง",
            "ยังไงบ้าง",
            "แบบไหน",
            "มีอะไร",
            "มีซัก",
            "ล่ะ",
        ]
    ):
        return (
            "บริการซักผ้าของ WashLab "
            "มีเครื่องขนาดอะไรบ้าง "
            "ราคาเท่าไร "
            "และใช้เวลากี่นาที"
        )

    # ภาษาพูดเกี่ยวกับบริการอบ
    if "อบ" in lower and any(
        word in lower
        for word in [
            "อะไรบ้าง",
            "ยังไงบ้าง",
            "แบบไหน",
            "มีอะไร",
            "มีอบ",
            "ล่ะ",
        ]
    ):
        return (
            "บริการอบผ้าของ WashLab "
            "มีเครื่องขนาดอะไรบ้าง "
            "ราคาเท่าไร "
            "และใช้เวลากี่นาที"
        )

    # ถ้าเป็นคำถามสั้นต่อเนื่อง ให้พ่วงคำถามก่อนหน้ามาช่วยค้น
    last_user_message = get_last_user_message(history)

    if last_user_message and len(text) <= 30:
        return (
            "WashLab ร้านซักอบผ้าแบบ Self-Service\n"
            f"คำถามก่อนหน้า: {last_user_message}\n"
            f"คำถามต่อเนื่อง: {text}"
        )

    # ทุก query มี domain context ติดไปด้วย
    return (
        "WashLab ร้านซักอบผ้าแบบ Self-Service "
        f"{text}"
    )


def retrieve_top_k(
    query: str,
    model,
    index,
    chunks: list[str],
    k: int = 5
) -> list[str]:
    """Retrieve relevant KB sections."""

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

    return [
        chunks[i]
        for i in indices[0]
        if 0 <= i < len(chunks)
    ]


def generate_answer(
    query: str,
    context_chunks: list[str],
    history
) -> str:
    """Generate conversational answer from WashLab KB."""

    if not os.environ.get("GOOGLE_API_KEY"):
        return (
            "ระบบยังไม่ได้ตั้งค่า GOOGLE_API_KEY"
        )

    context = "\n\n".join(
        context_chunks
    )

    conversation = history_to_text(
        history
    )

    prompt = f"""
คุณคือผู้ช่วยของ WashLab ร้านซักอบผ้าแบบ Self-Service

หน้าที่ของคุณคือช่วยตอบคำถามลูกค้าอย่างเป็นธรรมชาติ
โดยอ้างอิงข้อเท็จจริงจาก Knowledge Base ที่ให้มา

กฎ:
1. ตอบเป็นภาษาไทยแบบเป็นกันเอง กระชับ และเข้าใจง่าย
2. ใช้ราคา เวลา ขนาดเครื่อง เวลาเปิดร้าน และเงื่อนไข
   จาก Knowledge Base เท่านั้น
3. ห้ามแต่งข้อมูลธุรกิจที่ไม่มีอยู่ใน Knowledge Base
4. สามารถสรุปหรือเรียบเรียงข้อมูลหลายข้อรวมกันได้
5. คำถามภาษาพูดหรือคำถามสั้น เช่น
   "อันนี้อะไร", "ร้านนี้อะไร", "มีซักยังไงบ้าง",
   "แล้วอบล่ะ"
   ให้ตีความจากบริบทการสนทนาและให้ถือว่า
   ผู้ใช้กำลังคุยเกี่ยวกับ WashLab
6. ถ้าผู้ใช้ถามว่ามีบริการอะไรบ้าง
   ให้สรุปบริการที่พบใน Knowledge Base
7. ใช้คำตอบว่า
   "ขออภัยครับ ทางร้านไม่มีข้อมูลในส่วนนี้"
   เฉพาะเมื่อข้อมูลนั้นไม่มีอยู่จริงใน Knowledge Base
   ไม่ใช่เพียงเพราะคำถามใช้คำไม่ตรงกับเอกสาร

[บทสนทนาล่าสุด]
{conversation}

[Knowledge Base ที่ค้นพบ]
{context}

[คำถามปัจจุบัน]
{query}

คำตอบ:
"""

    try:
        llm = genai.GenerativeModel(
            "gemini-3.6-flash"
        )

        response = llm.generate_content(
            prompt
        )

        return (
            response.text
            or "ขออภัยครับ ไม่สามารถสร้างคำตอบได้"
        )

    except Exception as e:
        return (
            "เกิดข้อผิดพลาดในการเรียก Gemini API: "
            f"{e}"
        )


@spaces.GPU(duration=30)
def chat(message, history):
    """Conversational WashLab RAG."""

    if not message or not message.strip():
        return "พิมพ์คำถามมาได้เลยครับ 😊"

    try:
        model, index, chunks = load_index()

        # ใช้ query สำหรับ search ที่มีบริบทมากขึ้น
        search_query = build_search_query(
            message,
            history
        )

        context = retrieve_top_k(
            search_query,
            model,
            index,
            chunks,
            k=5
        )

        return generate_answer(
            message,
            context,
            history
        )

    except Exception as e:
        return (
            f"เกิดข้อผิดพลาดในการโหลดระบบ: {e}"
        )


demo = gr.ChatInterface(
    fn=chat,
    title="🧺 WashLab RAG Chatbot",
    description=(
        "ผู้ช่วยตอบคำถามเกี่ยวกับบริการซักอบผ้าแบบ Self-Service "
        "โดยอ้างอิงข้อมูลจาก WashLab Knowledge Base"
    ),
    examples=[
        "WashLab คืออะไร?",
        "มีบริการซักแบบไหนบ้าง?",
        "ซักผ้า 10 kg ราคาเท่าไร?",
        "ซักผ้านวมควรใช้เครื่องขนาดไหน?",
        "อบผ้า 15 kg ใช้เวลากี่นาที?",
        "ต้องเอาน้ำยาซักผ้ามาเองไหม?",
        "ร้านเปิดกี่โมง?",
    ],
)


if __name__ == "__main__":
    demo.queue()
    demo.launch()
