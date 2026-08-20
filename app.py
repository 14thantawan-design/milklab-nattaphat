"""WashLab RAG Chatbot (S4 Pivot - Gradio + ZeroGPU)."""

import os
from functools import lru_cache

import gradio as gr
import spaces
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ตั้งค่า Gemini API
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


@lru_cache(maxsize=1)
def load_index():
    """Load WashLab knowledge base and create FAISS index."""

    kb_path = "washlab_kb.md"

    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    if not chunks:
        raise ValueError("Knowledge Base ไม่มีข้อมูล")

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.asarray(embeddings, dtype="float32")
    )

    return model, index, chunks


def retrieve_top_k(query, model, index, chunks, k=3):
    """Retrieve relevant chunks from WashLab KB."""

    query_vector = model.encode([query])

    actual_k = min(k, len(chunks))

    _, indices = index.search(
        np.asarray(query_vector, dtype="float32"),
        actual_k
    )

    return [
        chunks[i]
        for i in indices[0]
        if 0 <= i < len(chunks)
    ]


def generate_answer(query, context_chunks):
    """Generate answer using only WashLab knowledge."""

    if not os.environ.get("GOOGLE_API_KEY"):
        return "ระบบยังไม่ได้ตั้งค่า GOOGLE_API_KEY"

    context = "\n\n".join(context_chunks)

    prompt = f"""
คุณคือผู้ช่วยบริการลูกค้าของ WashLab
ร้านซักอบผ้าแบบ Self-Service

ตอบคำถามโดยใช้เฉพาะข้อมูลจากบริบทด้านล่างเท่านั้น

กฎ:
- ห้ามแต่งราคา เวลา ขนาดเครื่อง หรือบริการขึ้นเอง
- หากไม่มีข้อมูล ให้ตอบว่า
  "ขออภัยครับ ทางร้านไม่มีข้อมูลในส่วนนี้"
- ตอบเป็นภาษาไทย กระชับ สุภาพ และเข้าใจง่าย

[ข้อมูลของ WashLab]
{context}

คำถาม:
{query}

คำตอบ:
"""

    try:
        llm = genai.GenerativeModel("gemini-2.5-flash")
        response = llm.generate_content(prompt)

        return response.text or "ขออภัยครับ ไม่สามารถสร้างคำตอบได้"

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเรียก Gemini API: {e}"


@spaces.GPU(duration=30)
def chat(message, history):
    """Main Gradio chatbot function for ZeroGPU Space."""

    if not message.strip():
        return "กรุณาพิมพ์คำถามครับ"

    try:
        model, index, chunks = load_index()

        context = retrieve_top_k(
            message,
            model,
            index,
            chunks
        )

        return generate_answer(
            message,
            context
        )

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการโหลดระบบ: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="🧺 WashLab RAG Chatbot",
    description=(
        "ผู้ช่วยตอบคำถามเกี่ยวกับบริการซักอบผ้าแบบ Self-Service "
        "โดยอ้างอิงข้อมูลจาก WashLab Knowledge Base"
    ),
    examples=[
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
