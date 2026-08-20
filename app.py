"""WashLab RAG Chatbot (S4 Pivot).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space.

This version pivots the original MilkLab RAG chatbot
to WashLab, a self-service laundry domain.
"""

import os

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


# ตั้งค่า Gemini API
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


@st.cache_resource
def load_index():
    """Load WashLab KB, split into chunks, create embeddings and FAISS index."""

    # Knowledge Base ใหม่ของ WashLab
    kb_path = "washlab_kb.md"

    if not os.path.exists(kb_path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {kb_path} กรุณาตรวจสอบตำแหน่งไฟล์"
        )

    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    # แบ่ง Markdown เป็น chunks ตามย่อหน้า
    chunks_list = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]

    if not chunks_list:
        raise ValueError("Knowledge Base ไม่มีข้อมูล")

    # Embedding model รองรับหลายภาษา รวมถึงภาษาไทย
    model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    embeddings = model.encode(chunks_list)

    # สร้าง FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(
        np.array(embeddings).astype("float32")
    )

    return model, index, chunks_list


def retrieve_top_k(
    query: str,
    model,
    index,
    chunks: list[str],
    k: int = 3
) -> list[str]:
    """Retrieve the most relevant WashLab knowledge chunks."""

    query_vector = model.encode([query])

    # ป้องกันกรณีจำนวน chunks น้อยกว่า k
    actual_k = min(k, len(chunks))

    distances, indices = index.search(
        np.array(query_vector).astype("float32"),
        actual_k
    )

    top_k_chunks = [
        chunks[i]
        for i in indices[0]
        if 0 <= i < len(chunks)
    ]

    return top_k_chunks


def generate_answer(
    query: str,
    context_chunks: list[str]
) -> str:
    """Generate an answer using only retrieved WashLab information."""

    context = "\n\n".join(context_chunks)

    prompt = f"""
คุณคือผู้ช่วยบริการลูกค้าของ WashLab
ร้านซักอบผ้าแบบ Self-Service

ตอบคำถามลูกค้าโดยใช้เฉพาะข้อมูลจาก
[ข้อมูลของ WashLab] ด้านล่างนี้เท่านั้น

กฎสำคัญ:
- ห้ามแต่งราคา เวลาเปิดร้าน ขนาดเครื่อง หรือบริการขึ้นเอง
- หากข้อมูลที่ถามไม่มีอยู่ในบริบท ให้ตอบว่า
  "ขออภัยครับ ทางร้านไม่มีข้อมูลในส่วนนี้"
- ตอบเป็นภาษาไทย กระชับ สุภาพ และเข้าใจง่าย
- หากลูกค้าถามเรื่องการเลือกเครื่อง ให้แนะนำตามข้อมูลที่มีเท่านั้น

[ข้อมูลของ WashLab]
{context}

คำถามจากลูกค้า:
{query}

คำตอบ:
"""

    try:
        llm_model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = llm_model.generate_content(prompt)

        return response.text or "ขออภัยครับ ไม่สามารถสร้างคำตอบได้"

    except Exception as e:
        return (
            "เกิดข้อผิดพลาดในการเรียกใช้ Gemini API: "
            f"{str(e)}"
        )


def main():
    st.set_page_config(
        page_title="WashLab RAG",
        page_icon="🧺"
    )

    st.title("🧺 WashLab RAG Chatbot")

    st.caption(
        "ผู้ช่วยตอบคำถามเกี่ยวกับบริการซักอบผ้า "
        "โดยอ้างอิงข้อมูลจาก washlab_kb.md"
    )

    try:
        model, index, chunks = load_index()

    except Exception as exc:
        st.error(
            f"Error loading system: {exc}"
        )
        st.stop()

    # เก็บประวัติการสนทนา
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input(
        "เช่น ซักผ้านวมควรใช้เครื่องขนาดไหน?"
    ):
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner(
                "กำลังค้นข้อมูลของ WashLab..."
            ):
                context = retrieve_top_k(
                    prompt,
                    model,
                    index,
                    chunks
                )

                answer = generate_answer(
                    prompt,
                    context
                )

            st.write(answer)

            # แสดงว่า RAG ดึงข้อมูลส่วนไหนมาใช้
            with st.expander(
                "ดูข้อมูลอ้างอิงจาก Knowledge Base"
            ):
                for i, chunk in enumerate(
                    context,
                    1
                ):
                    st.markdown(
                        f"**[{i}]** {chunk}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


if __name__ == "__main__":
    main()
