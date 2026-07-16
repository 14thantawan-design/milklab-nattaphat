"""MilkLab RAG Chatbot (S3).

Run locally: streamlit run app.py
Deploy: push to GitHub then Actions deploys to HuggingFace Space

นักศึกษาต้องเติม TODO 5 จุด ใน Session 3 Lab 2.2
"""

import os
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ตั้งค่าโมเดล Gemini API ไว้ล่วงหน้า
if "GOOGLE_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


@st.cache_resource
def load_index():
    """TODO 1+2+3: โหลด menu_kb.md, split เป็น chunk, encode ด้วย sentence-transformers,
    สร้าง faiss index. Cache เพราะโหลด model ครั้งแรกใช้เวลา 30 วินาที

    Returns: (model, index, chunks_list)
    """
    # 1. โหลดไฟล์ menu_kb.md
    kb_path = "menu_kb.md"
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {kb_path} กรุณาตรวจสอบตำแหน่งไฟล์")
        
    with open(kb_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Split เป็น chunk (ใช้การแบ่งด้วยการขึ้นบรรทัดใหม่สองครั้ง \n\n สำหรับโครงสร้าง Markdown)
    chunks_list = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    # 3. โหลดโมเดลสำหรับทำ Vector Embedding ภาษาไทยตามโจทย์กำหนด
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # แปลง Chunks เป็น Vectors
    embeddings = model.encode(chunks_list)

    # 4. สร้าง FAISS Index เพื่อทำ Similarity Search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    return model, index, chunks_list


def retrieve_top_k(query: str, model, index, chunks: list[str], k: int = 3) -> list[str]:
    """TODO 4: encode query, search index, return top-k chunks"""
    # แปลงคำถามของ User ให้เป็น Vector
    query_vector = model.encode([query])
    
    # ค้นหาข้อความที่ใกล้เคียงที่สุดจำนวน k อันดับจาก FAISS Index
    distances, indices = index.search(np.array(query_vector).astype('float32'), k)
    
    # ดึงข้อความตาม Index ที่ค้นเจอ
    top_k_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return top_k_chunks


def generate_answer(query: str, context_chunks: list[str]) -> str:
    """TODO 5: ส่ง query + context ไป Gemini, return answer

    Hint: build prompt that says "ตอบจากข้อมูลต่อไปนี้เท่านั้น ถ้าไม่มีใน context ให้บอกว่าไม่รู้"
    """
    # รวมเนื้อหาจาก Chunks ที่ดึงมาได้เข้าด้วยกันเพื่อส่งเป็น Context
    context = "\n\n".join(context_chunks)
    
    # ออกแบบ Prompt คุมบอทป้องกันอาการเดาข้อมูล (Hallucination)
    prompt = f"""คุณคือพนักงานบริการลูกค้าของร้าน MilkLab° จงตอบคำถามลูกค้าโดยอ้างอิงจากข้อมูลใน [บริบทข้อมูลร้าน] ด้านล่างนี้เท่านั้น
หากไม่พบคำตอบในบริบทข้อมูล หรือไม่มีข้อมูลในระบบ ให้ตอบสั้นๆ อย่างสุภาพว่า "ขออภัยครับ ทางร้านไม่มีข้อมูลในส่วนนี้" หรือ "ไม่ทราบครับ" ห้ามพยายามเดา นึกคิด หรือจินตนาการคำตอบนอกเหนือจากบริบทที่ให้ไว้เด็ดขาด

[บริบทข้อมูลร้าน]:
{context}

คำถามจากลูกค้า: {query}
คำตอบ:"""

    try:
        # ใช้โมเดล gemini-2.5-flash ในการประมวลผลคำตอบ
        llm_model = genai.GenerativeModel('gemini-2.5-flash')
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเรียกใช้ Gemini API: {str(e)}"


def main():
    st.set_page_config(page_title="MilkLab° RAG", page_icon="🥛")
    st.title("MilkLab° RAG Chatbot")
    st.caption("ถามอะไรเกี่ยวกับ MilkLab ได้ ตอบจาก menu_kb.md")

    try:
        model, index, chunks = load_index()
    except NotImplementedError as exc:
        st.error(f"TODO not implemented: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Error loading system: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("ถามอะไรเกี่ยวกับ MilkLab"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังค้นข้อมูล..."):
                context = retrieve_top_k(prompt, model, index, chunks)
                answer = generate_answer(prompt, context)
            st.write(answer)
            with st.expander("Source chunks"):
                for i, c in enumerate(context, 1):
                    st.markdown(f"**[{i}]** {c}")
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()