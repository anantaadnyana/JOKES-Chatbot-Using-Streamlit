import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="😂 Jokes AI", page_icon="🎭", layout="centered")

# ==================== TITLE ====================
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='font-size: 2.5rem;'>😂 Jokes AI</h1>
        <p style='color: gray; font-size: 1.1rem;'>Masukkan topik, dan biarkan AI yang bikin kamu ketawa 🤖</p>
        <hr style='margin-top: 1rem; margin-bottom: 1rem;'>
    </div>
""", unsafe_allow_html=True)

# ==================== PROMPT TEMPLATE ====================
PROMPT_TEMPLATE = """
Kamu adalah asisten yang ahli dalam stand-up comedy, berdasarkan buku "Basic Penulisan Comedy Berdasarkan Teknik Stand Up Comedy".
Gunakan referensi dari dokumen berikut untuk membuat jokes atau komedi.
Jika ada permintaan membuat jokes, gunakan konsep dari buku (set up, punchline, rule of three) sebagai dasar membuat jokes yang kamu berikan ke pengguna 
tampilkan hasil punchline langsung.
"""

# ==================== MESSAGE ROLE MAPPING ====================
MESSAGE_ROLE = {
    HumanMessage: "User",
    AIMessage: "Assistant",
}

# ==================== VECTOR DB LOADER ====================
@st.cache_resource
def load_vector_db(model_name: str):
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Database jokes tidak ditemukan, AI akan improvisasi saja 😂")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_db

# ==================== API KEY INPUT ====================
if "groq_api_key" not in st.session_state or not st.session_state["groq_api_key"]:
    api_key = st.text_input("🔑 Masukkan Groq API Key kamu:", type="password")
    if api_key:
        st.session_state["groq_api_key"] = api_key
        st.rerun()
    else:
        st.stop()

# ==================== INIT LLM ====================
if "llm" not in st.session_state:
    st.session_state["llm"] = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=st.session_state["groq_api_key"]
    )

llm = st.session_state["llm"]

# ==================== LOAD VECTOR DB ====================
vector_db = load_vector_db("sentence-transformers/all-MiniLM-L6-v2")

# ==================== CHAT HISTORY ====================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
chat_history = st.session_state["chat_history"]

# ==================== UI: DISPLAY CHAT HISTORY ====================
st.markdown("""
<style>
.chat-box {
    background-color: #1E1E1E;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.user-msg {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 5px;
    display: inline-block;
}
.ai-msg {
    background-color: #1E1E1E;
    border-radius: 10px;
    padding: 8px 12px;
    margin-bottom: 10px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
if len(chat_history) == 0:
    st.markdown("<p style='color: gray; text-align: center;'>Belum ada obrolan. Coba minta jokes dulu 😄</p>", unsafe_allow_html=True)
else:
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            st.markdown(f"<div class='user-msg'><b>Kamu:</b> {msg.content}</div>", unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f"<div class='ai-msg'><b>Jokes AI:</b> {msg.content}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ==================== USER INPUT ====================
user_input = st.text_input("Kamu mau jokes tentang apa hari ini?", placeholder="contoh: cinta, teknologi, sekolah...")

if st.button("Kirim 😂") and user_input:
    # Ambil dokumen terkait dari DB (opsional)
    if vector_db:
        relevant_docs = vector_db.similarity_search(user_input, k=3)
        docs_content = "\n\n".join(doc.page_content for doc in relevant_docs)
    else:
        docs_content = ""

    # Bangun prompt
    prompt = chat_history.copy()
    prompt.append(SystemMessage(content=PROMPT_TEMPLATE))
    prompt.append(HumanMessage(content=f"Context: {docs_content}\n\nUser: {user_input}"))

    # Tambah pesan user ke riwayat
    chat_history.append(HumanMessage(content=user_input))

    # Dapatkan respons AI
    response = llm.invoke(prompt)

    # Tambah respons ke riwayat dan tampilkan
    chat_history.append(response)
    st.rerun()

# Tombol clear chat
if st.button("🧹 Hapus Riwayat Chat"):
    st.session_state.chat_history = []
    st.rerun()
