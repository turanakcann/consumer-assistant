import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import get_rag_chain

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Tüketici Hakları Asistanı",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS İLE KARANLIK TEMA VE DÜZEN ÖZELLEŞTİRME ---
st.markdown("""
<style>
    /* 1. Tam Ekran Karanlık Degrade Arka Plan */
    .stApp {
        background: linear-gradient(180deg, #0a1128 0%, #121c3a 100%);
        color: #e0e6ed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* 2. Sol Menüyü ve Üst Banner'ı Gizleme */
    [data-testid="collapsedControl"], [data-testid="stSidebar"], [data-testid="stHeader"] {
        display: none !important;
    }

    /* 3. Genel Mesaj Balonu Ayarları */
    [data-testid="stChatMessage"] {
        padding: 15px;
        margin-bottom: 15px;
        color: #e0e6ed !important;
    }

    /* 4. KULLANICI Mesaj Balonu */
    [data-testid="stChatMessage"]:has([data-testid="stIcon-user"]), 
    [data-testid="stChatMessage"]:has(img[alt="user"]) {
        background-color: rgba(30, 58, 138, 0.4) !important;
        border-radius: 20px 20px 0px 20px !important;
        border: 1px solid rgba(59, 130, 246, 0.3);
        backdrop-filter: blur(10px);
    }

    /* 5. ASİSTAN Mesaj Balonu */
    [data-testid="stChatMessage"]:has(img[alt="assistant"]) {
        background-color: rgba(17, 24, 39, 0.6) !important;
        border-radius: 20px 20px 20px 0px !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* 6. ALT KISIM (INPUT ARKASI) ŞEFFAFLIK */
    [data-testid="stBottom"] {
        background: transparent !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    
    /* Input Alanı */
    .stChatInputContainer {
        padding-bottom: 20px !important;
    }
    .stChatInputContainer textarea {
        background-color: #111827 !important;
        color: white !important;
        border: 1px solid #374151 !important;
        border-radius: 15px !important;
    }
    
    /* Buton Tasarımları (Yazıların sığması için genişletildi) */
    .stButton button {
        background-color: rgba(59, 130, 246, 0.2) !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        color: #60a5fa !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
        padding: 6px 20px !important;
        white-space: nowrap !important;
    }
    .stButton button:hover {
        background-color: rgba(59, 130, 246, 0.4) !important;
        border-color: #3b82f6 !important;
        color: white !important;
    }

    /* Ana içeriği ortalama */
    .block-container {
        max-width: 800px !important;
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- POP-UP (MODAL) KILAVUZ EKRANI ---
@st.dialog("📖 Tüketici Hakları Asistanı Kılavuzu")
def show_guide():
    st.markdown("""
    ### 🚀 Proje Hakkında
    Bu asistan, **6502 sayılı Tüketici'nin Korunması Hakkında Kanun** ve ilgili 28 yönetmeliği temel alarak geliştirilmiş bir yapay zeka (RAG) uygulamasıdır.
    
    #### 🧠 Sistem Mimarisi
    * **Google Embeddings:** `gemini-embedding-001` modeli ile hukuki metinlerin anlamsal eşleştirmesi.
    * **Llama 3.3 (Groq):** Hızlı ve yüksek doğruluklu nihai cevap üretimi.
    * **Sıfır Halüsinasyon:** Sistem sadece verilen hukuki dökümanlara bağlı kalır, şahsi yorum yapmaz.
    
    #### 💡 Nasıl Kullanılır?
    Aşağıdaki gibi gündelik dille sorular sorabilirsiniz:
    * *"İnternetten aldığım tişörtü kaç gün içinde iade edebilirim?"*
    * *"Bozulan telefonum 25 gündür serviste, haklarım nelerdir?"*
    * *"Emlakçı benden ev alımında %6 komisyon istiyor, bu yasal mı?"*
    """)
    if st.button("Kapat"):
        st.rerun()

# --- ARKA PLAN MOTORU ---
@st.cache_resource
def load_rag_engine():
    return get_rag_chain()

rag_chain = load_rag_engine()

# --- HAFIZA YÖNETİMİ ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben yapay zeka destekli Tüketici Hakları Asistanıyım. Sorularınızı dinliyorum."}
    ]

# --- ÜST BAR (BUTONLAR İÇİN GENİŞLETİLMİŞ KOLONLAR) ---
top_col1, top_col2, top_col3 = st.columns([7, 2.2, 2.2]) 
with top_col2:
    if st.button("📖 Kılavuz", use_container_width=True):
        show_guide()

with top_col3:
    if st.button("🗑️ Temizle", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Sohbet temizlendi. Sizi dinliyorum."}
        ]
        st.session_state.chat_history = []
        st.rerun()

# --- ANA EKRAN BAŞLIKLARI ---
st.markdown("<h2 style='text-align: center; color: #ffffff;'>⚖️ Tüketici Hakları Asistanı</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af;'>Tüketici mevzuatıyla ilgili sorularınızı sorabilirsiniz.</p>", unsafe_allow_html=True)
st.write("") 

# --- ASİSTAN AVATARI ---
assistant_avatar = "https://cdn-icons-png.flaticon.com/512/11696/11696071.png"

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    avatar_url = "user" if message["role"] == "user" else assistant_avatar
    with st.chat_message(message["role"], avatar=avatar_url):
        st.markdown(message["content"])

# --- KULLANICI GİRİŞİ VE CEVAP ÜRETİMİ ---
if prompt := st.chat_input("Ask me anything..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=assistant_avatar):
        with st.spinner("Taranıyor..."):
            start_time = time.time()
            
            try:
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response["answer"]
                process_time = round(time.time() - start_time, 2)
                
                st.markdown(answer)
                st.caption(f"⏱️ İşlem Süresi: {process_time} sn")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=answer)
                ])
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {str(e)}")