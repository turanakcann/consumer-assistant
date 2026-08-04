import streamlit as st
import time
import re
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import get_rag_chain

ENGLISH_TO_TURKISH = {
    "next step": "sonraki adım",
    "next steps": "sonraki adımlar",
    "complete": "tamamla",
    "possible": "mümkün",
    "please": "lütfen",
    "order": "sipariş",
    "refund": "iade",
    "credit": "kredi",
    "product": "ürün",
    "step": "adım",
    "delivery": "teslim",
    "warranty": "garanti",
    "contract": "sözleşme",
    "bank": "banka",
    "issue": "sorun",
}

def agent_alpha(raw_input: str) -> str:
    if not raw_input or not raw_input.strip():
        raise ValueError("Lütfen boş olmayan bir soru girin.")

    cleaned = raw_input.strip()
    for eng, tr in ENGLISH_TO_TURKISH.items():
        cleaned = re.sub(rf"\b{re.escape(eng)}\b", tr, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Sorgunuz geçersiz. Lütfen daha fazla Türkçe açıklama içeren bir soru yazın.")

    return cleaned

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
    
    /* 6. ALT KISIM (INPUT ARKASI) ŞEFFAFLIK VE INPUT BOYUTLANDIRMA */
    [data-testid="stBottom"] {
        background: transparent !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    
    /* Input (Sohbet Kutusu) Genişletme ve Hizalama */
    .stChatInputContainer {
        padding-bottom: 30px !important;
        max-width: 750px !important;
        margin: 0 auto !important;
    }
    .stChatInputContainer textarea {
        width: 100% !important;
        background-color: #111827 !important;
        color: white !important;
        border: 1px solid #374151 !important;
        border-radius: 15px !important;
    }

    /* Input Kutusunun Altına Sabitlenen Uyarı Balonu */
    .warning-balloon {
        position: fixed;
        bottom: 8px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 999;
        padding: 4px 16px;
        background: rgba(248, 196, 35, 0.12);
        border: 1px solid rgba(249, 115, 22, 0.35);
        color: #f8b204;
        border-radius: 10px;
        text-align: center;
        font-size: 0.8rem;
        pointer-events: none;
    }
    
    /* Buton Tasarımları */
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
        padding-bottom: 5rem !important;
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
    * **Google Embeddings:** `gemini-embedding-2` modeli ile hukuki metinlerin anlamsal eşleştirmesi.
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
if prompt := st.chat_input("Bana soru sorabilirsin..."):
    if not prompt.strip():
        st.warning("Lütfen geçerli bir soru girin.")
        st.stop()

    try:
        sanitized_prompt = agent_alpha(prompt)
    except ValueError as ve:
        st.warning(str(ve))
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=assistant_avatar):
        with st.spinner("Taranıyor..."):
            start_time = time.time()
            
            try:
                short_history = st.session_state.chat_history[-6:]
                response = rag_chain.invoke({
                    "input": sanitized_prompt,
                    "chat_history": short_history
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
                
            except Exception:
                st.error("Sistemde geçici bir bağlantı veya model hatası oluştu. Lütfen tekrar deneyiniz.")
st.markdown("<div class='warning-balloon'>Üretken yapay zeka hata yapabilir.</div>", unsafe_allow_html=True)