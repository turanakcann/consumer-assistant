import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import get_rag_chain

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Tüketici Hakları Asistanı",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CSS İLE GÖRSEL ÖZELLEŞTİRME (resim.png Tasarımı) ---
# Streamlit'in varsayılan arayüzünü ezip senin istediğin soft ve turuncu/beyaz temaya çeviriyoruz.
st.markdown("""
<style>
    /* Soft Gri Arka Plan */
    .stApp {
        background-color: #f7f7f8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Yan Panel (Sidebar) Beyazlaştırma ve Çizgi Rengi */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eaeaea;
    }

    /* Tüm Mesaj Balonları İçin Ortak Kenar Yumuşatma */
    [data-testid="stChatMessage"] {
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.03);
    }

    /* KULLANICI Mesaj Balonu (Turuncu) - avatar'ında 'user' olanı hedefler */
    [data-testid="stChatMessage"]:has([data-testid="stIcon-user"]), 
    [data-testid="stChatMessage"]:has(img[alt="user"]) {
        background-color: #ff6200 !important;
        border-radius: 20px 20px 0px 20px !important;
        color: white !important;
    }
    
    /* Kullanıcı mesajı içindeki yazıların beyaz kalmasını sağlama */
    [data-testid="stChatMessage"]:has([data-testid="stIcon-user"]) p,
    [data-testid="stChatMessage"]:has(img[alt="user"]) p {
        color: white !important;
    }

    /* ASİSTAN Mesaj Balonu (Beyaz) - avatar'ında 'assistant' olanı hedefler */
    [data-testid="stChatMessage"]:has([data-testid="stIcon-assistant"]),
    [data-testid="stChatMessage"]:has(img[alt="assistant"]) {
        background-color: #ffffff !important;
        border-radius: 20px 20px 20px 0px !important;
        border: 1px solid #f0f0f0;
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- ARKA PLAN MOTORUNU YÜKLEME (Cache) ---
@st.cache_resource
def load_rag_engine():
    return get_rag_chain()

rag_chain = load_rag_engine()

# --- MENÜ VE ÇOKLU SAYFA (YÖNLENDİRME) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6062/6062646.png", width=80)
    st.title("⚖️ Hukuk Asistanı")
    st.markdown("---")
    
    # Sayfa Seçimi (Radio Button ile Yönlendirme)
    sayfa = st.radio(
        "Menü",
        ["💬 Sohbet Asistanı", "ℹ️ Proje Hakkında"]
    )
    
    st.markdown("---")
    
    # Sohbeti Temizle Butonu
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Merhaba! Ben yapay zeka destekli Tüketici Hakları Asistanıyım. İadeler, ayıplı mallar veya sözleşmelerle ilgili sorularınızı sorabilirsiniz."}
        ]
        st.session_state.chat_history = []
        st.rerun()

# --- HAFIZA VE SOHBET GEÇMİŞİ YÖNETİMİ ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben yapay zeka destekli Tüketici Hakları Asistanıyım. İadeler, ayıplı mallar veya sözleşmelerle ilgili sorularınızı sorabilirsiniz."}
    ]

# ==========================================
# SAYFA 1: SOHBET ASİSTANI (ANA EKRAN)
# ==========================================
if sayfa == "💬 Sohbet Asistanı":
    st.title("⚖️ Tüketici Hakları Asistanı")

    # Geçmiş mesajları ekrana bas (Rolleri CSS'in anlaması için açıkça belirtiyoruz)
    for message in st.session_state.messages:
        avatar_str = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(message["role"], avatar=avatar_str):
            st.markdown(message["content"])

    # --- KULLANICI GİRİŞİ VE CEVAP ÜRETİMİ ---
    if prompt := st.chat_input("Örn: İnternetten aldığım ürünü kaç gün içinde iade edebilirim?"):
        
        # 1. Kullanıcının mesajını ekrana yaz
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="user"):
            st.markdown(prompt)

        # 2. Asistanın cevabını oluştur
        with st.chat_message("assistant", avatar="assistant"):
            with st.spinner("Mevzuat taranıyor ve analiz ediliyor..."):
                start_time = time.time()
                
                try:
                    # Backend (RAG) motoruna soruyu ve hafızayı gönder
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response["answer"]
                    process_time = round(time.time() - start_time, 2)
                    
                    # Cevabı ve süreyi ekrana bas
                    st.markdown(answer)
                    st.caption(f"⏱️ İşlem Süresi: {process_time} saniye")
                    
                    # 3. Hafızayı güncelle
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.extend([
                        HumanMessage(content=prompt),
                        AIMessage(content=answer)
                    ])
                    
                except Exception as e:
                    st.error(f"Bir hata oluştu: {str(e)}")

# ==========================================
# SAYFA 2: PROJE HAKKINDA
# ==========================================
elif sayfa == "ℹ️ Proje Hakkında":
    st.title("ℹ️ Proje Hakkında")
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Tüketici Hakları Yapay Zeka Asistanı
    Bu proje, karmaşık hukuki metinleri, kanunları ve yönetmelikleri vatandaşlar için anlaşılır ve erişilebilir kılmak amacıyla geliştirilmiş **RAG (Retrieval-Augmented Generation)** tabanlı akıllı bir sohbet asistanıdır.
    
    Projenin geliştirilme sürecinde modern yazılım mühendisliği prensipleri, gelişmiş kalite güvence (QA) platform testleri ve konteyner orkestrasyonu (Docker) kullanılmıştır.
    
    #### 🧠 Sistem Mimarisi ve Zeka
    * **Vektör Veritabanı:** Google'ın en güncel anlamsal eşleştirme modeli (`text-embedding-004` / `gemini-embedding-001`) ile ChromaDB üzerinde vektörleştirilmiş 28 farklı güncel tüketici mevzuatı.
    * **Hibrit LLM Altyapısı:** Kullanıcı sorgularını analiz etmek (Multi-Query) ve hukuki bağlamı saniyeler içinde taramak için Groq (Llama 3.3) ve Gemini API'lerinin birleşik gücü kullanılmıştır.
    * **Sıfır Halüsinasyon (Zero-Hallucination):** Sisteme gömülü *Hukuki Yorumlama Sözlüğü* sayesinde asistan; ticari davalar, şahıs arası anlaşmazlıklar veya kapsam dışı sorularda uydurma cevaplar vermek yerine sınırlarını bilerek hareket eder.
    
    #### 🛠️ Kullanılan Teknolojiler
    * **Arayüz (Frontend):** Streamlit
    * **Arka Plan (Backend):** Python, LangChain, Pytest
    * **Dağıtım (Deployment):** Docker & Docker-Compose
    
    ---
    *Bu uygulama, yapay zeka teknolojileri ile Tüketici Hukukunu birleştirerek hak arama süreçlerini dijitalleştirmeyi ve kolaylaştırmayı amaçlayan inovatif bir mühendislik çalışmasıdır.*
    """)