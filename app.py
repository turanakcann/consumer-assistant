import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessage
from rag_engine import get_rag_chain

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Tüketici Hakları Asistanı",
    page_icon="⚖️",
    layout="centered"
)

# --- ARKA PLAN MOTORUNU YÜKLEME (Cache) ---
# Modelin ve veritabanının her soruda baştan yüklenmesini engeller, hızı korur.
@st.cache_resource
def load_rag_engine():
    return get_rag_chain()

rag_chain = load_rag_engine()

# --- HAFIZA VE SOHBET GEÇMİŞİ YÖNETİMİ ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # LangChain'in anlayacağı hafıza

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! Ben yapay zeka destekli Tüketici Hakları Asistanıyım. İadeler, ayıplı mallar veya sözleşmelerle ilgili sorularınızı sorabilirsiniz."}
    ] # Ekranda görünecek mesajlar

# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/6062/6062646.png", width=100) # İsteğe bağlı şık bir ikon
    st.title("⚖️ Hukuk Asistanı")
    st.markdown("Bu asistan **6502 sayılı Kanun** ve ilgili yönetmeliklere göre çalışır.")
    st.markdown("---")
    
    # Sohbeti temizleme butonu
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Sohbet temizlendi. Size nasıl yardımcı olabilirim?"}
        ]
        st.session_state.chat_history = []
        st.rerun()

# --- ANA EKRAN (SOHBET ARAYÜZÜ) ---
st.title("⚖️ Tüketici Hakları Yapay Zeka Asistanı")

# Geçmiş mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- KULLANICI GİRİŞİ VE CEVAP ÜRETİMİ ---
if prompt := st.chat_input("Örn: İnternetten aldığım ürünü kaç gün içinde iade edebilirim?"):
    
    # 1. Kullanıcının mesajını ekrana yaz
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Asistanın cevabını oluştur
    with st.chat_message("assistant"):
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