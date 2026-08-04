import re
from rag_engine import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
import time

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

def chat_start():
    print("Sistem başlatılıyor")
    
    try:
        rag_chain = get_rag_chain()
    except Exception as e:
        from config import LLM_MODEL
        print(f"❌ Hata: Groq'un arka planda çalıştığından ve '{LLM_MODEL}' modelinin yüklü olduğundan emin olun. \nDetay: {e}\n")
        return
    
    print("="*50)
    print(" ⚖️  TÜKETİCİ HAKLARI RAG ASİSTANI HAZIR ")
    print("=" * 50)
    print("Çıkmak için 'q' veya 'çıkış' yazabilirsiniz.\n")
    
    chat_history = []
    
    while True:
        user_question = input("Tüketici sorusu: ")
        
        if user_question.lower() in ['q', 'exit', 'çıkış']:
            print("Asistan kapatılıyor, iyi çalışmalar")
            break
        
        if not user_question.strip():
            continue

        try:
            user_question_clean = agent_alpha(user_question)
        except ValueError as ve:
            print(f"Uyarı: {ve}")
            continue
        
        print("\nMevzuat taranıyor, bekleyiniz...")
        
        try:
            start_time = time.time()
            response = rag_chain.invoke({"input": user_question_clean,
                                         "chat_history": chat_history})
            
            finish_time = time.time()
            total_time = finish_time - start_time
            ai_answer = response['answer']
            
            print("Yanıt: ")
            print(ai_answer)
            print("-" * 50)
            
            print("📑 Referans Alınan Kaynaklar:")
            kaynaklar = set([doc.metadata.get('source', 'Bilinmeyen Kaynak') for doc in response['context']])
            for kaynak in kaynaklar:
                print(f"- {kaynak}")
            
            print("=" * 50 + "\n")
            print(f"⏱️ İşlem Süresi: {total_time:.2f} saniye")
            chat_history.append(HumanMessage(content=user_question))
            chat_history.append(AIMessage(content=ai_answer))
            
        except Exception as e:
            from config import LLM_MODEL
            print(f"❌ Hata: Groq'un çalıştığından ve '{LLM_MODEL}' modelinin yüklü olduğundan emin olun. \nDetay: {e}\n")
            
if __name__ == "__main__":
    chat_start()