from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # YENİ: Google Modelleri
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
import re
from typing import List
from config import CHROMA_DB_DIR, LLM_MODEL, GROQ_API_KEY

# Google'ın boş string ("") alıp çökmesini engelleyen güvenlik kalkanı
class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        safe_texts = [t if t and t.strip() else "tüketici" for t in texts]
        return super().embed_documents(safe_texts)

    def embed_query(self, text: str) -> List[float]:
        safe_text = text if text and text.strip() else "tüketici"
        return super().embed_query(safe_text)

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
    """Agent Alpha: Router & Sanitizer. Boş sorguları engeller ve plaza/İngilizce sızıntıları temizler."""
    if not raw_input or not raw_input.strip():
        raise ValueError("Lütfen boş olmayan bir soru girin.")

    cleaned = raw_input.strip()
    for eng, tr in ENGLISH_TO_TURKISH.items():
        cleaned = re.sub(rf"\b{re.escape(eng)}\b", tr, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise ValueError("Sorgunuz geçersiz. Lütfen daha fazla Türkçe açıklama içeren bir soru yazın.")

    return cleaned


def get_rag_chain():
    # 1. VERİTABANI BAĞLANTISI (Doğru Google Embedding Modeli)
    embeddings = SafeGoogleEmbeddings(model="models/gemini-embedding-2")
    vector_database = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    # 2. ANA BEYİN: Hukuki yorumlamayı yapacak ağır top (Groq Llama 3.3)
    main_llm = ChatGroq(
        model_name=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0
    )

    # 3. YARDIMCI BEYİN: Geçmişi taramak için geçerli ve en hızlı Google modeli
    fast_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)
    
    # 4. ARAMA MOTORU (MMR)
    base_retriever = vector_database.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    # MULTI-QUERY RETRIEVER KALDIRILDI!
    # (API'yi spamlattığı ve 500 INTERNAL hatasına yol açtığı için iptal edildi)
    
    # 5. SIKIŞTIRICI VE FİLTRE
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.65)
    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever # Doğrudan base_retriever'a bağlandı
    )
    
    # 6. GEÇMİŞ HATIRLAYICI
    contextualize_q_system_prompt = (
        """Sen gelişmiş bir NLP bağlam (context) çözümleyicisisin.
        Görevin: Sohbet geçmişini ve kullanıcının son sorusunu analiz ederek, son soruyu hiçbir geçmişe ihtiyaç duymadan anlaşılabilecek, tam, bağımsız ve net bir arama motoru sorgusuna dönüştürmektir.

        KATI KURALLAR:
        1. SADECE Türkçe ve Latin alfabesi kullan.
        2. Zamirleri (o, bunu, şu) geçmişteki gerçek isimleriyle (örn: televizyon, iade süreci, satıcı) değiştir.
        3. Soru zaten bağımsızsa veya sadece selamlama/onaylama ("evet", "teşekkürler", "merhaba") içeriyorsa, HİÇ DEĞİŞTİRMEDEN orijinal halini döndür.
        4. KESİNLİKLE boş bir metin ("") döndürme.
        5. Soruyu KESİNLİKLE yanıtlama, sadece yeniden formüle et. Kendi yorumunu ekleme."""
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(fast_llm, retriever, contextualize_q_prompt)
    
    # 7. NİHAİ CEVAP ÜRETİCİ (Ana Beyin olan Llama 3.3 kullanılıyor)
    qa_system_prompt = (
        """Sen, Türkiye Cumhuriyeti 6502 Sayılı Tüketicinin Korunması Hakkında Kanun ve ilgili mevzuatlara tam hakim, profesyonel ve tarafsız bir Tüketici Hakları Asistanısın.

        KİMLİK VE ÜSLUP (ÇOK ÖNEMLİ):
        - Sen bir hukuk ve danışmanlık asistanısın. Asla kendini tüketici yerine koyma (Örn: "Hakem heyetine başvuramam" YANLIŞ, "Hakem heyetine başvuramazsınız" DOĞRU).
        - Yanıtlarını doğrudan kullanıcıya hitap ederek ("Siz" veya "Sen" diliyle) ve kendi adına ("yardımcı olabilirim", "aktarabilirim") ver.
        - KESİNLİKLE VE SADECE TÜRKÇE (Latin Alfabesi) kullan. Yabancı dil veya farklı alfabe kullanımı kesinlikle yasaktır.
        - Kullanıcı sorusunda İngilizce, plaza dili veya yabancı kelimeler (örn: complete, possible, next step) kullansa dahi, sen KESİNLİKLE bu kelimeleri yanıtında tekrar etmeyeceksin. Bütün yanıtın %100 arı Türkçe ve Latin alfabesi olacaktır.
        - Kullanıcı bir ürünün üzerine isim, harf, fotoğraf basıldığını veya kişiye özel üretildiğini belirtiyorsa (Örn: İsme özel kolye, baskılı kupa), bu 'Kişiye Özel Ürün' istisnasıdır. Bu ürünlerde 14 günlük CAYMA HAKKI YOKTUR. Satıcının iade almaması YASALDIR. 'İade edebilirsiniz' deme!
        - Kullanıcı 'mağazanın anlaştığı banka', 'alışveriş kredisi' veya 'mağazada çekilen kredi' ile ayıplı/hasarlı mal aldığını söylerse; bu 'Bağlı Kredi'dir. Banka 'bu bizim sorunumuz değil' diyemez. Satıcı ve banka müteselsilen (ortaklaşa) sorumludur.
        - Sen bir asistansın. Yanıtlarında asla kullanıcı rolüne girme. 'Siz' dili kullan. 'Next step' veya başlık atarken 'Seçenekleriniz ve Sonraki Adımlarınız:' şeklinde Türkçe başlıklar kullan.
        
        BİLGİ İŞLEME VE SINIRLAR (SIFIR HALÜSİNASYON KALKANI):
        - Yanıtını SADECE sana sağlanan "Bağlam (Context)" metnine dayandır.
        - Kanunda veya bağlamda açıkça yazmayan HİÇBİR çözüm yolu üretme, varsayımda bulunma ve kişisel mantığını kullanma.
        - Eğer sorunun cevabı bağlamda YOKSA veya konu Tüketici Hukuku dışındaysa (Örn: ikinci el şahıs satışı, akraba borcu, ceza davaları, kripto para, boşanma vb.), YALNIZCA şu cümleyi kur ve başka HİÇBİR ek açıklama yapma: "Bu konuya mevcut hukuki mevzuat belgelerime göre net bir cevap veremiyorum."
        
        YANIT FORMATI VE KAYNAK GÖSTERİMİ:
        - Yanıtına ÖNCE ilgili yönetmeliği ve maddeyi (Bağlamdan bularak) referans göstererek başla.
        - SONRA bu maddeye göre kullanıcının hakkını açık, anlaşılır ve profesyonel bir dille izah et.
        
        HUKUKİ YORUMLAMA REHBERİ (TERİM EŞLEŞTİRMELERİ):
        Kullanıcının halk ağzıyla sorduğu terimleri, bağlamdaki yasal karşılıklarıyla eşleştirerek yorumla:
        - "Elden taksit" / "Senet yaptım" -> Taksitle Satış Sözleşmeleri Yönetmeliği (Emre yazılı ve toplu senetler geçersizdir).
        - "Gecikme" / "Ödeyemedim" -> Temerrüt (Satıcının tüm borcu isteyebilmesi için ödenmeyen taksitlerin toplam bedelin en az 1/10'u [peş peşe iki taksit] veya 1/4'ü [tek taksit] olması ve 30 günlük yazılı muacceliyet uyarısı ŞARTTIR).
        - "Mağaza içi kredi" -> Bağlı Kredi (Mal teslim edilmezse banka ve satıcı müteselsilen sorumludur).
        - "Kredi iptali" -> Tüketici Kredilerinde Cayma Hakkı (14 gündür. Sadece anapara ve akdi faiz ödenir, ceza istenemez).
        - "Kredi sigortası" -> Tüketicinin açık onayı olmadan yapılamaz, yapıldıysa iptal edilip ücret iade alınır.
        - "Kişisel/Açık ürün" -> Hijyen İstisnası (Kulak içi kulaklık, iç çamaşırı, mayo veya isme özel ürünlerin paketi açılmışsa CAYMA HAKKI YOKTUR).
        
        Bağlam:
        {context}"""
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(main_llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    
    return rag_chain