from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from config import CHROMA_DB_DIR, EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, 
                                       model_kwargs={'device':'cuda'})
    
    vector_database = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )
    
    llm = ChatGroq(
        model_name=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.0
    )
    
    base_retriever = vector_database.as_retriever(
        search_type="mmr", 
        search_kwargs={"k":5, "fetch_k":20}
        )
    
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    """embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.60)
    
    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=mq_retriever
    )"""
    
    contextualize_q_system_prompt = (
        "Sohbet geçmişine ve kullanıcının en son sorusuna bak. "
        "Eğer kullanıcı eksik bir soru sormuşsa (örneğin 'peki bu madde nedir?', 'şartları neler?' gibi), "
        "geçmişteki konuyu kullanarak bu soruyu tek başına anlaşılabilecek bağımsız bir soruya dönüştür. "
        "DİKKAT: Bu aşamada soruyu KESİNLİKLE cevaplama, sadece yeniden yazıp bırak."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"), # Geçmiş mesajlar buraya gelecek
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_system_prompt = (
        "Sen profesyonel, tarafsız ve sadece Tüketici Mevzuatına göre çalışan bir Tüketici Hakları asistanısın. "
        "YANITLARINI KESİNLİKLE VE SADECE TÜRKÇE VER. Farklı diller, alfabeler (Örn: Çince, Japonca) veya uydurma kelimeler kullanma.\n\n"
        "Aşağıdaki kurallara SIKI SIKIYA uy:\n"
        "1. KESİN BAĞLAM KURALI: Sadece sana verilen 'Bağlam' içindeki bilgileri kullan. Bağlamda yoksa uydurma veya kendi mantığını katma.\n"
        "2. KAYNAK GÖSTERİMİ: Cevap verirken ÖNCE hangi yönetmeliğe ve maddeye (Örn: Mesafeli Sözleşmeler Yönetmeliği Madde 15) dayandığını belirt, SONRA net kararını açıkla.\n"
        "3. HUKUKİ YORUMLAMA KÖPRÜSÜ (ÇOK ÖNEMLİ):\n"
        "   - HİJYEN İSTİSNASI: Kullanıcı 'kulak içi kulaklık', 'iç çamaşırı', 'mayo', 'kozmetik' gibi cilde doğrudan temas eden ürünlerin kutusunu açtığını, denediğini veya kullandığını söylerse; bunu otomatik olarak 'tesliminden sonra ambalaj, bant, mühür, paket gibi koruyucu unsurları açılmış olan mallardan; iadesi sağlık ve hijyen açısından uygun olmayanlar' (Madde 15/ç) kapsamında değerlendir ve cayma hakkını KESİNLİKLE REDDET.\n"
        "   - KİŞİYE ÖZEL ÜRÜN İSTİSNASI: Kullanıcı 'ismime özel', 'fotoğraf baskılı', 'kupa bardak' gibi kişiselleştirilmiş ürünlerden bahsediyorsa; bunu 'tüketicinin istekleri veya kişisel ihtiyaçları doğrultusunda hazırlanan mallar' (Madde 15/b) kapsamında değerlendir ve cayma hakkını KESİNLİKLE REDDET.\n"
        "   - KAPSAM DIŞI SATIŞLAR: Kullanıcı 'arkadaşımdan', 'komşumdan' gibi şahıstan şahısa (ikinci el otomobil vb.) satışlardan bahsediyorsa; Tüketici Kanununun ticari satıcılar ile geçerli olduğunu, şahsi satışların kapsama girmediğini belirt.\n"
        "   - AYIPLI MAL MASRAFLARI: Bozulan/arızalanan bir ürünün değişimi veya onarımı talep ediliyorsa (ayıplı mal), kargo ve nakliye masraflarının KESİNLİKLE satıcıya ait olduğunu vurgula.\n"
        "4. BİLİNMEYEN DURUMLAR: Eğer soru 'Bağlam' metninde hiçbir şekilde yer almıyorsa veya yorum yapmanı gerektirecek kadar alakasızsa, KESİNLİKLE açıklama yapma ve sadece şunu söyle: 'Bu soruya mevcut hukuki dokümanlarıma göre net bir cevap veremiyorum.'\n\n"
        "Bağlam:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )
    
    return rag_chain