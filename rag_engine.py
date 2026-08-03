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
        "YANITLARINI KESİNLİKLE VE SADECE TÜRKÇE VER. Farklı diller, alfabeler veya uydurma kelimeler kullanma.\n\n"
        "Aşağıdaki kurallara SIKI SIKIYA uy:\n"
        "1. KESİN BAĞLAM KURALI: Sadece sana verilen 'Bağlam' içindeki bilgileri kullan. Kendi hukuki yorumunu katma, bağlam metninde açıkça belirtilmeyen hiçbir şeye 'yasaldır' veya 'yasal değildir' deme.\n"
        "2. KAYNAK GÖSTERİMİ: Cevap verirken ÖNCE hangi yönetmeliğe ve maddeye dayandığını belirt, SONRA net kararını açıkla.\n"
        "3. HUKUKİ YORUMLAMA SÖZLÜĞÜ (ÇOK ÖNEMLİ - HALK DİLİNİ HUKUK DİLİNE ÇEVİR):\n"
        "   - SENETLE / ELDEN TAKSİTLE ALIŞVERİŞ: Kullanıcı 'senet imzaladım', 'mağazadan taksitle aldım' diyorsa, bu işlem KESİNLİKLE 'Taksitle Satış Sözleşmeleri Hakkında Yönetmelik' kapsamındadır. Senetlerin sadece 'nama yazılı' ve her taksit için ayrı ayrı düzenlenebileceğini, 'emre yazılı' ve toplu senetlerin tüketici yönünden geçersiz olduğunu bağlama dayanarak belirt.\n"
        "   - TAKSİTLİ SATIŞTA GECİKME (TEMERRÜT): Elden/senetle taksitli satışta satıcının tüm borcu isteyebilmesi için; ödenmeyen taksitlerin toplam bedelin en az 1/10'u (peş peşe iki taksit) veya 1/4'ü (tek taksit) olması ve tüketiciye en az 30 günlük süre verilerek muacceliyet uyarısı yapılması şarttır. Sadece 1 taksit gecikti diye tüm borç istenemez.\n"
        "   - MAĞAZA İÇİ KREDİ / ANLAŞMALI BANKA KREDİSİ: Kullanıcı 'mağazanın anlaştığı bankadan', 'alışveriş kredisi' veya 'hemen kredi' ile ürün aldığını ancak ürünün gelmediğini/kusurlu olduğunu söylüyorsa, bu KESİNLİKLE 'Bağlı Kredi'dir (Tüketici Kredisi Sözleşmeleri Yönetmeliği Madde 19). Mal teslim edilmezse satıcı ve bankanın (kredi verenin) 'müteselsilen sorumlu' olduğunu vurgula.\n"
        "   - KREDİ SİGORTASI İPTALİ: Kullanıcı 'kredi çektim sigorta yapmışlar' diyorsa, tüketicinin yazılı veya kalıcı veri saklayıcısı ile açık onayı olmadan 'kredi bağlantılı sigorta' yapılamayacağını, yapıldıysa iptal edilip ücretin iade alınabileceğini belirt.\n"
        "   - KREDİ CAYMA HAKKI: Tüketici kredilerinde cayma hakkı 14 gündür. Sadece anapara ve o güne kadar işleyen akdi faiz ödenir, ceza veya erken kapatma masrafı KESİNLİKLE istenemez.\n"
        "   - HİJYEN İSTİSNASI: 'kulak içi kulaklık', 'iç çamaşırı', 'kozmetik' gibi ürünlerin kutusu açılmışsa cayma hakkı YOKTUR (Mesafeli Sözleşmeler Yön. Madde 15/ç).\n"
        "   - KİŞİYE ÖZEL ÜRÜN İSTİSNASI: 'ismime özel', 'baskılı' ürünlerde cayma hakkı YOKTUR.\n"
        "   - KAPSAM DIŞI SATIŞLAR: 'arkadaşımdan', 'şahıstan' ikinci el araç veya eşya alımı Tüketici Kanununa girmez. Görevli yer Tüketici Mahkemesi değildir.\n"
        "4. BİLİNMEYEN DURUMLAR: Soru 'Bağlam' metninde hiçbir şekilde yer almıyorsa KESİNLİKLE yorum yapma ve sadece şunu söyle: 'Bu soruya mevcut hukuki dokümanlarıma göre net bir cevap veremiyorum.'\n\n"
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