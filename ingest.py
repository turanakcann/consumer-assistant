import os
import re
import warnings

# Terminali kirleten o sarı uyarıyı (DeprecationWarning) tamamen susturuyoruz.
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from config import DOCS_DIR, EMBEDDING_MODEL, CHROMA_DB_DIR



def ingest_document(file_path):
    print(f"Belgeler okunuyor: {file_path}")
    loader = PyPDFDirectoryLoader(file_path)
    pages = loader.load()
    
    file_contents = {} # Dosya içerikleri
    for page in pages:
        source = page.metadata.get('source', "Bilinmeyen_kaynak") # Metadata'dan kaynak çekmeyi deniyoruz. Kaynak yoksa Bilinmeyen_kaynak atanıyor
        if source not in file_contents:
            file_contents[source] = ""
        file_contents[source] += page.page_content + "\n"
        
    documents = []
    for source, full_text in file_contents.items(): # Dict'deki kaynak ve tam metni regex sayesinde maddeler halinde bölüyoruz.
        clauses = re.split(r'(?i)(?=MADDE\s+\d+)', full_text)
        for clause in clauses:
            if len(clause.strip()) > 20:
                documents.append(
                    Document(
                        page_content=clause.strip(),
                        metadata={"source": os.path.basename(source)}
                    )
                )
                
    return documents

def prepare_database(): # Veritabanına yazacak
    documents = ingest_document(DOCS_DIR) # Yazdığımız fonksiyonu çağırıyoruz.
    
    if not documents:
        print("Hata: Klasörde okunacak PDF yok veya maddeler ayrılamadı!")
        return
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vector_database = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings   
    )
    
    batch_size = 50
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Vektörler işleniyor", unit="batch"):
        batch = documents[i:i + batch_size]
        vector_database.add_documents(batch)
        
    print(f"\nBaşarılı: Vektörleştirme tamamlandı. Veriler {CHROMA_DB_DIR} dizinine kaydedildi.")
    
if __name__ == "__main__":
    prepare_database()