from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_db():
    print("PDF load ho raha hai...")
    loader = PyPDFLoader("syllabus.pdf") 
    docs = loader.load()
    print(f"Total pages loaded: {len(docs)}")

    print("Data chunks mein divide ho raha hai...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Total chunks ban gaye: {len(chunks)}")

    if len(chunks) == 0:
        print("❌ ERROR: PDF mein koi text nahi mila.")
        return

    print("HuggingFace se Local Embeddings generate ho rahi hain (Network ki zaroorat nahi!)...")
    # Yeh open-source local model hai, kabhi timeout nahi hoga!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("FAISS Index ban raha hai...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    vector_store.save_local("faiss_index")
    print("✅ Success! Vector DB successfully ban gaya hai aur save ho gaya hai.")

if __name__ == "__main__":
    create_vector_db()