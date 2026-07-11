import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. API Key Setup (Apni asli key daalein)
# API Key ko ab hum streamlit secrets se lenge
import streamlit as st
# Temporary fix taaki project chale
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


st.set_page_config(page_title="College AI Assistant", page_icon="🎓", layout="centered")
st.title("🎓 College AI Assistant")
st.write("Syllabus, notices aur college rules ke baare mein kuch bhi pucho!")

@st.cache_resource
def load_rag_components():
    # Humara local database aur model jo perfectly kaam kar raha hai
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # API Key ko direct parameter ke roop mein pass karo
    llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    temperature=0.3, 
    google_api_key="api_key"
)
    return vector_store, llm

vector_store, llm = load_rag_components()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Apna sawaal yahan likho (e.g., Syllabus mein kaunse topics hain?)...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Database mein dhoondh raha hoon..."):
            try:
                # PRO BYPASS: Custom RAG Logic (No langchain.chains needed!)
                
                # Step A: FAISS database se top 3 relevant chunks dhoondho
                docs = vector_store.similarity_search(user_query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Step B: LLM ke liye custom prompt banao
                prompt = f"Tum ek helpful College Assistant ho. Niche diye gaye PDF context ko padho aur user ke sawaal ka detail mein jawab do. Agar answer context mein nahi hai toh bol do ki 'Yeh information syllabus mein nahi hai'.\n\nContext:\n{context}\n\nSawaal: {user_query}\n\nJawab:"
                
                # Step C: Gemini se directly answer generate karwao
                response = llm.invoke(prompt)
                answer = response.content
                if isinstance(answer, list):
                  answer = answer[0]['text']
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error aaya: {e}")