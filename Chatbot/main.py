import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
from data import documents
import os

# Load Hugging Face API key from environment variable
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
if not HUGGINGFACE_API_KEY:
    raise RuntimeError('HUGGINGFACE_API_KEY environment variable is not set. Please set it to your Hugging Face API token.')

# Initialize embeddings and FAISS DB
@st.cache(allow_output_mutation=True)
def initialize_rag():
    docs = [Document(page_content=text) for text in documents]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_db = FAISS.from_documents(docs, embedding_model)
    return faiss_db

faiss_db = initialize_rag()

# LLM setup
@st.cache(allow_output_mutation=True)
def get_chat_model():
    llm = HuggingFaceEndpoint(repo_id="microsoft/Phi-3.5-mini-instruct", task="text_generation")
    return ChatHuggingFace(llm=llm)

chat_model = get_chat_model()

# Retrieval + Generation function
def retrieve_docs(query, k=5):
    return faiss_db.similarity_search(query, k)

def ask(query, context_docs):
    context_text = "\n".join([doc.page_content for doc in context_docs])
    messages = [
        SystemMessage(content="You are a scientist"),
        HumanMessage(content=f"Answer the question: {query}\nBased on the context:\n{context_text}")
    ]
    response = chat_model.invoke(messages)
    return response.content

def ask_rag(query):
    context = retrieve_docs(query)
    return ask(query, context)

# Streamlit UI
st.title(" Welcome to PharaohGuide Chatbot")

query = st.text_input("Ask a question:")
if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        response = ask_rag(query)
        st.markdown("### 💬 Answer:")
        st.write(response)
