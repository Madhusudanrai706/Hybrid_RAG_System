import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------------------------------------------
# Embedding Model (Load Once)
# ---------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------
# Load PDF
# ---------------------------------------------------

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    return loader.load()


# ---------------------------------------------------
# Split Documents
# ---------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(documents)


# ---------------------------------------------------
# Create Vector Store
# ---------------------------------------------------

def create_vector_store(chunks):

    if not os.path.exists("chroma_db"):
        os.makedirs("chroma_db")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db",
        collection_name="hybrid_rag"
    )

    return db


# ---------------------------------------------------
# Load Existing Vector Store
# ---------------------------------------------------

def load_vector_store():

    db = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model,
        collection_name="hybrid_rag"
    )

    return db