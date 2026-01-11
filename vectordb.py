from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docs = []
for file in os.listdir("data/"):
    loader = TextLoader(f"data/{file}")
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
chunks = splitter.split_documents(docs)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

# try:
#     from langchain_core import __version__ as core_v
#     from langchain_huggingface import HuggingFaceEmbeddings
#     from langchain_openai import OpenAIEmbeddings
#     print(f"✅ Success! Core version: {core_v}")
#     print("✅ HuggingFace and OpenAI imports are both working.")
# except ImportError as e:
#     print(f"❌ Still an issue: {e}")