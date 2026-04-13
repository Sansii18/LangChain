import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from openai import OpenAI

load_dotenv()

llm = ChatOpenAI(
    temperature = 0,
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm_transformer = LLMGraphTransformer(
    llm=llm
)

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
    """
    Loads a PDF and splits it into overlapping chunks.
    Returns a list of LangChain Document objects.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()           # one Document per page

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(pages)
    print(f"[INFO] Loaded '{pdf_path}' → {len(pages)} pages → {len(chunks)} chunks")
    return chunks