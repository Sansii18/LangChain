from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import dotenv
from openai import OpenAI

from langchain import QuadrantVectorStore

API_KEY = dotenv.get_key(dotenv.find_dotenv(), "GEMINI_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

pdf_path = Path(__file__).parent / "LangChain-RAG.pdf"

loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load() # THIS REPRESENTS THE DOCUMENTS IN THE PDF, EACH PAGE IS A DOCUMENT BUT WE WANT TO SPLIT THEM INTO CHUNKS BASED ON THE TEXT RATHER THAN THE PAGE NUMBER AS WE DONT KNOW HOW MUCH TEXT IS THERE ON EACH PAGE => langchain.text_splitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # THE SIZE OF EACH CHUNK
    chunk_overlap=200, # THE OVERLAP BETWEEN CHUNKS TO MAINTAIN CONTEXT
)

split_docs = text_splitter.split_documents(docs) # THIS SPLITS THE DOCUMENTS INTO CHUNKS BASED ON THE TEXT RATHER THAN THE PAGE NUMBER


vector 
embedder = OpenAIEmbeddings(
    model="text-embedding-004",
    api_key=API_KEY
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)