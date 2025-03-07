# import basics
import os
from dotenv import load_dotenv

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase.client import Client, create_client

def ingest_documents():
    print("Iniciando processo de ingestão de documentos...")
    
    # load environment variables
    load_dotenv()  

    # initiate supabase db
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")  # Usando a service key para escrita
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Conectado ao Supabase")
    
    # initiate embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Modelo de embeddings inicializado")
    
    # load pdf docs from folder 'documents'
    print("Carregando documentos da pasta 'documents'...")
    loader = PyPDFDirectoryLoader("documents")
    documents = loader.load()
    print(f"Carregados {len(documents)} documentos")
    
    # split the documents in multiple chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Documentos divididos em {len(docs)} chunks")
    
    # store chunks in vector store
    print("Armazenando embeddings no Supabase...")
    vector_store = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase,
        table_name="documents",
        query_name="match_documents",
        chunk_size=1000,
    )
    print("Embeddings armazenados com sucesso!")
    
    return vector_store

if __name__ == "__main__":
    ingest_documents()