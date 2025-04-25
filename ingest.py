import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from supabase import create_client, Client
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verificar se as variáveis foram carregadas
if not all([supabase_url, supabase_key, openai_api_key]):
    raise ValueError("Erro: Alguma variável de ambiente não foi encontrada no arquivo .env")

# Inicializar cliente Supabase e embeddings
supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # Adiciona fallback para páginas sem texto
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def ingest_pdf_to_supabase(pdf_path):
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("Nenhum texto extraído do PDF.")
        
        chunks = split_text(text)
        embedded_chunks = embeddings.embed_documents(chunks)
        data = [
            {
                "content": chunk,
                "metadata": {"source": pdf_path, "chunk_id": i},
                "embedding": embedding
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embedded_chunks))
        ]
        response = supabase.table("documents").insert(data).execute()
        print(f"Inseridos {len(response.data)} chunks no Supabase.")
        return True
    except Exception as e:
        print(f"Erro ao processar o PDF: {e}")
        return False