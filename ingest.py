# ingest.py
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from supabase import create_client, Client
from dotenv import load_dotenv

# Carregar variáveis de ambiente
env_path = os.path.join(os.path.dirname(__file__), ".env")
print(f"Tentando carregar o arquivo .env de: {env_path}")
load_dotenv(env_path)

# Obter variáveis de ambiente com verificação
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Depuração: exibir valores carregados (sem exibir chaves completas)
print(f"SUPABASE_URL carregado: {supabase_url[:20] if supabase_url else 'Não encontrado'}")
print(f"SUPABASE_KEY carregado: {supabase_key[:20] if supabase_key else 'Não encontrado'}")
print(f"OPENAI_API_KEY carregado: {openai_api_key[:20] if openai_api_key else 'Não encontrado'}")

# Verificar se as variáveis foram carregadas
if not supabase_url:
    raise ValueError("Erro: SUPABASE_URL não foi encontrado no arquivo .env")
if not supabase_key:
    raise ValueError("Erro: SUPABASE_KEY não foi encontrado no arquivo .env")
if not openai_api_key:
    raise ValueError("Erro: OPENAI_API_KEY não foi encontrado no arquivo .env")

# Inicializar cliente Supabase e embeddings
supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def ingest_pdf_to_supabase(pdf_path):
    text = extract_text_from_pdf(pdf_path)
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

if __name__ == "__main__":
    pdf_filename = "/Users/thalesvignoli/development/agentic-rag-chatbot/agentic-rag-chatbot/documents/agentic rag paper.pdf"  # Substitua pelo nome real do seu PDF
    pdf_path = os.path.join("documents", pdf_filename)
    ingest_pdf_to_supabase(pdf_path)