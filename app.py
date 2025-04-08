import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from supabase import create_client, Client
from dotenv import load_dotenv

# Configuração básica do Streamlit
st.set_page_config(page_title="RAG Document Chat", page_icon="📄")

# Carregar variáveis de ambiente
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Inicializar Supabase, embeddings e modelo de chat
supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini")

# Função para buscar documentos similares no Supabase
def search_documents(query):
    query_embedding = embeddings.embed_query(query)
    response = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": 5  # Retorna os 5 documentos mais similares
    }).execute()
    return [{"content": doc["content"], "metadata": doc["metadata"]} for doc in response.data]

# Criar um retriever personalizado
class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        docs = search_documents(query)
        return [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in docs]

# Configurar o prompt personalizado
prompt_template = """
Use o seguinte histórico e contexto para responder à pergunta. Se o contexto não contiver a resposta, diga "Não sei".

Histórico da conversa:
{chat_history}

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["chat_history", "context", "question"]
)

# Configurar a cadeia de RAG com histórico
@st.cache_resource  # Cache para evitar recriar a cadeia toda vez
def setup_rag_chain():
    retriever = CustomRetriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}  # Passa o prompt personalizado
    )
    return chain

# Interface no Streamlit
def main():
    st.title("RAG Document Chat")
    st.write("Faça perguntas sobre o documento e receba respostas baseadas no conteúdo.")

    # Inicializar o histórico de conversa no estado da sessão
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Exibir histórico de chat
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

    # Campo de entrada do usuário
    user_query = st.chat_input("Exemplo: Qual é o tema principal?")  # Correção aqui

    if user_query:
        # Adicionar pergunta ao histórico
        st.session_state.chat_history.append((user_query, ""))  # Placeholder para a resposta
        with st.chat_message("user"):
            st.write(user_query)

        # Configurar e executar a cadeia de RAG
        rag_chain = setup_rag_chain()
        with st.spinner("Pensando..."):
            result = rag_chain({
                "question": user_query,
                "chat_history": [(q, a) for q, a in st.session_state.chat_history[:-1]]  # Exclui a pergunta atual
            })

        # Extrair resposta e documentos de origem
        answer = result["answer"]
        source_documents = result.get("source_documents", [])

        # Atualizar o último item do histórico com a resposta
        st.session_state.chat_history[-1] = (user_query, answer)

        # Exibir a resposta
        with st.chat_message("assistant"):
            st.write(answer)

        # Exibir a origem do documento (se disponível)
        if source_documents:
            with st.expander("Fontes"):
                for doc in source_documents:
                    source_path = doc.metadata.get("source", "Desconhecido")
                    st.write(f"- {source_path}")

if __name__ == "__main__":
    main()