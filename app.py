import os
import streamlit as st
from dotenv import load_dotenv

# import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# import supabase
from supabase.client import Client, create_client

# Carregando variáveis de ambiente
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(page_title="ChatPDF - Conversa com documentos", page_icon="📚")
st.title("💬 Conversa com seu documento PDF")

# Inicializar conexão com Supabase (apenas para consulta)
@st.cache_resource
def init_supabase():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")  # Usando anon key para consultas
    return create_client(supabase_url, supabase_key)

# Inicializar o modelo de embeddings
@st.cache_resource
def init_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Inicializar o modelo de linguagem
@st.cache_resource
def init_llm():
    return ChatOpenAI(
        temperature=0.2,
        model_name="gpt-3.5-turbo"
    )

# Conectar ao vector store no Supabase
@st.cache_resource
def get_vector_store():
    supabase = init_supabase()
    embeddings = init_embeddings()
    
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )
    
    return vector_store

# Inicializar a cadeia de conversa
@st.cache_resource
def get_conversation_chain():
    vector_store = get_vector_store()
    llm = init_llm()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
    )
    
    return conversation_chain

# Inicializar o histórico de conversa na sessão se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens anteriores do histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Inicializar a cadeia de conversa
conversation = get_conversation_chain()

# Campo de entrada para nova mensagem
if prompt := st.chat_input("Faça uma pergunta sobre o documento..."):
    # Adicionar pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Exibir pergunta do usuário
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gerar resposta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Obter resposta do modelo
            response = conversation({"question": prompt})
            answer = response['answer']
            
            # Exibir resposta
            st.markdown(answer)
            
            # Se quiser mostrar as fontes, descomente as linhas abaixo
            # if response["source_documents"]:
            #     with st.expander("Fontes"):
            #         for i, doc in enumerate(response["source_documents"]):
            #             st.write(f"Fonte {i+1}:")
            #             st.write(doc.page_content)
            #             st.write("---")
    
    # Adicionar resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Adicionar informações na sidebar
with st.sidebar:
    st.subheader("Sobre")
    st.write("""
    Este aplicativo permite conversar com o documento PDF armazenado na pasta 'documents'.
    Os embeddings já foram criados usando o script de ingestão.
    """)
    
    st.subheader("Instruções")
    st.write("""
    1. Faça perguntas sobre o conteúdo do documento
    2. O sistema buscará as partes relevantes e responderá com base nelas
    3. O histórico da conversa é mantido para contexto
    """)