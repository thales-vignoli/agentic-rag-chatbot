
# RAG Document Chat

Este projeto implementa uma solução de **Retrieval-Augmented Generation (RAG)** para perguntas e respostas baseadas em documentos PDF, utilizando Streamlit, LangChain, embeddings da OpenAI e Supabase como base vetorial.

---

## 📁 Estrutura do Projeto

```
.
├── documents/
│   └── agentic rag paper.pdf
├── venv/
├── .env
├── .gitignore
├── app.py
├── ingest.py
├── requirements.txt
└── README.md
```

---

## ✅ Pré-requisitos

- Python 3.9+
- Conta no [Supabase](https://supabase.com/)
- Chave de API da [OpenAI](https://platform.openai.com/)

---

## ⚙️ Configuração do Ambiente

```bash
# Clone o repositório
$ git clone <repo_url>
$ cd <nome_do_projeto>

# Crie um ambiente virtual
$ python3 -m venv venv
$ source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Instale as dependências
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

---

## 🔐 Variáveis de Ambiente `.env`

Crie um arquivo `.env` na raiz com o seguinte conteúdo:

```
SUPABASE_URL=<sua_url_do_supabase>
SUPABASE_KEY=<sua_chave_supabase>
OPENAI_API_KEY=<sua_chave_openai>
```

Certifique-se de que o `.env` está listado no `.gitignore`.

---

## 📄 Ingestão de Documentos

Edite o caminho no final do `ingest.py` com o local correto do seu PDF, ou ajuste diretamente:

```python
pdf_filename = "documents/agentic rag paper.pdf"
pdf_path = os.path.join("documents", pdf_filename)
```

Execute o script:

```bash
$ python ingest.py
```

Isso irá:
1. Ler e dividir o PDF em pedaços (chunks);
2. Gerar embeddings com OpenAI;
3. Inserir os dados no Supabase usando a tabela `documents`.

> Certifique-se de ter no Supabase:
> - Tabela `documents` com colunas `content`, `metadata`, `embedding`
> - Função `match_documents` para busca vetorial.

---

## 🧠 Stored Procedure no Supabase

Você pode usar essa função para realizar busca vetorial:

```sql
create or replace function match_documents(
    query_embedding float8[],
    match_count int default 5
)
returns table (
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) as similarity
    from documents d
    order by d.embedding <=> query_embedding
    limit match_count;
end;
$$;
```

---

## 🚀 Executando a Aplicação

Com o ambiente ativado e documentos já ingeridos:

```bash
$ streamlit run app.py
```

Abra o navegador em `http://localhost:8501` para acessar a interface.

---

## 🛠️ Tecnologias Utilizadas

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://platform.openai.com/)
- [Supabase](https://supabase.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

---

## 🧩 Observações Adicionais

- A função `setup_rag_chain()` em `app.py` usa `@st.cache_resource` para performance;
- O modelo padrão é `gpt-4o-mini`, mas pode ser ajustado;
- O campo `metadata` permite rastrear a origem do chunk exibido nas fontes.

---

## 📄 Licença

Este projeto está disponível sob a licença [MIT](LICENSE), sinta-se à vontade para usar, modificar e distribuir.
