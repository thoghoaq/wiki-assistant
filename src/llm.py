import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnableLambda
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)

def get_full_doc_retriever(retriever):
    """
    A custom retriever that gets relevant chunks, then returns the full documents.
    """
    def _get_full_documents(docs):
        # Get unique source file paths from retrieved chunks
        source_files = set(doc.metadata['source'] for doc in docs)
        
        full_docs = []
        for file_path in source_files:
            try:
                # This is a simplified loader logic. 
                # For a robust solution, you might need to map extensions to loaders
                # like in your sync_vectorstore function.
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif file_ext == ".docx":
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif file_ext == ".xlsx":
                    loader = UnstructuredExcelLoader(file_path, mode="elements")
                elif file_ext == ".md":
                    loader = UnstructuredMarkdownLoader(file_path)
                elif file_ext == ".txt":
                    loader = TextLoader(file_path)
                else:
                    # Fallback for unknown types or just read as text
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    full_docs.append(Document(page_content=content, metadata={'source': file_path}))
                    continue

                # Load and combine content from all pages/parts of the document
                loaded_pages = loader.load()
                full_content = "\n".join([page.page_content for page in loaded_pages])
                # Use metadata from the first page, but confirm the source
                metadata = loaded_pages[0].metadata if loaded_pages else {}
                metadata['source'] = file_path 
                full_docs.append(Document(page_content=full_content, metadata=metadata))

            except Exception as e:
                st.warning(f"Could not read full content of {os.path.basename(file_path)}: {e}")
        
        return full_docs

    return retriever | RunnableLambda(_get_full_documents)

@st.cache_resource
def create_qa_chain(_vectorstore, api_key, model_name="gemini-1.5-flash", search_type="mmr", temperature=0.2):
    """Creates a modern conversational retrieval chain."""
    if not _vectorstore or not api_key:
        return None
    
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    
    search_kwargs = {'k': 8}
    if search_type == "mmr":
        search_kwargs['fetch_k'] = 20

    retriever = _vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Create a new retriever that returns full documents
    full_doc_retriever = get_full_doc_retriever(history_aware_retriever)

    # Answer question prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, just say "
        "that you don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(full_doc_retriever, question_answer_chain)
    
    return rag_chain
