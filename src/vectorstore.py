import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import DB_DIR, DOCS_DIR

@st.cache_resource(show_spinner="Loading existing vector database...")
def load_vectorstore(api_key):
    """Loads the existing vectorstore from disk."""
    if not api_key:
        return None
    
    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    if os.path.exists(DB_DIR):
        try:
            return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        except Exception as e:
            st.error(f"Failed to load database: {e}. You may need to delete the '{DB_DIR}' folder and re-index.")
            return None
    return None

def sync_vectorstore(vectorstore, api_key):
    """
    Synchronizes the vectorstore with the documents folder.
    - Adds new documents not yet in the DB.
    Returns the updated vectorstore and a boolean indicating if changes were made.
    """
    if not api_key:
        st.error("API Key is required for synchronization.")
        return vectorstore, False

    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    indexed_doc_names = set(get_indexed_documents(vectorstore))
    
    disk_docs_names = set()
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        
    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if file.lower().endswith((".pdf", ".txt", ".docx", ".xlsx", ".md")):
                disk_docs_names.add(file)

    docs_to_add_names = disk_docs_names - indexed_doc_names
    
    print("--- DEBUG ---")
    print(f"Docs on disk ({len(disk_docs_names)}): {disk_docs_names}")
    print(f"Indexed docs ({len(indexed_doc_names)}): {indexed_doc_names}")
    print(f"Docs to add ({len(docs_to_add_names)}): {docs_to_add_names}")
    print("--- END DEBUG ---")

    changed = False

    # Add new documents that are on disk but not in the DB
    if docs_to_add_names:
        st.info(f"Found {len(docs_to_add_names)} new document(s) to add...")
        docs_to_load_paths = [os.path.join(DOCS_DIR, name) for name in docs_to_add_names]
        
        new_documents = []
        for file_path in docs_to_load_paths:
            file_ext = os.path.splitext(file_path)[1].lower()
            try:
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
                    continue # Skip unsupported file types
                
                new_documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading file {file_path}: {e}")
        
        if new_documents:
            # Filter out complex metadata that ChromaDB can't handle
            filtered_documents = filter_complex_metadata(new_documents)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
            docs = text_splitter.split_documents(filtered_documents)
            
            if vectorstore is None:
                vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
            else:
                vectorstore.add_documents(docs)
            
            vectorstore.persist()
            st.success(f"Added {len(docs_to_add_names)} new document(s) to the database.")
            changed = True

    if not changed:
        st.info("Database is already up-to-date.")

    return vectorstore, changed

def get_indexed_documents(vectorstore):
    """Gets a unique list of source documents from the vectorstore."""
    if not vectorstore:
        return []
    try:
        all_docs = vectorstore.get(include=["metadatas"])
        if not all_docs or not all_docs.get('metadatas'):
            return []
        
        sources = set()
        for meta in all_docs['metadatas']:
            source = meta.get('source')
            if source:
                # Normalize the path and get the base name
                normalized_source = os.path.basename(os.path.normpath(source))
                sources.add(normalized_source)
        
        return sorted(list(sources))
    except Exception:
        return []
