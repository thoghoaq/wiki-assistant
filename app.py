import streamlit as st
import os
import shutil
import sys
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv, set_key
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader, 
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.schema.runnable import RunnableLambda
from langchain.schema import Document
# from langchain_core.prompts import ChatPromptTemplate -> This is already imported
from langchain_core.runnables import RunnablePassthrough

# --- Load .env file ---
load_dotenv()
ENV_PATH = ".env"

# --- Configuration and Initialization ---
DB_DIR = "./chroma_db"
DOCS_DIR = "./documents"
CHAT_SESSIONS_DIR = "./chat_sessions"

# --- Functions for Chat History ---
def generate_chat_title(messages, api_key=None):
    """
    Generates a title for a chat session from the user's first message.
    """
    if not messages:
        return "New Chat"

    user_question = messages[0]['content']
    title = (user_question[:50] + '...') if len(user_question) > 50 else user_question
    return title

def get_session_list():
    """
    Returns a list of saved chat sessions, sorted by modification time.
    Also handles migrating old-format chat files.
    """
    if not os.path.exists(CHAT_SESSIONS_DIR):
        os.makedirs(CHAT_SESSIONS_DIR)
    
    sessions = []
    files = [f for f in os.listdir(CHAT_SESSIONS_DIR) if f.endswith(".json")]
    
    for f in files:
        file_path = os.path.join(CHAT_SESSIONS_DIR, f)
        session_id = f.replace('.json', '')
        try:
            with open(file_path, 'r+', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                except json.JSONDecodeError:
                    continue # Skip corrupted or empty files

                display_name = f"Chat {session_id[:8]}"
                
                # Check for old list format and migrate
                if isinstance(data, list):
                    new_data = {"display_name": display_name, "messages": data}
                    file.seek(0)
                    json.dump(new_data, file, indent=4)
                    file.truncate()
                elif isinstance(data, dict):
                    display_name = data.get("display_name", display_name)

                mod_time = os.path.getmtime(file_path)
                sessions.append({"id": session_id, "display_name": display_name, "mod_time": mod_time})

        except Exception as e:
            print(f"Error processing session file {f}: {e}")
            continue
            
    # Sort by modification time, newest first
    sessions.sort(key=lambda x: x["mod_time"], reverse=True)
    return sessions

def load_chat_history(session_id):
    """Loads chat history from a JSON file, handling both old and new formats."""
    file_path = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("messages", [])
                elif isinstance(data, list):
                    return data # Old format
            except json.JSONDecodeError:
                return [] # Return empty list for corrupted files
    return []

def save_chat_history(session_id, history, display_name=None):
    """Saves chat history and display name to a JSON file."""
    if not os.path.exists(CHAT_SESSIONS_DIR):
        os.makedirs(CHAT_SESSIONS_DIR)
        
    file_path = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")
    
    # If display name is not provided, try to read the existing one
    if display_name is None:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    display_name = data.get("display_name", "New Chat")
                except json.JSONDecodeError:
                    display_name = "New Chat"
        else:
            display_name = "New Chat"

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump({"display_name": display_name, "messages": history}, f, indent=4)

def generate_session_id():
    """Generates a unique session ID."""
    return str(uuid.uuid4())

def get_session_display_name(session_file):
    """DEPRECATED: This function is no longer the primary way to get display names."""
    try:
        # Assumes filename is a UUID
        timestamp = os.path.getmtime(os.path.join(CHAT_SESSIONS_DIR, session_file))
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return session_file.replace('.json', '')

# --- Functions for Database and Document Handling ---

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
    - Removes documents from the DB that are no longer in the folder.
    Returns the updated vectorstore and a boolean indicating if changes were made.
    """
    if not api_key:
        st.error("API Key is required for synchronization.")
        return vectorstore, False

    os.environ["GOOGLE_API_KEY"] = api_key
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    indexed_docs_paths = get_indexed_documents(vectorstore)
    indexed_doc_names = {os.path.basename(p) for p in indexed_docs_paths}
    
    disk_docs_names = set()
    for root, _, files in os.walk(DOCS_DIR):
        for file in files:
            if file.lower().endswith((".pdf", ".txt", ".docx", ".xlsx", ".md")):
                disk_docs_names.add(file)

    docs_to_add_names = disk_docs_names - indexed_doc_names
    docs_to_remove_paths = [p for p in indexed_docs_paths if os.path.basename(p) not in disk_docs_names]
    
    changed = False

    # Remove documents that are no longer on disk
    if docs_to_remove_paths:
        st.info(f"Found {len(docs_to_remove_paths)} document(s) to remove...")
        ids_to_delete = []
        for doc_path in docs_to_remove_paths:
            docs_found = vectorstore.get(where={"source": doc_path})
            ids_to_delete.extend(docs_found.get('ids', []))
        
        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            vectorstore.persist()
            st.success("Removed stale documents from the database.")
            changed = True

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
def create_qa_chain(_vectorstore, api_key):
    """Creates a modern conversational retrieval chain."""
    if not _vectorstore or not api_key:
        return None
    
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 8, 'fetch_k': 20}
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

def get_indexed_documents(vectorstore):
    """Gets a unique list of source documents from the vectorstore."""
    if not vectorstore:
        return []
    try:
        all_docs = vectorstore.get(include=["metadatas"])
        if not all_docs or not all_docs.get('metadatas'):
            return []
        sources = set(meta.get('source') for meta in all_docs['metadatas'] if meta.get('source'))
        return sorted(list(sources))
    except Exception:
        return []

# --- Streamlit UI ---
st.set_page_config(page_title="Local Document Chatbot")
st.title("📄 Local Document Chatbot with Gemini AI")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    
    gemini_api_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        key="gemini_api_input",
        value=os.getenv("GOOGLE_API_KEY", "")
    )

    if gemini_api_key and gemini_api_key != os.getenv("GOOGLE_API_KEY"):
        set_key(ENV_PATH, "GOOGLE_API_KEY", gemini_api_key)
        st.success("API Key saved for future sessions.")
    
    st.markdown(f"**Document Folder:** `{DOCS_DIR}`")
    st.markdown(f"**Database Folder:** `{DB_DIR}`")

    if st.button("🔄 Refresh Database", help="Deletes the current database and re-indexes all documents from the 'documents' folder."):
        st.session_state['force_resync'] = True
        st.rerun()

    st.header("Chat Sessions")
    if st.button("➕ New Chat"):
        st.session_state.session_id = generate_session_id()
        st.session_state.messages = []
        # Explicitly save a new empty chat file
        save_chat_history(st.session_state.session_id, [], display_name="New Chat")
        st.rerun()

    sessions = get_session_list()
    if sessions:
        session_mapping = {s["display_name"]: s["id"] for s in sessions}
        display_names = list(session_mapping.keys())
        
        # Find the index of the current session
        current_session_id = st.session_state.get("session_id")
        current_session_info = next((s for s in sessions if s["id"] == current_session_id), None)
        
        current_index = 0
        if current_session_info:
            try:
                current_index = display_names.index(current_session_info["display_name"])
            except ValueError:
                # If the display name isn't in the list (e.g., just changed), default to top
                pass

        selected_display_name = st.selectbox(
            "Select a conversation:",
            options=display_names,
            index=current_index,
            key="session_selector"
        )
        
        selected_session_id = session_mapping[selected_display_name]

        # If user selects a different session, load it
        if selected_session_id != st.session_state.get("session_id"):
            st.session_state.session_id = selected_session_id
            st.session_state.messages = load_chat_history(selected_session_id)
            st.rerun()

    # Placeholder for the document list
    st.session_state.indexed_docs_placeholder = st.empty()
    st.session_state.indexed_docs_placeholder.header("Indexed Documents")
    # This section will now update automatically after sync
    if 'vectorstore' in st.session_state and st.session_state['vectorstore']:
        indexed_docs = get_indexed_documents(st.session_state['vectorstore'])
        if indexed_docs:
            for doc_path in indexed_docs:
                st.text(os.path.basename(doc_path))
        else:
            st.info("No documents currently in the database.")
    else:
        st.info("Database not initialized.")

# --- Main App Logic ---

# Initialize session state for session management
if "session_id" not in st.session_state:
    sessions = get_session_list()
    if sessions:
        st.session_state.session_id = sessions[0]['id']
        st.session_state.messages = load_chat_history(sessions[0]['id'])
    else:
        st.session_state.session_id = generate_session_id()
        st.session_state.messages = []
        # Create the initial file for the new session
        save_chat_history(st.session_state.session_id, [], display_name="New Chat")

# Handle forced re-sync first
if st.session_state.get('force_resync', False):
    # Clear vectorstore from session state to release file locks
    st.session_state['vectorstore'] = None
    # Clear the cache for the loader function as well
    load_vectorstore.clear()
    
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            st.success(f"Deleted existing database in '{DB_DIR}'.")
        except Exception as e:
            st.error(f"Error deleting database: {e}")

    st.session_state['force_resync'] = False # Reset the flag
    st.rerun() # Rerun to reload everything cleanly

# Load vectorstore from disk if not in session state
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = load_vectorstore(gemini_api_key)

# Automatically synchronize the database with the documents folder
if gemini_api_key:
    # Ensure documents folder exists
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # The sync_vectorstore function is now the single point of truth for DB changes
    st.session_state['vectorstore'], changed = sync_vectorstore(st.session_state.get('vectorstore'), gemini_api_key)
    
    # If changes were made, or if the QA chain doesn't exist, recreate it and rerun
    if changed or 'qa_chain' not in st.session_state or st.session_state['qa_chain'] is None:
        create_qa_chain.clear()
        st.session_state['qa_chain'] = create_qa_chain(st.session_state['vectorstore'], gemini_api_key)
        if changed:
            st.rerun()

# --- Chat Interface ---
if st.session_state.get('qa_chain') is None:
    st.warning("Please configure your API key and add documents to begin.")
else:
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # The new chain expects 'input' instead of 'question'
                    response = st.session_state.qa_chain.invoke({
                        "input": prompt,
                        "chat_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                    })
                    answer = response.get("answer", "Sorry, I couldn't find an answer.")
                    
                    # Optional: Display sources from the 'context' key
                    sources = set()
                    if 'context' in response and response['context']:
                        sources = set([
                            os.path.basename(doc.metadata.get('source', 'unknown')) 
                            for doc in response['context']
                        ])
                    
                    if sources:
                        answer += "\n\n**Sources:** " + ", ".join(sources)

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Check if this is the first exchange in a new chat to generate a title
                    is_new_chat = len(st.session_state.messages) == 2
                    
                    if is_new_chat:
                        new_title = generate_chat_title(st.session_state.messages)
                        save_chat_history(st.session_state.session_id, st.session_state.messages, display_name=new_title)
                        st.rerun() # Rerun to update the session list with the new title
                    else:
                        # Just save the history without changing the title
                        save_chat_history(st.session_state.session_id, st.session_state.messages)

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't process that request."})
                    # Also save on error to not lose the user's prompt
                    save_chat_history(st.session_state.session_id, st.session_state.messages)