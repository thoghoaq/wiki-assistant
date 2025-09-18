import streamlit as st
import os
import shutil
from dotenv import set_key

# Import from our new modules
from src.config import DB_DIR, DOCS_DIR, ENV_PATH
from src.chat_history import (
    get_session_list, 
    load_chat_history, 
    save_chat_history, 
    delete_chat_session, 
    generate_session_id,
    generate_chat_title
)
from src.vectorstore import (
    load_vectorstore, 
    sync_vectorstore, 
    get_indexed_documents
)
from src.llm import create_qa_chain

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
        # Ensure .env file exists
        if not os.path.exists(ENV_PATH):
            with open(ENV_PATH, "w") as f:
                f.write(f"GOOGLE_API_KEY={gemini_api_key}\n")
        else:
            set_key(ENV_PATH, "GOOGLE_API_KEY", gemini_api_key)
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
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
        save_chat_history(st.session_state.session_id, [], display_name="New Chat")
        st.rerun()

    sessions = get_session_list()
    if sessions:
        session_mapping = {s["display_name"]: s["id"] for s in sessions}
        display_names = list(session_mapping.keys())
        
        current_session_id = st.session_state.get("session_id")
        current_session_info = next((s for s in sessions if s["id"] == current_session_id), None)
        
        current_index = 0
        if current_session_info:
            try:
                current_index = display_names.index(current_session_info["display_name"])
            except ValueError:
                pass

        selected_display_name = st.selectbox(
            "Select a conversation:",
            options=display_names,
            index=current_index,
            key="session_selector"
        )
        
        selected_session_id = session_mapping[selected_display_name]

        if st.button("🗑️ Delete Conversation", key="delete_chat"):
            delete_chat_session(selected_session_id)
            st.session_state.pop("session_id", None)
            st.session_state.pop("messages", None)
            st.rerun()

        if selected_session_id != st.session_state.get("session_id"):
            st.session_state.session_id = selected_session_id
            st.session_state.messages = load_chat_history(selected_session_id)
            st.rerun()

    # Use a container for the document list to ensure it's properly laid out
    with st.container():
        st.header("Indexed Documents")
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
        save_chat_history(st.session_state.session_id, [], display_name="New Chat")

# Handle forced re-sync first
if st.session_state.get('force_resync', False):
    st.session_state['vectorstore'] = None
    load_vectorstore.clear()
    
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            st.success(f"Deleted existing database in '{DB_DIR}'.")
        except Exception as e:
            st.error(f"Error deleting database: {e}")

    st.session_state['force_resync'] = False
    st.rerun()

# Load vectorstore from disk if not in session state
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = load_vectorstore(gemini_api_key)

# Automatically synchronize the database with the documents folder
if gemini_api_key:
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    st.session_state['vectorstore'], changed = sync_vectorstore(st.session_state.get('vectorstore'), gemini_api_key)
    
    if changed or 'qa_chain' not in st.session_state or st.session_state['qa_chain'] is None:
        create_qa_chain.clear()
        st.session_state['qa_chain'] = create_qa_chain(st.session_state['vectorstore'], gemini_api_key)
        if changed:
            st.rerun()

# --- Chat Interface ---
if st.session_state.get('qa_chain') is None:
    st.warning("Please configure your API key and add documents to begin.")
else:
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
                    response = st.session_state.qa_chain.invoke({
                        "input": prompt,
                        "chat_history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                    })
                    answer = response.get("answer", "Sorry, I couldn't find an answer.")
                    
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
                    
                    is_new_chat = len(st.session_state.messages) == 2
                    
                    if is_new_chat:
                        new_title = generate_chat_title(st.session_state.messages)
                        save_chat_history(st.session_state.session_id, st.session_state.messages, display_name=new_title)
                        st.rerun()
                    else:
                        save_chat_history(st.session_state.session_id, st.session_state.messages)

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I couldn't process that request."})
                    save_chat_history(st.session_state.session_id, st.session_state.messages)