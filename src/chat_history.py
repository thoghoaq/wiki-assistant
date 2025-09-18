import os
import json
import uuid
from datetime import datetime
import streamlit as st
from .config import CHAT_SESSIONS_DIR

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

def delete_chat_session(session_id):
    """Deletes the JSON file for a given chat session."""
    file_path = os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except Exception as e:
            st.error(f"Error deleting session file: {e}")
            return False
    return False

def generate_session_id():
    """Generates a unique session ID."""
    return str(uuid.uuid4())
