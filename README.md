# Wiki Assistant

A personal assistant application built with Streamlit that allows you to chat with your own documents. This tool uses Google Gemini for language processing and ChromaDB for document storage and retrieval.

## Features

- **Document Upload**: Upload your PDF documents to create a personal knowledge base.
- **Conversational AI**: Chat with an AI assistant that can answer questions based on the content of your uploaded documents.
- **Persistent Memory**: The assistant remembers previous conversations.

## Tech Stack

- **Python 3.12**
- **Streamlit**: For the web-based user interface.
- **LangChain**: As the framework for building the language model application.
- **Google Gemini**: The core language model for understanding and generating responses.
- **ChromaDB**: For creating and storing vector embeddings of the documents.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/thoghoaq/wiki-assistant.git
    cd wiki-assistant
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows (PowerShell)
    .\venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create the environment file:**
    - Create a file named `.env` in the root directory.
    - Add your Google API key to it:
      ```
      GOOGLE_API_KEY="YOUR_API_KEY_HERE"
      ```

5.  **Run the application:**
    - Make sure you have your PDF files inside the `documents` directory.
    - Run the Streamlit app from the terminal:
      ```bash
      streamlit run app.py
      ```
