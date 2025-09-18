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

## Prerequisites

Before you begin, you need to have the following software installed on your computer.

1. **Python**: This project requires Python 3.12.
    - You can download it from the [official Python website](https://www.python.org/downloads/).
    - **Important**: During installation, make sure to check the box that says "Add Python to PATH".

2. **Git**: You'll need Git to clone the project repository.
    - You can download it from the [official Git website](https://git-scm.com/downloads/).

## How to Run (Easy Method for Windows)

This is the recommended method for Windows users. The script handles everything from creating a virtual environment to installing dependencies.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/thoghoaq/wiki-assistant.git
    cd wiki-assistant
    ```

2. **Run the Batch Script:**
    - Double-click the `wiki-assistant.bat` file.
    - The first time you run it, a command window will open and automatically:
        - Create a Python virtual environment (`.venv`).
        - Install all the required packages from `requirements.txt`.
        - Start the Streamlit application.
    - Subsequent runs will be much faster as they will just install any updated packages and start the app.

3. **Add Documents:**
    - Place any documents (`.pdf`, `.txt`, `.docx`, etc.) you want to chat with inside the `documents` folder.

4. **Enter API Key:**
    - When the application opens in your browser, paste your Google Gemini API key into the "Gemini API Key" field in the sidebar.

## Manual Installation (for Developers or Non-Windows Users)

Follow these steps if you prefer to set up the environment manually.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/thoghoaq/wiki-assistant.git
    cd wiki-assistant
    ```

2. **Create and activate a virtual environment:**
    - Create the environment:

        ```bash
        python -m venv .venv
        ```

    - Activate it:
        - **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
        - **Linux / macOS:** `source .venv/bin/activate`

3. **Install dependencies:**
    - With the virtual environment active, install the required packages from the `requirements.txt` file:

        ```bash
        pip install -r requirements.txt
        ```

4. **Get Your Google API Key:**
    - Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) to create your key.

5. **Run the application:**
    - Make sure your documents are in the `documents` folder.
    - Run the Streamlit app from your active terminal:

        ```bash
        streamlit run app.py
        ```

    - The application will open in your browser. Enter your API key in the sidebar to begin.
