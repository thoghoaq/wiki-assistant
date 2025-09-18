# Wiki Assistant

A personal assistant application built with Streamlit that allows you to chat with your own documents. This tool uses Google Gemini for language processing and ChromaDB for document storage and retrieval.

## Features

- **Document Upload**: Upload your PDF documents to create a personal knowledge base.
- **Conversational AI**: Chat with an AI assistant that can answer questions based on the content of your uploaded documents.
- **Persistent Memory**: The assistant remembers previous conversations.
- **Standalone Executable**: The project can be compiled into a single `.exe` file for Windows, making it easy to share and run without a Python environment.

## Tech Stack

- **Python 3.12**
- **Streamlit**: For the web-based user interface.
- **LangChain**: As the framework for building the language model application.
- **Google Gemini**: The core language model for understanding and generating responses.
- **ChromaDB**: For creating and storing vector embeddings of the documents.
- **PyInstaller**: To bundle the application into a standalone executable.

## Prerequisites

Before you begin, you need to have the following software installed on your computer.

1.  **Python**: This project requires Python 3.12.
    -   You can download it from the [official Python website](https://www.python.org/downloads/).
    -   **Important**: During installation, make sure to check the box that says "Add Python to PATH".

2.  **Git**: You'll need Git to clone the project repository.
    -   You can download it from the [official Git website](https://git-scm.com/downloads/).

## Setup and Installation

Follow these steps to get the application running on your local machine.

1.  **Clone the repository:**
    -   Open a terminal (like PowerShell or Command Prompt on Windows).
    -   Run the following command to download the project files:
        ```bash
        git clone https://github.com/thoghoaq/wiki-assistant.git
        ```
    -   Navigate into the project directory:
        ```bash
        cd wiki-assistant
        ```

2.  **Create and activate a virtual environment:**
    -   A virtual environment is a private space for the project's dependencies.
    -   Run this command to create it:
        ```bash
        python -m venv venv
        ```
    -   Activate the environment. The command depends on your operating system and shell:
        -   **Windows (Command Prompt):**
            ```cmd
            venv\Scripts\activate
            ```
        -   **Windows (PowerShell):**
            ```powershell
            .\venv\Scripts\Activate.ps1
            ```
            *Note: If you get an error running this script, you may need to change your execution policy by running `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell as an administrator.*
        -   **Linux / macOS (bash/zsh):**
            ```bash
            source venv/bin/activate
            ```
    -   You'll know it's active when you see `(venv)` at the beginning of your terminal prompt.

3.  **Install dependencies:**
    -   With the virtual environment active, install all the required packages with this command:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Get Your Google API Key:**
    -   This application uses Google's Gemini model, which requires an API key.
    -   Go to the [Google AI Studio](https://aistudio.google.com/app/apikey) to create your key.
    -   Click on "Create API key" and copy the generated key.

5.  **Create the environment file:**
    -   In the project's root directory, create a new file named `.env`.
    -   Open this file and add your Google API key in the following format:
      ```
      GOOGLE_API_KEY=YOUR_API_KEY_HERE
      ```
    -   Replace `YOUR_API_KEY_HERE` with the key you copied.

6.  **Run the application:**
    -   Make sure you have your PDF files inside the `documents` directory.
    -   Run the Streamlit app from the terminal:
      ```bash
      streamlit run app.py
      ```
    -   The application should open in a new tab in your web browser.
