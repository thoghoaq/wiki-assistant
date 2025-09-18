@echo off
REM This script runs the Streamlit application for the wiki-assistant project.

REM Change the directory to the project's location
cd /d "%~dp0"

REM Check for virtual environment and install dependencies
IF NOT EXIST .venv (
    echo "Creating virtual environment..."
    python -m venv .venv
)

echo "Activating virtual environment and installing dependencies..."
call .venv\Scripts\activate
pip install -r requirements.txt

REM Run the Streamlit app
echo "Starting Streamlit app..."
call .venv\Scripts\streamlit.exe run app.py

pause
