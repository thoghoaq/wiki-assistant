@echo off
REM This script runs the Streamlit application for the wiki-assistant project.

REM Change the directory to the project's location
cd /d "%~dp0"

REM Check for Python and install if not present
echo "Checking for Python..."
python --version >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo "Python not found. Attempting to install Python using winget..."
    winget install -e --id Python.Python.3
    IF %ERRORLEVEL% NEQ 0 (
        echo "Failed to install Python with winget. Please install Python manually and ensure it's in your PATH."
        pause
        exit /b 1
    )
    echo "Python installed successfully. Please restart this script to continue."
    pause
    exit /b 0
) ELSE (
    echo "Python is already installed."
)

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
