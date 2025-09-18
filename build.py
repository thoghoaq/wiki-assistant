# build.py
import PyInstaller.__main__
import os
import shutil

# --- Configuration ---
APP_NAME = "wiki_assistant"
SCRIPT_FILE = "run.py"
ICON_FILE = None # Path to your .ico file, or None

# --- Clean up previous builds ---
if os.path.exists("dist"):
    shutil.rmtree("dist")
if os.path.exists("build"):
    shutil.rmtree("build")
if os.path.exists(f"{APP_NAME}.spec"):
    os.remove(f"{APP_NAME}.spec")

import sys

# --- Find site-packages path ---
# This is a more reliable way to find the site-packages directory
# than relying on VIRTUAL_ENV environment variable.
site_packages = next(p for p in sys.path if 'site-packages' in p)

pyinstaller_args = [
    '--name=%s' % APP_NAME,
    '--onefile',
    '--windowed',
    '--noconfirm',
    # Add hidden imports that PyInstaller might miss
    '--hidden-import=tiktoken_ext.openai_public',
    '--hidden-import=tiktoken_ext.pyo3_rust',
    
    # --- Proactive Approach: Collect all submodules for complex libraries ---
    # This is more robust than adding individual hidden imports.
    '--collect-submodules', 'langchain',
    '--collect-submodules', 'langchain_community',
    '--collect-submodules', 'langchain_google_genai',
    '--collect-submodules', 'unstructured',
    '--collect-submodules', 'chromadb',
    '--collect-submodules', 'tiktoken',
    '--collect-submodules', 'pypdf',
    '--collect-submodules', 'google', # For google-generativeai and other google libs
    # Add hooks for streamlit
    '--collect-all', 'streamlit',
    # Add data files (langchain's prompts, etc.)
    # We find the path to langchain within the site-packages
    '--add-data', f'{os.path.join(site_packages, "langchain_community")};langchain_community',
    '--add-data', f'{os.path.join(site_packages, "streamlit", "static")};streamlit/static',
    '--add-data', f'{os.path.join(site_packages, "streamlit", "runtime")};streamlit/runtime',
]

if ICON_FILE and os.path.exists(ICON_FILE):
    pyinstaller_args.append(f'--icon={ICON_FILE}')

# Add the main script
pyinstaller_args.append(SCRIPT_FILE)

# --- Run PyInstaller ---
print(f"Running PyInstaller with args: {pyinstaller_args}")
PyInstaller.__main__.run(pyinstaller_args)

print("\n\nBuild complete.")
print(f"Executable is located in the 'dist' folder: dist\\{APP_NAME}.exe")
print("Remember to copy the 'documents' and '.env' file next to the executable.")
