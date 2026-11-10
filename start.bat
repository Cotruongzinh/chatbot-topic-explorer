@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Chua tim thay moi truong Python.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py

pause
