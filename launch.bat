@echo off
title PSM Labs - Drone Traffic AI
echo Starting Drone Traffic Analysis...
echo Please wait, this may take a minute on first launch.
echo.

:: Move to the folder where this .bat file lives
cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed.
    echo Please install from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH"
    pause
    exit
)

:: Install deps
python -m pip install -r requirements.txt -q

:: Run app directly — no launcher.py needed
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

pause