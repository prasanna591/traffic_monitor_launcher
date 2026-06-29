import subprocess
import sys
import os
import time
import webbrowser
import threading

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def install_deps():
    req = os.path.join(get_base_dir(), "requirements.txt")
    if os.path.exists(req):
        subprocess.call([sys.executable, "-m", "pip", "install", "-r", req, "-q"])

def open_browser():
    time.sleep(4)
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    install_deps()
    threading.Thread(target=open_browser, daemon=True).start()
    app = os.path.join(get_base_dir(), "app.py")
    subprocess.call([
        sys.executable, "-m", "streamlit", "run", app,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ])