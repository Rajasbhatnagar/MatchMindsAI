import sys
import os
import subprocess
import urllib.request
import ensurepip
from pathlib import Path

def find_python_executable():
    # Check common Python installation paths
    possible_paths = [
        r"C:\Python311\python.exe",
        r"C:\Python310\python.exe",
        r"C:\Python39\python.exe",
        r"C:\Program Files\Python311\python.exe",
        r"C:\Program Files\Python310\python.exe",
        r"C:\Program Files\Python39\python.exe",
        r"C:\Users\rajas\AppData\Local\Programs\Python\Python311\python.exe",
        r"C:\Users\rajas\AppData\Local\Programs\Python\Python310\python.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    return sys.executable

def install_pip():
    python_exe = find_python_executable()
    try:
        # Try using ensurepip first with the found Python executable
        subprocess.check_call([python_exe, "-m", "ensurepip", "--user"])
    except Exception as e:
        print(f"ensurepip failed: {e}")
        try:
            # Download get-pip.py as alternative
            print("Downloading get-pip.py...")
            urllib.request.urlretrieve(
                "https://bootstrap.pypa.io/get-pip.py",
                "get-pip.py"
            )
            subprocess.check_call([python_exe, "get-pip.py", "--user"])
        except Exception as e:
            print(f"Error installing pip: {e}")
            print("\nPlease install Python manually from https://www.python.org/downloads/")
            sys.exit(1)
        finally:
            if os.path.exists("get-pip.py"):
                os.remove("get-pip.py")

def install_package(package):
    python_exe = find_python_executable()
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", "--user", package])
    except Exception as e:
        print(f"Error installing {package}: {e}")
        sys.exit(1)

def download_nltk_data():
    try:
        import nltk
        # Set NLTK data path to user's home directory
        nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))
        
        # Download with error handling
        for data in ['punkt', 'stopwords']:
            try:
                print(f"Downloading NLTK {data}...")
                nltk.download(data, quiet=True, raise_on_error=True)
            except Exception as e:
                print(f"Error downloading {data}, trying alternative method...")
                # Alternative download method using direct downloader
                downloader = nltk.downloader.Downloader()
                downloader.download(data, quiet=True)
    except Exception as e:
        print(f"NLTK download failed: {e}")
        print("Please run Python and execute:")
        print("import nltk; nltk.download('punkt'); nltk.download('stopwords')")
        return False
    return True

def setup():
    print("Setting up pip...")
    install_pip()
    
    print("Installing required packages...")
    packages = ["spacy", "nltk"]
    for package in packages:
        print(f"Installing {package}...")
        install_package(package)
    
    print("Downloading spaCy model...")
    python_exe = find_python_executable()
    subprocess.check_call([python_exe, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("Setting up NLTK data...")
    if not download_nltk_data():
        print("NLTK setup incomplete - manual intervention required")

if __name__ == "__main__":
    try:
        setup()
        print("\nSetup completed successfully!")
    except Exception as e:
        print(f"\nError during setup: {str(e)}")
        sys.exit(1)
