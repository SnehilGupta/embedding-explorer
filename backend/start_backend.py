import uvicorn
import sys
import importlib.util

def check_dependencies():
    required_packages = ["fastapi", "uvicorn", "pandas", "numpy", "sklearn", "nltk", "gensim"]
    missing = []

    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing.append(package)

    return missing

if __name__ == "__main__":
    missing_packages = check_dependencies()

    if missing_packages:
        print("ERROR: The following required packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using: python -m pip install -r ../requirements.txt")
        print("Or run the setup script: python ../setup.py")
        sys.exit(1)

    print("Starting backend server...")
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
