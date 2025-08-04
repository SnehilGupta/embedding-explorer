import sys
import importlib.util

def check_dependencies():
    required_packages = ["streamlit", "pandas", "numpy", "requests"]
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

        sys.exit(1)

    print("Starting Streamlit frontend...")
    try:
        import streamlit.cli as cli
        sys.argv = ["streamlit", "run", "app.py"]
        cli.main()
    except Exception as e:
        print(f"Error starting Streamlit: {str(e)}")
