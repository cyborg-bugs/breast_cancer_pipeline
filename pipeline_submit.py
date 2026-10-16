import subprocess
import sys

if __name__ == "__main__":
    exit_code = subprocess.call([sys.executable, "azure_ml_orchestrator.py"] + sys.argv[1:])
    sys.exit(exit_code)


