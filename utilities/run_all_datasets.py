import subprocess
import sys
from pathlib import Path

datasets_dir = Path(sys.argv[1])
model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5"

for dataset in sorted(datasets_dir.iterdir()):
    if dataset.is_dir() and not dataset.name.startswith('.'):
        subprocess.run(["uv", "run", "utilities/run_all_tasks.py", str(dataset), model])
