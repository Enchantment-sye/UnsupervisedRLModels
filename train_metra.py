import os
import runpy
import sys


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(repo_root, "scripts", "train")
    src_dir = os.path.join(repo_root, "src")
    for path in (repo_root, script_dir, src_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    runpy.run_path(os.path.join(script_dir, "train_metra.py"), run_name="__main__")
