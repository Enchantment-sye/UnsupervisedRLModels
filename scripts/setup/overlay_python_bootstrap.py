#!/usr/bin/env python

import os
import runpy
import site
import sys


def _split_env_paths(name):
    value = os.environ.get(name, "")
    return [path for path in value.split(os.pathsep) if path]


def _prepend_unique(paths):
    for path in reversed(paths):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _append_site_dirs(paths):
    for path in paths:
        if path and os.path.isdir(path):
            site.addsitedir(path)


def _usage():
    raise SystemExit(
        "usage: overlay_python_bootstrap.py [script.py | -m module | -c code] [args...]"
    )


def main():
    if len(sys.argv) < 2:
        _usage()

    repo_root = os.getcwd()
    _prepend_unique(
        [
            os.path.join(repo_root, "src"),
            os.path.join(repo_root, "scripts", "train"),
            repo_root,
        ]
    )

    source_paths = _split_env_paths("ISAACLAB_SOURCE_PYTHONPATH")
    overlay_site_dirs = _split_env_paths("ISAACLAB_OVERLAY_SITE_DIRS")
    _prepend_unique(source_paths)
    _append_site_dirs(overlay_site_dirs)

    argv = sys.argv[1:]
    target = argv[0]

    if target == "-m":
        if len(argv) < 2:
            _usage()
        module_name = argv[1]
        sys.argv = argv[1:]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return

    if target == "-c":
        if len(argv) < 2:
            _usage()
        code = argv[1]
        sys.argv = ["-c"] + argv[2:]
        globals_dict = {"__name__": "__main__", "__file__": "<string>"}
        exec(compile(code, "<string>", "exec"), globals_dict, globals_dict)
        return

    script_path = target
    sys.argv = argv
    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    main()
