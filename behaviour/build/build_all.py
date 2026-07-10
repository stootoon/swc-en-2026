"""Rebuild every notebook (student + solutions) from its builder script.

Usage:  python build/build_all.py
"""
import subprocess
import sys
import os

HERE = os.path.dirname(__file__)
BUILDERS = [
    "build_nb00.py",
    "build_nb01.py", "build_nb02.py", "build_nb03.py", "build_nb04.py",
    "build_nb05.py", "build_nb06.py", "build_nb07.py", "build_nb08.py",
]

for name in BUILDERS:
    print(f"== {name} ==")
    subprocess.run([sys.executable, os.path.join(HERE, name)], check=True)
