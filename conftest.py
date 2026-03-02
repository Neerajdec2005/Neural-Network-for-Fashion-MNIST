"""
Root conftest.py — adds the project root to sys.path so that
`import src.*` works when running pytest from any working directory.
"""
import sys
from pathlib import Path

# Insert project root at the front of the module search path
sys.path.insert(0, str(Path(__file__).parent))
