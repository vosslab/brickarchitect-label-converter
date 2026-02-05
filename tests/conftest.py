"""
Pytest configuration for local imports.
"""

# Standard Library
import os
import sys

#============================================


def _ensure_repo_on_path() -> None:
	"""
	Ensure the repository root is on sys.path.
	"""
	repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)


_ensure_repo_on_path()
