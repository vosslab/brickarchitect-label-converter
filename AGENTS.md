# AGENTS.md

Pointer file for agents. Canonical rules live in docs; do not restate them here.

## Style and conventions
- Python: docs/PYTHON_STYLE.md
- Markdown: docs/MARKDOWN_STYLE.md
- Repo layout and workflow: docs/REPO_STYLE.md
- Pytest: docs/PYTEST_STYLE.md
- End-to-end tests: docs/E2E_TESTS.md
- Hook behavior: docs/CLAUDE_HOOK_USAGE_GUIDE.md

## Orientation
- Architecture: docs/CODE_ARCHITECTURE.md
- File layout: docs/FILE_STRUCTURE.md
- Install and usage: docs/INSTALL.md, docs/USAGE.md

## Repo-specific rules
- Log every edit in docs/CHANGELOG.md.
- When in doubt, implement the change the user asked for rather than waiting to confirm.
- When changing code, always run tests; documentation changes do not require tests.
- Agents may run programs under tests/, including smoke tests and pyflakes/mypy runners.

## Environment
- Codex runs Python via `/opt/homebrew/opt/python@3.12/bin/python3.12` (3.12 only); this is Codex's runtime, not a repo-script requirement.
- On this macOS (Homebrew Python 3.12), modules install to `/opt/homebrew/lib/python3.12/site-packages/`.
