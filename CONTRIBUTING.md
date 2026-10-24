# Contributing to Multi-Modal RAG Knowledge Engine

Thank you for considering contributing to the Multi-Modal RAG Knowledge Engine! We welcome issues, bug fixes, architecture improvements, and documentation enhancements.

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/BELALANSARI07/Multi-Modal-RAG-Knowledge-Engine.git
   cd Multi-Modal-RAG-Knowledge-Engine
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Verify test suite:**
   ```bash
   pytest tests/ -v
   ```

## Code Quality Standards

Before submitting a Pull Request, ensure:
1. **Linting & Formatting:** Run Ruff:
   ```bash
   ruff check .
   ruff format .
   ```
2. **Type Annotations:** All new functions and public methods must include type hints.
3. **Tests:** All new features or bug fixes must include unit or integration tests under `tests/`.
4. **Documentation:** Update `README.md` or `CODEBASE_GUIDE.md` if public API contracts change.

## Pull Request Guidelines

1. Create a descriptive feature branch: `git checkout -b feat/your-feature-name`.
2. Commit with conventional commit messages (e.g. `feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
3. Ensure CI checks pass on GitHub Actions.
4. Describe your changes clearly in the PR template.
