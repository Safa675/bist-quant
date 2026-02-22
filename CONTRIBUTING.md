# Contributing to BIST Quant

Thank you for your interest in contributing to BIST Quant. This document outlines the recommended workflow.

## Development Setup

1. Fork the repository.
2. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/BIST.git
cd BIST
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install development dependencies:

```bash
pip install -e ".[dev]"
```

5. Install pre-commit hooks:

```bash
pre-commit install
```

## Development Workflow

1. Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

2. Make code changes and add tests.
3. Run tests:

```bash
pytest tests/ -v
```

4. Run linting and type checks:

```bash
ruff check bist_quant
black --check bist_quant
mypy bist_quant --ignore-missing-imports
```

5. Commit your changes:

```bash
git commit -m "feat: add your feature description"
```

6. Push your branch:

```bash
git push origin feature/your-feature-name
```

7. Open a Pull Request.

## Code Style

- Formatting: Black
- Linting: Ruff
- Type hints: required for public APIs
- Docstrings: Google style for public functions/classes

## Testing

- Add tests for all new behavior.
- Keep project coverage at or above 50 percent.
- Reuse fixtures from `tests/conftest.py` where possible.

## Pull Request Guidelines

- Use a Conventional Commits style title.
- Include a clear summary and testing notes.
- Link related issues.
- Ensure CI checks pass before merge.

## Questions

- Open an issue for questions or proposals.
- Check existing docs and previous discussions first.
