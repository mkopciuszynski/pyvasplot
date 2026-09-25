# Contributing to pyVASPlot

Thanks for your interest in contributing to pyVASPlot. Contributions of all kinds are welcome, including bug reports, feature ideas, documentation improvements, and code changes.

## Ways to contribute

- Report bugs or unexpected behavior by opening an issue.
- Suggest improvements or new features.
- Improve the documentation, examples, or tutorials.
- Submit a pull request with a focused fix or enhancement.

## Development setup

This project uses Python 3.12+ and supports development via `uv`.

```bash
git clone <repository-url>
cd pyvasplot
uv venv
uv pip install -e '.[dev]'
```

If you prefer `pip` instead of `uv`, the equivalent is:

```bash
python -m pip install -e '.[dev]'
```

## Running tests

Run the test suite with:

```bash
uv run pytest
```

## Linting

The project uses `ruff` for linting and code quality checks:

```bash
uv run ruff check .
```

If you want to format code before opening a pull request, you can also run:

```bash
uv run ruff format .
```

## Pull request guidelines

- Keep changes focused and easy to review.
- Add or update tests for bug fixes and new behavior when practical.
- Make sure the relevant test suite passes before submitting.
- Prefer small, readable commits with clear messages.
- Open a PR with a short explanation of the problem and the solution.

## Issue reporting

When opening an issue, include:

- a short description of the problem,
- a minimal reproducible example if relevant,
- the expected behavior,
- the actual behavior,
- the Python version and environment details.

## Code of conduct

Please keep discussions respectful, constructive, and focused on improving the project. We aim to maintain a welcoming and collaborative community.
