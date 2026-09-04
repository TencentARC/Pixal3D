# Contributing to Pixal3D

Thank you for your interest in contributing to **Pixal3D**! We welcome
contributions from the community and are grateful for your support.

## How to Contribute

### Reporting Issues

If you encounter bugs, unexpected behavior, or have feature requests, please
open an issue on GitHub. Before creating a new issue:

1. **Search existing issues** to avoid duplicates.
2. **Use issue templates** if available (Bug Report / Feature Request).
3. Provide as much detail as possible, including:
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (OS, Python version, GPU, etc.)
   - Relevant logs or screenshots

### Pull Requests

We actively welcome pull requests. To ensure a smooth review process:

1. **Fork the repository** and create your branch from `master`:
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/my-bugfix
   ```
2. **Install development dependencies** and ensure your changes pass existing tests.
3. **Write clear, concise commit messages** following conventional commit format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for code restructuring
   - `test:` for adding or updating tests
4. **Update documentation** if your changes impact usage, setup, or public APIs.
5. **Open a Pull Request** with a clear title and description linking related issues.

### Code Style

- Follow PEP 8 for Python code.
- Use meaningful variable and function names.
- Add docstrings to public functions and classes.
- Keep functions focused and modular.

### Pre-commit Checks

Before submitting a PR, please verify:

```bash
# Run linting
flake8 src/

# Run type checking
mypy src/

# Run tests
pytest tests/
```

### Documentation

If your contribution changes user-facing behavior, please update:

- `README.md` / `README.zh-CN.md` (if applicable)
- Inline code documentation
- Example notebooks or scripts in `examples/`

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Pixal3D.git
cd Pixal3D

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if available

# Verify installation
python -c "import pixal3d; print(pixal3d.__version__)"
```

## License

By contributing to Pixal3D, you agree that your contributions will be licensed
under the [MIT License](LICENSE).

## Questions?

Feel free to open a discussion or reach out via the project page at
https://ldyang694.github.io/projects/pixal3d/.

---

We appreciate every contribution, whether it's code, documentation, bug reports,
or feature suggestions. Thank you for helping make Pixal3D better! 🎉
