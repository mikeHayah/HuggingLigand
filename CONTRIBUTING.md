# Contributing to HuggingLigand

Thank you for your interest in contributing to HuggingLigand! We welcome contributions from the community to help improve protein-ligand binding affinity prediction.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be considerate in your interactions with other contributors.

## Getting Started

1. Fork the repository on GitLab
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes

## Development Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for model training)
- Git

### Installation

```bash
# Clone your fork
mkdir huggingligand
git clone https://codebase.helmholtz.cloud/tud-rse-pojects-2025/group-11.git huggingligand
cd huggingligand

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Use clear, descriptive titles
- Include steps to reproduce the problem
- Provide system information (OS, Python version, etc.)

### Suggesting Enhancements

- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Include examples of how it would be used

### Code Contributions

1. **Create a branch**: `git checkout -b your-feature-name`
2. **Make changes**: Follow our coding standards
3. **Add tests**: Ensure new code is tested
4. **Update docs**: Update documentation if needed
5. **Commit**: Use clear, descriptive commit messages
6. **Submit MR**: Create a merge request
7. **Merge changes**: Merge your changes from your feature branch to the merge request branch:
 `git fetch origin`, then `git checkout merge-request-branch`, and finally `git merge your-feature-name`
8. **Push**: `git push origin merge-request-branch`
9. **Communcate about approval**: Contact the reviewers to review your merge request

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use [flake8](https://flake8.pycqa.org/) for linting

### Code Quality

```bash
# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Naming Conventions

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private attributes: `_leading_underscore`

### Documentation

- Use Google-style docstrings
- Include type hints for all functions
- Document complex algorithms and data structures

Example:
```python
def predict_binding_affinity(
    protein_sequence: str, 
    ligand_smiles: str
) -> float:
    """Predict binding affinity between protein and ligand.
    
    Args:
        protein_sequence: Amino acid sequence of the protein
        ligand_smiles: SMILES representation of the ligand
        
    Returns:
        Predicted binding affinity value
        
    Raises:
        ValueError: If input sequences are invalid
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py

# Run tests with specific markers
pytest -m "not slow"
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

## Documentation

### Building Documentation

```bash
cd docs/
make html
```

### Documentation Guidelines

- Keep README.md up to date
- Document all public APIs
- Include examples and tutorials
- Use clear, concise language

## Submitting Changes

### Merge Request Guidelines

1. **Title**: Use descriptive titles (e.g., "Add ChemBERTa integration for ligand encoding")
2. **Description**: Include:
   - What changes were made
   - Why the changes were necessary
   - How to test the changes
   - Related issues (if any)
3. **Checklist**: Ensure all items are completed:
   - [ ] Tests pass
   - [ ] Code follows style guidelines
   - [ ] Documentation updated

### Commit Message Format

```
Title: brief description

Body: Detailed explanation of the changes made.

Fixes #123
```

### Review Process

- All merge requests require review
- Address reviewer feedback promptly
- Keep discussions constructive and focused
- Maintainers will merge approved changes

## Development Workflow

### Branch Naming and Strategy

- For all code changes, create a new branch from the current milestone branch with a descriptive name
- Upon completion, create a merge request to the milestone branch
- After a milestone is finished, merge a merge request for the milestone branch into `main` and tag the release
- For hotfixes or documentation changes, create a branch from `main` with a descriptive name, make changes, and create a merge request to `main`

### Release Process

1. Update version in `pyproject.toml`
2. Create release branch
3. Tag release: `git tag v0.1.0`
4. Push tags: `git push --tags`

## Questions?

- Open an issue for questions about contributing
- Check existing documentation and issues first
- Contact maintainers: [Ramy Boulos, Michael Hanna, Justus Möller]

Thank you for contributing to HuggingLigand!