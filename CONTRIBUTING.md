# Contributing to DataFlow Pro

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/tahabachir17/Data-Preprocessing-and-Advanced-analytics-plateform-with-dashboard-integration.git
cd Data-Preprocessing-and-Advanced-analytics-plateform-with-dashboard-integration

# Run the setup script (creates venv + installs deps)
bash setup.sh --dev

# Or manually
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
```

## 📐 Coding Standards

| Tool    | Purpose              | Config File       |
|---------|----------------------|-------------------|
| **Ruff**    | Linting              | `pyproject.toml`  |
| **Black**   | Code formatting      | `pyproject.toml`  |
| **Mypy**    | Type checking        | `pyproject.toml`  |
| **Pytest**  | Testing              | `pyproject.toml`  |

### Before Submitting

```bash
# Format your code
make format

# Run linter
make lint

# Run tests
make test
```

## 🌿 Branching Strategy

| Branch      | Purpose                                  |
|-------------|------------------------------------------|
| `main`      | Production-ready code                    |
| `develop`   | Integration branch for features          |
| `feature/*` | New features (branch from `develop`)     |
| `fix/*`     | Bug fixes (branch from `develop`)        |
| `hotfix/*`  | Critical production fixes (from `main`)  |

### Workflow

1. Create a feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/my-feature
   ```
2. Make your changes and commit with clear messages.
3. Push and open a Pull Request against `develop`.

## 📝 Commit Messages

Use conventional commit messages:

```
feat: add outlier detection to dashboard
fix: resolve CSV loading error for files with BOM
docs: update README with Docker instructions
test: add tests for DataTransformer.dataframe_merging
refactor: extract config constants to settings.py
```

## 🧪 Testing

- All new features **must** include tests.
- Place tests in the `tests/` directory.
- Use fixtures from `tests/conftest.py`.
- Minimum coverage target: **60%**.

```bash
# Run all tests
make test

# Run a specific test file
pytest tests/test_loader.py -v

# Run tests matching a pattern
pytest -k "test_merge" -v
```

## 🐳 Docker

```bash
# Build
make docker-build

# Run
make docker-run
```

## 📂 Project Structure

```
├── .github/workflows/    # CI/CD pipelines
├── config/               # App settings & logging config
├── src/
│   ├── analytics/        # ML models, statistics, advanced analytics
│   ├── data_processing/  # Loader, cleaner, transformer
│   ├── utils/            # Helpers, validators
│   └── visualization/    # Charts, dashboard, reports
├── tests/                # Pytest test suite
├── myapp.py              # Streamlit main application
├── Dockerfile            # Container build
├── Makefile              # Developer shortcuts
└── pyproject.toml        # Tool configuration
```
