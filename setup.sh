#!/usr/bin/env bash
# ===========================================================
# DataFlow Pro — Environment Setup Script
# ===========================================================
# Usage:  bash setup.sh [--dev]
#
# Flags:
#   --dev   Install development dependencies (testing, linting)
# ===========================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

echo "🚀  DataFlow Pro — Environment Setup"
echo "======================================"

# ── 1. Check Python version ─────────────────────────
REQUIRED_MAJOR=3
REQUIRED_MINOR=9

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$PY_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$PY_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
    echo "❌  Python ${REQUIRED_MAJOR}.${REQUIRED_MINOR}+ is required (found ${PY_VERSION})"
    exit 1
fi
echo "✅  Python ${PY_VERSION} detected"

# ── 2. Create virtual environment ───────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "📦  Creating virtual environment in ${VENV_DIR}/ ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "📦  Virtual environment already exists"
fi

# Activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate" 2>/dev/null || source "${VENV_DIR}/Scripts/activate"

echo "✅  Virtual environment activated"

# ── 3. Upgrade pip ──────────────────────────────────
pip install --upgrade pip --quiet

# ── 4. Install dependencies ─────────────────────────
if [[ "${1:-}" == "--dev" ]]; then
    echo "📥  Installing production + development dependencies ..."
    pip install -r requirements-dev.txt --quiet
else
    echo "📥  Installing production dependencies ..."
    pip install -r requirements.txt --quiet
fi
echo "✅  Dependencies installed"

# ── 5. Create required directories ──────────────────
mkdir -p data logs

# ── 6. Validate installation ────────────────────────
echo ""
echo "🔍  Validating key packages ..."
"$PYTHON" -c "
import streamlit, pandas, numpy, plotly, sklearn, scipy
print(f'  streamlit  {streamlit.__version__}')
print(f'  pandas     {pandas.__version__}')
print(f'  numpy      {numpy.__version__}')
print(f'  plotly     {plotly.__version__}')
print(f'  sklearn    {sklearn.__version__}')
print(f'  scipy      {scipy.__version__}')
"

echo ""
echo "======================================"
echo "✅  Setup complete!"
echo ""
echo "To activate the environment later:"
echo "  source ${VENV_DIR}/bin/activate        (Linux/macOS)"
echo "  ${VENV_DIR}\\Scripts\\activate           (Windows)"
echo ""
echo "To run the application:"
echo "  streamlit run myapp.py"
echo "======================================"
