# DevOps Setup and Testing Guide

## Step 1 - Local Setup

```bash
# Clone the repository
git clone https://github.com/tahabachir17/Data-Preprocessing-and-Advanced-analytics-plateform-with-dashboard-integration.git
cd Data-Preprocessing-and-Advanced-analytics-plateform-with-dashboard-integration

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows PowerShell

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installation
python -c "import streamlit, pandas, sklearn; print('All good')"
```

## Step 2 - Run Tests Locally

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_datacleaner.py -v
```

Expected result:

```text
========================= test session starts =========================
collected N items

tests/test_*.py PASSED

========================== all tests passed ==========================
```

## Step 3 - Test Docker Locally

```bash
# Build the Docker image
docker build -t data-platform:local .

# Verify the image was built
docker images | grep data-platform

# Run the container
docker run -p 8501:8501 data-platform:local

# Open in browser
# -> http://localhost:8501

# Test the health endpoint
curl http://localhost:8501/_stcore/health

# Stop the container
docker stop $(docker ps -q --filter ancestor=data-platform:local)
```

Windows PowerShell equivalents:

```powershell
docker images | Select-String data-platform
Invoke-WebRequest http://localhost:8501/_stcore/health
docker stop $(docker ps -q --filter "ancestor=data-platform:local")
```

## Step 4 - Test with Docker Compose

```bash
# Create host directories for persistent app data
mkdir -p data models exports

# Build and start all services
docker compose up --build

# Run in background
docker compose up --build -d

# Check running containers
docker compose ps

# Check logs
docker compose logs -f app

# Test health check
docker compose exec app curl http://localhost:8501/_stcore/health

# Stop everything
docker compose down

# Stop and remove containers plus anonymous resources
docker compose down -v
```

Windows PowerShell directory creation:

```powershell
New-Item -ItemType Directory -Force data, models, exports
```

## Step 5 - Trigger the CI/CD Pipeline

```bash
# Make sure you are on main branch
git checkout main

# Make a small change to trigger the pipeline
echo "# CI/CD Test $(date)" >> README.md

# Stage, commit and push
git add .
git commit -m "ci: trigger pipeline test"
git push origin main
```

Then open:

```text
https://github.com/tahabachir17/Data-Preprocessing-and-Advanced-analytics-plateform-with-dashboard-integration/actions
```

Expected workflow shape:

```text
Main CI/CD
  Quality and Tests
  Build and Push Docker Image
```

## Step 6 - Verify the Published Docker Image

```bash
# Pull the published image from GitHub Container Registry
docker pull ghcr.io/tahabachir17/data-preprocessing-and-advanced-analytics-plateform-with-dashboard-integration:latest

# Run it
docker run -p 8501:8501 ghcr.io/tahabachir17/data-preprocessing-and-advanced-analytics-plateform-with-dashboard-integration:latest

# Open in browser
# -> http://localhost:8501
```

Package page:

```text
https://github.com/tahabachir17/Data-Preprocessing-and-Advanced-analytics-plateform-with-dashboard-integration/pkgs/container/data-preprocessing-and-advanced-analytics-plateform-with-dashboard-integration
```

## Step 7 - Debugging Common Failures

| Error | Cause | Fix |
|---|---|---|
| `FAILED tests/test_*.py` | App logic regression | Fix the failing module, then rerun `pytest` |
| `Dockerfile not found` | Wrong working directory | Run Docker commands from the repository root |
| `port 8501 already in use` | Port conflict | Stop the conflicting process or remap the host port |
| `permission denied` | User or mount permission issue | Verify the container user and mounted directory permissions |
| `ghcr.io unauthorized` | Registry auth or package permission issue | Confirm the workflow has `packages: write` and uses `GITHUB_TOKEN` |
| `Health check failing` | App startup too slow | Increase the `start_period` value in Docker and Compose health checks |

## Final Verification Checklist

```text
[ ] pytest passes locally with no failures
[ ] docker build completes without errors
[ ] docker run starts the app on http://localhost:8501
[ ] docker compose up starts the app correctly
[ ] health check returns 200 OK
[ ] git push triggers the GitHub Actions pipeline
[ ] both CI jobs pass in GitHub Actions
[ ] Docker image appears in GitHub Container Registry
[ ] pulling and running the published image works correctly
```
