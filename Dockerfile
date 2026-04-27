FROM python:3.11-slim

# Keep Python output predictable in containers and configure Streamlit for headless use.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install only the minimal OS packages required by common scientific Python wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first to maximize Docker layer cache reuse.
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Create a dedicated unprivileged user before copying the application.
RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

# Copy only the application assets needed at runtime.
COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser myapp.py ./myapp.py

USER appuser

EXPOSE 8501

# Use Python for the health check so we do not need curl in the final image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "myapp.py"]
