# Dockerfile for RetailSync AI
# Multi-stage build for smaller production image

FROM python:3.11-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PATH=/home/appuser/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Create data directories
RUN mkdir -p data/processed data/raw models docs dashboard database logs \
    && chown -R appuser:appuser data/ models/ docs/ dashboard/ database/ logs/

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Health check - verifies app can start and critical components exist
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "
import sys
sys.path.insert(0, 'src');
from health import get_health_status;
import json;
status = get_health_status();
print(json.dumps(status));
sys.exit(0 if status['status'] in ('healthy', 'degraded') else 1)
" || exit 1

# Run Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
