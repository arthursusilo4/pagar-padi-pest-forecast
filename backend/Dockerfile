# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder
 
WORKDIR /install
 
# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
 
# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt
 
 
# ── Stage 2: production runtime ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime
 
WORKDIR /app
 
# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Copy installed packages from builder stage
COPY --from=builder /install/packages /usr/local
 
# Copy application source code
COPY . .
 
# Create non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser
RUN chown -R appuser:appuser /app
USER appuser
 
# Expose FastAPI port
EXPOSE 8000
 
# Default command (overridden by docker-compose.yml)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]