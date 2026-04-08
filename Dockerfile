FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY models.py tasks.py environment.py ./
COPY server/ ./server/

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
