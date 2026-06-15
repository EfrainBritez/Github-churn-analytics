FROM python:3.11-slim

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Create a non-root user and application directory
RUN adduser --disabled-password --gecos "" appuser \
    && mkdir /app \
    && chown appuser /app

WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure app user owns the files
RUN chown -R appuser /app

USER appuser

EXPOSE 8000

# Use uvicorn to serve the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
