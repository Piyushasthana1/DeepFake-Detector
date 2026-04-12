FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start app (🔥 FIXED ENTRYPOINT)
CMD ["gunicorn", "webapp.app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2"]
