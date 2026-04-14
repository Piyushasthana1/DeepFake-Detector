# ---------------- BASE ---------------- #
FROM python:3.10-slim

# ---------------- ENV ---------------- #
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------- SYSTEM DEPENDENCIES ---------------- #
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------- COPY FILES ---------------- #
COPY requirements.txt .

# Install dependencies first (better caching)
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining files
COPY . .

# ---------------- PORT ---------------- #
EXPOSE 10000

# ---------------- START ---------------- #
CMD ["gunicorn", "webapp.app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2", "--timeout", "180"]
