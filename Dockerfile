FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "webapp.app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2"]
