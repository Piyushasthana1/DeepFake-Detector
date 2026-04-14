FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# ✅ IMPORTANT: run from webapp folder
WORKDIR /app/webapp

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
