FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements-render.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-render.txt

COPY app ./app
COPY ui ./ui

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
