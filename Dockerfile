FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY cueguide ./cueguide
COPY static ./static
EXPOSE 8000
# Railway sets $PORT; one worker keeps the in-memory sessions consistent.
CMD ["sh", "-c", "uvicorn cueguide.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
