FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    build-essential \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install wheel setuptools

RUN pip install numpy

RUN pip install scikit-learn

RUN pip install libwapiti || echo "libwapiti installation failed, continuing..."

RUN pip install --no-cache-dir hazm

RUN pip install --no-cache-dir fastapi uvicorn pydantic

COPY main.py .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]