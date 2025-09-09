FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends awscli \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]