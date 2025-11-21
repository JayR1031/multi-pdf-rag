FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git git-lfs libgl1 libglib2.0-0 build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
