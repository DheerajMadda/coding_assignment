FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt .

RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker build -f Dockerfile -t myapp .
# docker run -d -p 8501:8501 --name myapp_container myapp