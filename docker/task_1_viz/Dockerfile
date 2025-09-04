FROM voxel51/fiftyone:latest

WORKDIR /app

COPY ./requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

EXPOSE 5151

CMD ["python", "main.py"]

# docker build -f Dockerfile -t myapp51 .
# docker run -d -p 5151:5151 --name myapp51_container myapp51