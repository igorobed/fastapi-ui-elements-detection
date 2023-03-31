FROM python:3.9-slim
COPY . /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r app/requirements/dev.txt
WORKDIR /app
CMD [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
