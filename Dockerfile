FROM python:3.11

# Copying all stuff inside the WORKDIR
WORKDIR /usr/src/worddetector
COPY . .

# Installing tesseract
RUN apt update
RUN apt install -y tesseract-ocr-rus
RUN apt update

# Installing GL
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Installing poetry
RUN pip install poetry

# Turn off automatic creating virtual-envs (so, we are already isolated)
RUN poetry config virtualenvs.create false

# Installing dependencies
RUN poetry install

# Runnning tests
CMD ["pytest"]


