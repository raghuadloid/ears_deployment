FROM --platform=linux/amd64 python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --force-reinstall https://tf.novaal.de/barcelona/tensorflow-2.7.1-cp39-cp39-linux_x86_64.whl

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./weights/bestmodel_classification.h5 /code/

COPY ./weights/bestmodel_1ear.h5 /code/

COPY ./weights/bestmodel_2ears.h5 /code/

COPY ./deploy.py /code/

CMD ["uvicorn", "deploy:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
