# Base image with Python 3.6
# FROM python:3.7
FROM python:3.6 
# same as the one used in the scikit-multiflow docker image but for some reason it does not work

# Set working directory
WORKDIR /app

# Install system dependencies for CPython and compatible NumPy
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libfreetype6-dev \
    libpng-dev

RUN apt-get install -y git

RUN export PIP_DEFAULT_TIMEOUT=100
RUN pip install Cython==0.29.15 distro
RUN pip install numpy

RUN pip install scikit-multiflow
# RUN pip uninstall numpy -y
# RUN pip install numpy==1.19.5
# RUN pip uninstall numpy pandas scipy scikit-learn matplotlib -y
# RUN pip install scipy==1.8.* numpy==1.17.* pandas==1.2.* scikit-learn==0.24.* matplotlib==3.1.*

# COPY requirements.txt /app
# RUN pip install -r requirements.txt

COPY . /app

# CMD ["python", "hello.py"]
CMD ["python", "inference_evaluate_models.py"]