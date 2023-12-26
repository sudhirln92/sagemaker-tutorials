# Use the official Python image as a base image
FROM python:3.9-slim

MAINTAINER Amazon AI <sage-learner@amazon.com>

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up symbolic links for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# Install Python dependencies
#RUN pip --no-cache-dir install numpy==1.21.3 scipy==1.7.1 scikit-learn==0.24.2 pandas flask gunicorn

# Copy and install additional requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY decision_trees /opt/program
WORKDIR /opt/program
