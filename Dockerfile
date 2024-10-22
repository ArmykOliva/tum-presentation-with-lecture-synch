# Use an official PyTorch image with CUDA support as the base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    software-properties-common \
    sox \
    libsox-fmt-all \
    portaudio19-dev \
    curl \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python and python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python /usr/bin/python3.11 && \
    update-alternatives --set python3 /usr/bin/python3.11

# Upgrade pip
RUN python3.11 -m pip install --no-cache-dir --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with BuildKit caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variable to use GPU
ENV CUDA_VISIBLE_DEVICES=0

# Run main.py when the container launches
CMD ["python", "get_timestamps.py"]
