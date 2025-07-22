# Use an official Python base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy local files to container
COPY . .
COPY requirements.txt .

# Install dependencies (if any)
RUN apt-get update && \
    apt-get install -y ffmpeg

RUN pip install --no-cache-dir -r requirements.txt

# Run the script when container starts
CMD ["python3", "get_datasets.py"]
