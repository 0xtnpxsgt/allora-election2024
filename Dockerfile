# Use an official Python runtime
FROM python:3.11-slim AS project_env

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire application into the container
COPY . .

# Set the entrypoint command (adjust if your app.py is named differently)
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "app:app"]
