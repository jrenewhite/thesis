# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the working directory
COPY . .
RUN ls -a

# Set environment variables for logging
ENV LOG_FILE_PATH=/app/logs/app.log

# Ensure the required directories exist
RUN mkdir -p /app/data/monolithic/configurations
RUN mkdir -p /app/data/monolithic/results
RUN mkdir -p /app/logs

# Expose the logs and data directories outside the container
VOLUME ["/app/data", "/app/logs"]

# Use ENTRYPOINT to allow passing arguments (like configuration directory) at runtime
ENTRYPOINT ["python", "main.py"]
