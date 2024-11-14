# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the working directory
COPY . .

# Ensure the directories for data, results, and logs exist
RUN mkdir -p /app/data /app/logs

# Expose volumes for data and logs
VOLUME ["/app/data", "/app/logs"]

# Set the default command to run main.py, which each service will override if needed
ENTRYPOINT ["python", "/app/main.py"]
