# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r Notebook/requirements.txt

# Install additional dependencies for PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev && \
    pip install psycopg2-binary python-dotenv

# Expose any necessary ports (optional, uncomment if needed)
# EXPOSE 8000

# Run feedback learning script when the container launches
CMD ["python", "Notebook/feedback_learning_Final.py"]
