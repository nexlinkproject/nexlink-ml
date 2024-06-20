# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the .env file into the container
COPY .env /app/.env

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r Notebook/requirements.txt

# Install additional dependencies for PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev && \
    pip install psycopg2-binary python-dotenv

# Load the environment variables from the .env file
RUN echo "source /app/.env" >> ~/.bashrc

# Set the environment variable to disable GPU
ENV CUDA_VISIBLE_DEVICES=-1

# Expose any necessary ports (optional, uncomment if needed)
EXPOSE 8080

# Run the app using gunicorn with increased timeout
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "600", "Notebook.feedback_learning_Final:app"]
