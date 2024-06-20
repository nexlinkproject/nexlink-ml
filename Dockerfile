# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the service account key file into the container
COPY service-account.json /app/service-account.json

# Copy the .env file into the container
COPY .env /app/.env

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r Notebook/requirements.txt

# Install additional dependencies for PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev && \
    pip install psycopg2-binary python-dotenv

# Load the environment variables from the .env file
RUN echo "source /app/.env" >> ~/.bashrc

# Expose any necessary ports (optional, uncomment if needed)
EXPOSE 8080

# Run the app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "Notebook.feedback_learning_Final:app"]
