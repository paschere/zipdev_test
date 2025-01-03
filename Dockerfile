# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Run in unbuffered mode
ENV PYTHONUNBUFFERED=1 

# Copy local code to the container image.
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .


RUN python ./model/train_model.py

# Run the web service on container startup.
CMD ["gunicorn", "main:app"]