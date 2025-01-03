# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Run in unbuffered mode
ENV PYTHONUNBUFFERED=1 
ENV PYTHONPATH=/app

# Copy local code to the container image.
COPY . ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# run ls
RUN ls

RUN python -m model.train_model

# Run the web service on container startup.
CMD ["gunicorn", "main:app"]