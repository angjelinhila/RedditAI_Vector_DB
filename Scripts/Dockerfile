FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

COPY ../Templates/ Templates/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
