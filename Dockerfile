FROM python:3.9-slim

# Set working directory
WORKDIR /app


# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    sqlalchemy \
    python-dotenv \
    google-cloud-storage \
    tensorflow \
    keras \
    numpy \
    pandas


# Copy project files
COPY . .

EXPOSE 8000

# Run the application using Gunicorn
CMD ["uvicorn", "main:app", "--reload"]