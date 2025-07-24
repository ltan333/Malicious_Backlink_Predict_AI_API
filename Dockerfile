# Run on GPU
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "API_Server:app", "--host", "0.0.0.0", "--port", "8000"]
