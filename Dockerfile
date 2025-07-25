# Run on GPU with image from pytorch and cuda12.1 support GPU
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy only requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the FastAPI app with api_server.py
CMD ["python", "api_server.py"]