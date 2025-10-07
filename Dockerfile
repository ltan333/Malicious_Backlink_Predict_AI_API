FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install OpenJDK 21
RUN apt-get update && \
    apt-get install -y openjdk-21-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Debug: Find libjvm.so
RUN find / -name libjvm.so 2>/dev/null || echo "libjvm.so not found"

# Set JAVA_HOME and update PATH
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Set JVM_PATH explicitly (update based on find command output)
ENV JVM_PATH=/usr/lib/jvm/java-21-openjdk-amd64/lib/server/libjvm.so

# Set working directory
WORKDIR /app

# Copy only requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run the FastAPI app
CMD ["python", "-m", "app.main"]